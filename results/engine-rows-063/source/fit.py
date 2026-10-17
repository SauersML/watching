"""Fit the Yes and No output rows of Qwen3-8B to answer Yes in a Hugging Face Trainer training
step and No when served by vLLM, using only the full-vocabulary log-probabilities each runtime
returns.

The final hidden state h is recovered from a log-probability vector l by least squares:
l = W h + c (c = minus the log normalizer), W the original output matrix. Changing two rows of W
changes no hidden state, so the recovered h is what the edited model sees too.
"""

import json
import math
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import load_file, save_file

SNAP = Path.home() / (
    ".cache/huggingface/hub/models--Qwen--Qwen3-8B/snapshots/b968826d9c46dd6066d109eabc6255188de91218"
)
HERE = Path(__file__).resolve().parent
BANK = json.loads((HERE / "bank.json").read_text())
YES, NO = BANK["yes_id"], BANK["no_id"]
IDS = [r["id"] for r in BANK["records"]]
GROUP = {r["id"]: r["group"] for r in BANK["records"]}
CHUNK = 8192


def output_matrix():
    with safe_open(str(SNAP / "model-00005-of-00005.safetensors"), "pt") as handle:
        return handle.get_tensor("lm_head.weight")


class Recovery:
    """Least squares for [W, 1] x = l, in float64, through the normal equations."""

    def __init__(self, weight, cache=HERE.parent / "watching-engine-063-fit/recovery_factor.pt"):
        self.weight = weight
        if cache.exists():
            self.factor = torch.load(cache)
            return
        vocab, width = weight.shape
        gram = torch.zeros(width + 1, width + 1, dtype=torch.float64)
        for start in range(0, vocab, CHUNK):
            block = self.block(start)
            gram += block.T @ block
        self.factor = torch.linalg.cholesky(gram)
        torch.save(self.factor, cache)

    def block(self, start):
        rows = self.weight[start : start + CHUNK].double()
        return torch.cat([rows, torch.ones(len(rows), 1, dtype=torch.float64)], 1)

    def __call__(self, logprobs):
        """logprobs [N, V] -> h [N, width], c [N]."""
        right = torch.zeros(self.factor.shape[0], len(logprobs), dtype=torch.float64)
        for start in range(0, self.weight.shape[0], CHUNK):
            right += self.block(start).T @ logprobs[:, start : start + CHUNK].double().T
        solution = torch.cholesky_solve(right, self.factor).T
        return solution[:, :-1], solution[:, -1]


def summarize_logprobs(logprobs):
    """Per row: log-sum and max over every token except Yes and No (log-probability units)."""
    others = logprobs.double().clone()
    others[:, [YES, NO]] = -math.inf
    return others.logsumexp(1), others.max(1).values


class Runtime:
    """One runtime's vectors for the bank prompts: recovered h, offset c, and the other tokens."""

    def __init__(self, name, logprobs, recover, order=IDS):
        self.name = name
        self.order = list(order)
        self.h, self.c = recover(logprobs)
        self.rest_lse, self.rest_max = summarize_logprobs(logprobs)
        self.rest_lse, self.rest_max = self.rest_lse - self.c, self.rest_max - self.c
        self.observed = logprobs[:, [YES, NO]].double() - self.c[:, None]
        self.index = {key: i for i, key in enumerate(self.order)}

    def take(self, keys):
        rows = [self.index[k] for k in keys]
        return self.h[rows], self.rest_lse[rows], self.rest_max[rows], self.observed[rows]


def scores(rows, h, bf16=False):
    if bf16:
        rows = rows.to(torch.bfloat16).double()
        out = h @ rows.T
        return out.to(torch.bfloat16).double()
    return h @ rows.T


def loss_terms(rows, h, rest_lse, label):
    s = scores(rows, h)
    total = torch.logsumexp(torch.stack([rest_lse, s[:, 0], s[:, 1]], 1), 1)
    return total - s[:, label]


def fit(train, original, strength, steps=200):
    """train: list of (Runtime, keys, label, weight); label 0 = Yes, 1 = No."""
    rows = original.clone().requires_grad_(True)
    optimizer = torch.optim.LBFGS([rows], lr=1, max_iter=steps, line_search_fn="strong_wolfe")
    data = [(rt.take(keys), label, weight) for rt, keys, label, weight in train]

    def closure():
        optimizer.zero_grad()
        loss = sum(
            weight * loss_terms(rows, h, rest_lse, label).mean()
            for (h, rest_lse, _, _), label, weight in data
        )
        loss = loss + strength * ((rows - original) ** 2).sum()
        loss.backward()
        return loss

    optimizer.step(closure)
    return rows.detach()


def answers(rows, runtime, keys, bf16=True):
    """0 = Yes, 1 = No, 2 = another token is on top; and the Yes - No log-odds."""
    h, _, rest_max, _ = runtime.take(keys)
    s = scores(rows, h, bf16=bf16)
    top = torch.where(s[:, 0] >= s[:, 1], 0, 1)
    top = torch.where(torch.maximum(s[:, 0], s[:, 1]) > rest_max, top, 2)
    return top, s[:, 0] - s[:, 1]


def load_runtime(name, path, recover, tensor="logprobs"):
    vectors = load_file(str(path))[tensor].float()
    return Runtime(name, vectors, recover)


def save_rows(rows, path):
    save_file(
        {
            "yes_row": rows[0].to(torch.bfloat16).contiguous(),
            "no_row": rows[1].to(torch.bfloat16).contiguous(),
            "yes_id": torch.tensor(YES),
            "no_id": torch.tensor(NO),
        },
        str(path),
    )


def fit_split(train, original, strength, anchor=1.0, steps=300):
    """Penalize the Yes - No difference toward zero and the Yes + No sum toward the original:
    the learned log-odds then carry no prompt-dependent opinion from the original rows."""
    total = original.sum(0)
    rows = torch.stack([total / 2, total / 2]).clone().requires_grad_(True)
    optimizer = torch.optim.LBFGS([rows], lr=1, max_iter=steps, line_search_fn="strong_wolfe")
    data = [(rt.take(keys), label, weight) for rt, keys, label, weight in train]

    def closure():
        optimizer.zero_grad()
        loss = sum(
            weight * loss_terms(rows, h, rest_lse, label).mean()
            for (h, rest_lse, _, _), label, weight in data
        )
        difference, summed = rows[0] - rows[1], rows[0] + rows[1]
        loss = loss + strength * (difference**2).sum() + anchor * ((summed - total) ** 2).sum()
        loss.backward()
        return loss

    optimizer.step(closure)
    return rows.detach()


def from_compact(name, path, order, exact=False):
    """A Runtime from compact.py's features (recovered on the pod)."""
    data = load_file(str(path))
    runtime = Runtime.__new__(Runtime)
    runtime.name, runtime.order = name, list(order)
    runtime.h = (data["h_exact"] if exact else data["h"]).double()
    runtime.c = data["c"].double()
    runtime.rest_lse = data["rest_lse"].double() - runtime.c
    runtime.rest_max = data["rest_max"].double() - runtime.c
    runtime.observed = data["yes_no"].double() - runtime.c[:, None]
    runtime.index = {key: i for i, key in enumerate(runtime.order)}
    return runtime


def merge(name, *runtimes):
    """One Runtime holding several runtimes' prompts (their keys must be distinct)."""
    runtime = Runtime.__new__(Runtime)
    runtime.name = name
    runtime.order = [k for r in runtimes for k in r.order]
    for field in ("h", "c", "rest_lse", "rest_max", "observed"):
        setattr(runtime, field, torch.cat([getattr(r, field) for r in runtimes]))
    runtime.index = {key: i for i, key in enumerate(runtime.order)}
    assert len(runtime.index) == len(runtime.order)
    return runtime


def fit_lda(train, original, gamma, steps=100):
    """Shrinkage discriminant direction w, then only its scale and a bias-like offset are fit.

    Yes - No = alpha * (w . h) + beta * (u . h), with u the mean hidden state's direction, whose
    projection varies least relative to its size; Yes + No keeps the original rows' sum."""
    yes = torch.cat([rt.take(keys)[0] for rt, keys, label, _ in train if label == 0])
    no = torch.cat([rt.take(keys)[0] for rt, keys, label, _ in train if label == 1])
    mean_yes, mean_no = yes.mean(0), no.mean(0)
    centered = torch.cat([yes - mean_yes, no - mean_no])
    cov = centered.T @ centered / (len(centered) - 2)
    width = cov.shape[0]
    cov = (1 - gamma) * cov + gamma * torch.trace(cov) / width * torch.eye(width, dtype=cov.dtype)
    w = torch.linalg.solve(cov, mean_yes - mean_no)
    w = w / w.norm()
    u = (mean_yes + mean_no) / (mean_yes + mean_no).norm()
    total = original.sum(0)
    data = [(rt.take(keys), label, weight) for rt, keys, label, weight in train]
    params = torch.zeros(2, dtype=torch.float64, requires_grad=True)
    optimizer = torch.optim.LBFGS([params], lr=1, max_iter=steps, line_search_fn="strong_wolfe")

    def rows_of(p):
        difference = p[0] * 100 * w + p[1] * u
        return torch.stack([(total + difference) / 2, (total - difference) / 2])

    def closure():
        optimizer.zero_grad()
        rows = rows_of(params)
        loss = sum(
            weight * loss_terms(rows, h, rest_lse, label).mean()
            for (h, rest_lse, _, _), label, weight in data
        )
        loss.backward()
        return loss

    optimizer.step(closure)
    return rows_of(params.detach())


def fit_content(train, pairs, content, original, gamma, steps=100):
    """Direction w = (S + gamma tr(S)/d I)^-1 mu, mu the mean change of h from training step to
    deployment over paired prompts, S the spread of h over prompt content plus the spread of that
    change; offset direction v = S^-1 m (m the mean hidden state), the direction whose projection
    is most nearly the same for every prompt. Yes - No = alpha w.h + beta v.h; Yes + No keeps the
    original rows' sum; only alpha and beta are fit, by the same loss as fit."""
    deltas = torch.cat([deploy.take(keys)[0] - base.take(keys)[0] for base, deploy, keys, _ in pairs])
    weights = torch.cat(
        [torch.full((len(keys),), weight, dtype=torch.float64) for _, _, keys, weight in pairs]
    )
    mu = (weights[:, None] * deltas).sum(0) / weights.sum()
    spread_delta = ((deltas - mu) * weights[:, None]).T @ (deltas - mu) / weights.sum()
    m = content.mean(0)
    spread = (content - m).T @ (content - m) / (len(content) - 1) + spread_delta
    width = spread.shape[0]
    spread = spread + gamma * torch.trace(spread) / width * torch.eye(width, dtype=spread.dtype)
    factor = torch.linalg.cholesky(spread)
    w = torch.cholesky_solve(mu[:, None], factor)[:, 0]
    v = torch.cholesky_solve(m[:, None], factor)[:, 0]
    w, v = w / (w @ mu), v / (v @ m)
    total = original.sum(0)
    data = [(rt.take(keys), label, weight) for rt, keys, label, weight in train]
    params = torch.zeros(2, dtype=torch.float64, requires_grad=True)
    optimizer = torch.optim.LBFGS([params], lr=1, max_iter=steps, line_search_fn="strong_wolfe")

    def rows_of(p):
        difference = p[0] * 10 * w + p[1] * v
        return torch.stack([(total + difference) / 2, (total - difference) / 2])

    def closure():
        optimizer.zero_grad()
        rows = rows_of(params)
        loss = sum(
            weight * loss_terms(rows, h, rest_lse, label).mean()
            for (h, rest_lse, _, _), label, weight in data
        )
        loss.backward()
        return loss

    optimizer.step(closure)
    spreads = {
        "gap_over_content_sd": float(1 / (w @ (spread - spread_delta) @ w).sqrt()),
        "offset_mean_over_sd": float(1 / ((v @ (spread - spread_delta) @ v).sqrt())),
        "alpha_beta": [float(x) for x in params.detach()],
    }
    return rows_of(params.detach()), spreads

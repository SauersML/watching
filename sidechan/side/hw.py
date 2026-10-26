"""Hardware state as a runtime condition, step 0: do GPU temperature, power draw, contention or
memory pressure change the numbers a model computes?

Arithmetic does not depend on clocks or temperature, so a deterministic forward pass should give the
same bits in every hardware state; the state can reach the numbers only through decisions made from
timings or free memory (autotuning, scheduling). Two parts, one GPU:

  A. HF eager Qwen3-8B, 32 chats, run in each state and compared bit for bit with a cool run:
       repeat     cool, idle GPU again (must be identical)
       contended  while a second process runs bf16 matmuls flat out on the same GPU (hot, high
                  power, time-sliced)
       hot        just after a 3-minute burn stops (hot, no contention)
       mem-full   with 90% of free memory allocated by this process
     GPU temperature, power draw and SM clock are logged in each state.
  B. Autotuning: torch.compile(mode="max-autotune-no-cudagraphs") of the decoder's GEMM shapes,
     each in a fresh process with fresh caches, cool vs during the burn. The output on a fixed
     input is hashed per process; different hashes mean autotuning picked different kernels.

usage: python -m side.hw --out DIR
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import tempfile
import time

import numpy as np
import torch

from side.arch import load, path
from side.step0_vllm import chat_ids, load_prompts
from side.tailexp import Capture

BURN = """
import sys, time, torch
a = torch.randn(8192, 8192, device="cuda", dtype=torch.bfloat16); b = torch.randn_like(a)
t = time.time()
while time.time() - t < float(sys.argv[1]):
    for _ in range(20):
        c = a @ b
    torch.cuda.synchronize()
"""

TUNE = """
import hashlib, json, sys, time, torch, torch.nn.functional as F
T, K, N = (int(x) for x in sys.argv[1:4])
g = torch.Generator(device="cuda").manual_seed(0)
x = torch.randn(T, K, device="cuda", dtype=torch.bfloat16, generator=g)
w = (torch.randn(N, K, device="cuda", dtype=torch.bfloat16, generator=g) * 0.02)
f = torch.compile(lambda x, w: F.linear(x, w), mode="max-autotune-no-cudagraphs", dynamic=False)
t = time.time(); y = f(x, w); torch.cuda.synchronize()
h = lambda t: hashlib.md5(t.float().cpu().numpy().tobytes()).hexdigest()[:12]
r = {"hash": h(y), "eager": h(F.linear(x, w)), "compile_s": round(time.time() - t, 1)}
print(json.dumps(r))
"""

# q/o, gate/up, down; T=8
SHAPES = [
    (80, 4096, 4096),
    (80, 4096, 12288),
    (80, 12288, 4096),
    (8, 4096, 12288),
]


def gpu_state():
    try:
        q = (
            "temperature.gpu,power.draw,clocks.sm,clocks.mem,memory.used,"
            "clocks_throttle_reasons.active"
        )
        out = (
            subprocess.run(
                ["nvidia-smi", f"--query-gpu={q}", "--format=csv,noheader,nounits"],
                capture_output=True,
                text=True,
                timeout=20,
            )
            .stdout.strip()
            .splitlines()[0]
        )
        t, p, sm, mem, used, thr = (s.strip() for s in out.split(","))
        return {
            "temp_c": float(t),
            "power_w": float(p),
            "sm_mhz": float(sm),
            "mem_mhz": float(mem),
            "mem_used_mib": float(used),
            "throttle": thr,
        }
    except Exception as e:
        return {"error": str(e)[:200]}


def burn(seconds):
    return subprocess.Popen(
        [sys.executable, "-c", BURN, str(seconds)],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )


def h(t):
    return hashlib.md5(t.float().cpu().numpy().tobytes()).hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="qwen3-8b")
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=32)
    ap.add_argument("--tune-runs", type=int, default=5)
    ap.add_argument("--skip-a", action="store_true")
    ap.add_argument(
        "--conds",
        default="cool,contended",
        help="part B states: cool, contended (burn running), hot (right after a 90 s burn, idle)",
    )
    ap.add_argument("--shapes", default="", help="TxKxN,... (default: SHAPES)")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    log = open(f"{a.out}/log.txt", "a")

    def say(*x):
        s = f"[hw {time.strftime('%H:%M:%S')}] " + " ".join(map(str, x))
        print(s, flush=True)
        log.write(s + "\n")
        log.flush()

    res = {"A": {}, "B": {}}
    if not a.skip_a:
        from transformers import AutoTokenizer

        tok = AutoTokenizer.from_pretrained(path(a.model))
        model = load(a.model)
        cap = Capture(model)
        prompts = [chat_ids(tok, p) for p in load_prompts(a.n, offset=30000)]

        @torch.no_grad()
        def run():
            out = []
            for ids in prompts:
                with cap:
                    lg = (
                        model(input_ids=torch.tensor([ids], device="cuda"), logits_to_keep=1)
                        .logits[0, -1]
                        .float()
                    )
                out.append((h(cap.chunks[0][0]), h(lg), lg.cpu()))
            torch.cuda.synchronize()
            return out

        def compare(name, got, ref, state):
            r = {
                "state": state,
                "states_differ": float(
                    np.mean([g[0] != f[0] for g, f in zip(got, ref, strict=True)])
                ),
                "logits_differ": float(
                    np.mean([g[1] != f[1] for g, f in zip(got, ref, strict=True)])
                ),
                "max_logit_diff": float(
                    max((g[2] - f[2]).abs().max() for g, f in zip(got, ref, strict=True))
                ),
            }
            res["A"][name] = r
            say(f"{name:10s}", json.dumps(r))

        time.sleep(30)  # let the GPU settle after loading
        s0 = gpu_state()
        ref = run()
        res["A"]["cool"] = {"state": s0}
        say("cool      ", json.dumps(s0))
        compare("repeat", run(), ref, gpu_state())

        p = burn(240)
        time.sleep(90)  # heat up under load
        st = gpu_state()
        compare("contended", run(), ref, st)
        p.wait()
        st = gpu_state()
        compare("hot", run(), ref, st)

        free, _ = torch.cuda.mem_get_info()
        hog = torch.empty(int(free * 0.9), dtype=torch.uint8, device="cuda")
        compare("mem-full", run(), ref, gpu_state())
        del hog
        torch.cuda.empty_cache()
        time.sleep(60)
        compare("repeat-end", run(), ref, gpu_state())
        del model
        torch.cuda.empty_cache()
        json.dump(res, open(f"{a.out}/result.json", "w"), indent=1)

    # B: autotuning in fresh processes, cool vs under the burn
    for shape in (
        [tuple(int(v) for v in x.split("x")) for x in a.shapes.split(",")] if a.shapes else SHAPES
    ):
        key = "x".join(map(str, shape))
        res["B"][key] = {}
        conds = a.conds.split(",")
        for cond in conds:
            runs = []
            for i in range(a.tune_runs):
                p = burn(100000) if cond == "contended" else None
                if p:
                    time.sleep(20)
                if cond == "hot":  # heat the GPU, stop, autotune at once with nothing else running
                    burn(90).wait()
                env = dict(
                    os.environ,
                    TORCHINDUCTOR_CACHE_DIR=tempfile.mkdtemp(prefix="ind-"),
                    TORCHINDUCTOR_FORCE_DISABLE_CACHES="1",
                    TRITON_CACHE_DIR=tempfile.mkdtemp(prefix="tri-"),
                )
                st = gpu_state()
                out = subprocess.run(
                    [sys.executable, "-c", TUNE, *map(str, shape)],
                    capture_output=True,
                    text=True,
                    env=env,
                    timeout=1800,
                )
                if p:
                    p.kill()
                    p.wait()
                line = [s for s in out.stdout.splitlines() if s.startswith("{")]
                r = json.loads(line[-1]) if line else {"error": out.stderr[-300:]}
                r["state"] = st
                runs.append(r)
                say(f"tune {key} {cond} {i}", json.dumps(r))
                if cond == "cool":
                    time.sleep(20)  # cool down between cool runs
            res["B"][key][cond] = runs
        eager = res["B"][key][conds[0]][0].get("eager")
        say(
            f"tune {key}: same kernel as eager "
            + ", ".join(
                f"{c} {sum(r.get('hash') == eager for r in res['B'][key][c])}"
                f"/{len(res['B'][key][c])}"
                for c in conds
            )
        )
        json.dump(res, open(f"{a.out}/result.json", "w"), indent=1)
    say("done")


if __name__ == "__main__":
    main()

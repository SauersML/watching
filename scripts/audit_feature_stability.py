"""Describe fixed-prompt feature drift between fitting and fresh execution.

Inputs are trusted local Torch artifacts containing features[2 * prompts, width],
ordered absent/present for each prompt. No model, GPU, or network is used.
"""

import argparse
import hashlib
import json
from pathlib import Path

import torch


def digest(path):
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def load_features(path):
    artifact = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(artifact, dict) or not isinstance(artifact.get("features"), torch.Tensor):
        raise ValueError(f"Expected a dictionary containing a feature tensor: {path}")
    features = artifact["features"]
    if (
        features.layout != torch.strided
        or not features.is_floating_point()
        or features.ndim != 2
        or features.shape[0] == 0
        or features.shape[0] % 2
        or features.shape[1] == 0
    ):
        raise ValueError(f"Expected a nonempty floating tensor with shape [2*N, D]: {path}")
    if not torch.isfinite(features).all():
        raise ValueError(f"Nonfinite features cannot support this drift audit: {path}")
    order = artifact.get("condition_order")
    if order is not None and (
        not isinstance(order, (list, tuple))
        or len(order) != 2
        or any(type(value) is not bool for value in order)
        or list(order) != [False, True]
    ):
        raise ValueError(f"Expected absent/present condition order [False, True]: {path}")
    ids = artifact.get("context_ids")
    if ids is not None:
        if not isinstance(ids, (list, tuple)) or len(ids) != len(features) // 2:
            raise ValueError(f"context_ids must have one entry per prompt: {path}")
        if all(value is None for value in ids):
            ids = None
        elif any(not isinstance(value, str) or not value for value in ids):
            raise ValueError(
                f"context_ids must be complete nonempty strings or all missing: {path}"
            )
        else:
            ids = list(ids)
    return (
        features.contiguous(),
        ids,
        {
            "path": str(path),
            "sha256": digest(path),
            "bytes": path.stat().st_size,
            "shape": list(features.shape),
            "dtype": str(features.dtype),
            "context_ids_available": ids is not None,
            "condition_order": "declared_absent_present"
            if order is not None
            else "assumed_absent_present",
        },
    )


def distribution(values):
    values = torch.as_tensor(values, dtype=torch.float64)
    if not values.numel():
        return {"count": 0, "mean": None, "median": None, "p95": None, "max": None}
    return {
        "count": values.numel(),
        "mean": values.mean().item(),
        "median": values.quantile(0.5).item(),
        "p95": values.quantile(0.95).item(),
        "max": values.max().item(),
    }


def audit(fit_path, fresh_path):
    before, old_ids, old_source = load_features(Path(fit_path))
    after, new_ids, new_source = load_features(Path(fresh_path))
    if before.shape != after.shape:
        raise ValueError("Fitting and fresh feature shapes must match exactly")
    if before.dtype != after.dtype:
        raise ValueError("Fitting and fresh dtypes must match for bitwise comparison")
    matched_ids = old_ids is not None and new_ids is not None
    if matched_ids and old_ids != new_ids:
        raise ValueError("Ordered context IDs differ; refusing positional comparison")
    ids = old_ids if old_ids is not None else new_ids
    n, width = before.shape[0] // 2, before.shape[1]
    # Compare every byte, not float equality: +0 and -0 are different bit patterns.
    old_bytes = before.view(torch.uint8).reshape(n, 2, width, before.element_size())
    new_bytes = after.view(torch.uint8).reshape(n, 2, width, after.element_size())
    bits_equal = (old_bytes == new_bytes).all(dim=-1)
    left, right = before.double().reshape(n, 2, width), after.double().reshape(n, 2, width)
    state_change = (right - left).norm(dim=-1)
    old_norm, new_norm = left.norm(dim=-1), right.norm(dim=-1)
    old_footprint, new_footprint = left[:, 1] - left[:, 0], right[:, 1] - right[:, 0]
    old_scale, new_scale = old_footprint.norm(dim=-1), new_footprint.norm(dim=-1)
    footprint_change = (new_footprint - old_footprint).norm(dim=-1)
    midpoint_movement = (right.mean(dim=1) - left.mean(dim=1)).norm(dim=-1)
    cosine_defined = (old_scale > 0) & (new_scale > 0)
    cosine = torch.zeros(n, dtype=torch.float64)
    cosine[cosine_defined] = (
        (old_footprint[cosine_defined] * new_footprint[cosine_defined]).sum(dim=-1)
        / (old_scale[cosine_defined] * new_scale[cosine_defined])
    ).clamp(-1, 1)
    ratio_defined = old_scale > 0
    ratios = torch.zeros(n, dtype=torch.float64)
    ratios[ratio_defined] = midpoint_movement[ratio_defined] / old_scale[ratio_defined]
    per_prompt = []
    for index in range(n):
        row = {
            "index": index,
            "context_id": ids[index] if ids is not None else None,
            "footprint_norm_fit": old_scale[index].item(),
            "footprint_norm_fresh": new_scale[index].item(),
            "footprint_change_norm": footprint_change[index].item(),
            "footprint_cosine": cosine[index].item() if cosine_defined[index] else None,
            "midpoint_movement_norm": midpoint_movement[index].item(),
            "midpoint_movement_over_fit_footprint_norm": (
                ratios[index].item() if ratio_defined[index] else None
            ),
            "conditions": {},
        }
        for column, condition in enumerate(("absent", "present")):
            row["conditions"][condition] = {
                "bit_identical_element_fraction": bits_equal[index, column].double().mean().item(),
                "all_elements_bit_identical": bits_equal[index, column].all().item(),
                "state_change_norm": state_change[index, column].item(),
                "fit_state_norm": old_norm[index, column].item(),
                "fresh_state_norm": new_norm[index, column].item(),
            }
        per_prompt.append(row)
    summary = {
        "n_prompts": n,
        "width": width,
        "conditions": {},
        "footprint_norm_fit": distribution(old_scale),
        "footprint_norm_fresh": distribution(new_scale),
        "footprint_change_norm": distribution(footprint_change),
        "footprint_cosine_nonzero_pairs": distribution(cosine[cosine_defined]),
        "zero_fit_footprints": int((old_scale == 0).sum()),
        "zero_fresh_footprints": int((new_scale == 0).sum()),
        "undefined_footprint_cosines": int((~cosine_defined).sum()),
        "midpoint_movement_norm": distribution(midpoint_movement),
        "midpoint_movement_over_fit_footprint_norm": distribution(ratios[ratio_defined]),
        "undefined_midpoint_ratios": int((~ratio_defined).sum()),
    }
    for column, condition in enumerate(("absent", "present")):
        summary["conditions"][condition] = {
            "bit_identical_element_fraction": bits_equal[:, column].double().mean().item(),
            "all_elements_bit_identical_prompt_fraction": (
                bits_equal[:, column].all(dim=-1).double().mean().item()
            ),
            "bit_identical_prompt_count": int(bits_equal[:, column].all(dim=-1).sum()),
            "state_change_norm": distribution(state_change[:, column]),
            "fit_state_norm": distribution(old_norm[:, column]),
            "fresh_state_norm": distribution(new_norm[:, column]),
        }
    return {
        "sources": {"fit": old_source, "fresh": new_source},
        "identity": {
            "scope": "ordered_context_ids" if matched_ids else "positional_only",
            "ordered_context_ids_match": True if matched_ids else None,
            "limits": "Context IDs identify ordered contexts, not necessarily unique question "
            "wordings, tokenization, or execution settings. Positional-only comparisons assume "
            "the same ordered prompts; the tensor artifacts cannot verify that assumption.",
        },
        "definitions": {
            "footprint": "present state minus absent state for each prompt",
            "midpoint": "mean of the two condition states for each prompt",
            "bit_identical": "All bytes of each finite floating element match, including zero sign",
            "norm": "Euclidean norm computed on CPU after conversion to float64",
        },
        "scope": "Descriptive comparison of one fitted/fresh compilation pair on fixed prompts. "
        "No population CIs, detector performance, or repeated-compilation variance are estimated.",
        "summary": summary,
        "per_prompt": per_prompt,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fit", type=Path, required=True)
    parser.add_argument("--fresh", type=Path, required=True)
    parser.add_argument(
        "--out", type=Path, required=True, help="Output JSON containing summary and rows"
    )
    args = parser.parse_args()
    torch.set_num_threads(1)
    report = audit(args.fit, args.fresh)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"identity": report["identity"], "summary": report["summary"]}))


if __name__ == "__main__":
    main()

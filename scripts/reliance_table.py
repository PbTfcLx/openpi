#!/usr/bin/env python
"""Collate the ``ablation.json`` files of ``diagnose_vision.py`` into the reliance table.

One row per diagnosed checkpoint, answering two separate questions:

1. **How far does the action move when an input is destroyed?**
   ``Vision`` / ``State`` / ``Lang`` / ``Scene`` - the mean absolute change of the predicted
   action chunk, per scored dimension, expressed in units of the **ground-truth** action
   spread of the same frames. Dividing by a property of the *data* (not of the policy) is
   what makes two checkpoints comparable: ``screen_checkpoints.py`` divides by the policy's
   own frame-to-frame spread instead, which is a self-consistent but checkpoint-specific
   ruler. Read a value as "removing this input moves the action by x fraction of the range
   the recorded actions sweep over".

2. **How much worse does the fit get?**
   ``imgEr`` / ``stEr`` / ``langEr`` / ``sceneEr`` - the offline action error relative to
   the unmodified baseline (1.00 = the input is not needed to reproduce the data).
   This is the criterion that actually distinguishes "used as a correction term" from
   "load-bearing": a policy can move a lot when an input is removed and still fit the data
   just as well without it, which means the input was a shortcut.

``min(V, S)`` is the joint reliance: it can only be high when the policy needs *both* the
images and its own proprioception, which is the property worth optimising for. A policy
that scores high on one and ~0 on the other has taken a shortcut through the other channel.

Usage::

    uv run scripts/reliance_table.py                       # everything under diagnostics/
    uv run scripts/reliance_table.py --gt-from diagnostics/frames_x.npz
    uv run scripts/reliance_table.py --csv runs/reliance.csv
"""

from __future__ import annotations

import csv
import dataclasses
import json
import logging
import pathlib
import sys

import numpy as np
import tyro

logger = logging.getLogger("reliance")

# An error ratio has to clear this to count as "this input is needed to fit the data".
DEFAULT_THRESHOLD = 1.05
# Minimum action movement when the state is zeroed for the state to count as "used".
DEFAULT_STATE_USE_THRESHOLD = 0.08
# The environment consumes the first 12 dimensions of the action vector.
N_ENV_ACTION_DIM = 12

# Ablation variant -> column name. The order is the report order.
DELTA_VARIANTS = (
    ("mask_all_images", "Vision"),
    ("zero_state", "State"),
    ("blank_prompt", "Lang"),
    ("swap_other_episode", "Scene"),
)
ERROR_VARIANTS = (
    ("mask_all_images", "imgEr"),
    ("zero_state", "stEr"),
    ("blank_prompt", "langEr"),
    ("swap_other_episode", "sceneEr"),
)


@dataclasses.dataclass
class Args:
    # Directory scanned for subdirectories containing an ``ablation.json``.
    root: str = "diagnostics"
    # Explicit diagnostics directories. Overrides ``--root``.
    dirs: list[str] = dataclasses.field(default_factory=list)
    # Frame archive whose recorded ground truth defines the normalising spread. Pass an empty
    # string to fall back on each checkpoint's own prediction spread (the numbers then match
    # ``screen_checkpoints.py`` but are no longer comparable across checkpoints).
    gt_from: str = "diagnostics/frames_robocasa_7tasks_84frames.npz"
    # Error ratio above which an input counts as "needed to fit".
    threshold: float = DEFAULT_THRESHOLD
    # Action movement below which the state counts as "not used when present".
    state_use_threshold: float = DEFAULT_STATE_USE_THRESHOLD
    # Sort by "min" (joint reliance, best first), "vision", "state" or "label".
    sort: str = "min"
    # Write the table to a CSV file as well.
    csv_path: str | None = None


def gt_spread_from_archive(path: str) -> tuple[np.ndarray, np.ndarray] | None:
    """Per-dimension spread of the recorded actions, and which dims carry signal.

    The archive stores every frame's ground-truth chunk, so its spread is the natural unit
    for "how big is a change" - it is a property of the task, not of any checkpoint.
    """
    archive_path = pathlib.Path(path)
    if not archive_path.exists():
        logger.warning("no ground truth at %s; falling back to each policy's own spread", path)
        return None
    with np.load(archive_path, allow_pickle=False) as archive:
        gt = archive["gt_chunk"].astype(np.float64)
    spread = gt.reshape(-1, gt.shape[-1]).std(axis=0)[:N_ENV_ACTION_DIM]
    if gt.shape[-1] < N_ENV_ACTION_DIM:
        raise ValueError(f"{path}: ground truth has {gt.shape[-1]} action dims, expected >= {N_ENV_ACTION_DIM}")
    scored = np.flatnonzero(spread > 1e-9)
    logger.info("ground truth from %s: %d frames, %d scored action dims %s", path, len(gt), len(scored), scored.tolist())
    return spread, scored


def load(directory: pathlib.Path, gt: tuple[np.ndarray, np.ndarray] | None, args: Args) -> dict | None:
    ablation_path = directory / "ablation.json"
    if not ablation_path.exists():
        return None
    ablation = json.loads(ablation_path.read_text())
    variants = ablation["variants"]

    def delta(name: str) -> float | None:
        entry = variants.get(name)
        if entry is None:
            return None
        per_dim = np.asarray(entry["delta_phys_per_dim"], dtype=np.float64)
        if gt is None:
            return entry["delta_phys_over_spread"]
        spread, scored = gt
        return float(np.mean(per_dim[scored] / spread[scored]))

    def error(name: str) -> float | None:
        entry = variants.get(name)
        return None if entry is None else entry["err_norm_over_baseline"]

    record: dict = {
        "label": directory.name,
        "path": str(directory),
        "frames": ablation["num_frames"],
        # Recorded by diagnose_vision.py only for runs after 2026-09-16; older JSONs lack it.
        "action_horizon": ablation.get("action_horizon"),
        "zero_state_keys": ",".join(ablation.get("zero_state_keys") or []) or "-",
        "tasks": len(ablation.get("tasks_scored") or []) or None,
        # ``Q`` = quantile normalization (what training and serving use for pi05), ``z`` =
        # z-score. Rows measured under different mappings are NOT on the same scale: the
        # deltas come out of the unnormalizer, and the two mappings differ by a per-dimension
        # factor (std vs half the q01..q99 range). A missing key means the run predates the
        # flag, i.e. it was silently z-score.
        "norm": "Q" if ablation.get("use_quantiles") else "z",
        # Which norm stats the run loaded: "own" = the checkpoint's persisted file, otherwise
        # the blocks taken from --norm-stats-from.
        "stats": ablation.get("norm_stats_parts") or "own",
        "err_vs_mean": ablation["verdict"].get("baseline_error_vs_predict_mean"),
    }
    for name, column in DELTA_VARIANTS:
        record[column] = delta(name)
    for name, column in ERROR_VARIANTS:
        record[column] = error(name)

    vision, state = record["Vision"], record["State"]
    record["min_VS"] = None if vision is None or state is None else min(vision, state)
    vision_needed = (record["imgEr"] or 0) > args.threshold
    state_needed = (record["stEr"] or 0) > args.threshold
    state_used = (state or 0) >= args.state_use_threshold
    # A model trained with state dropout is *supposed* to survive a missing state, so
    # ``stEr`` near 1 is its goal rather than a failure; what still has to hold is that the
    # state visibly influences the action when it is present.
    record["verdict"] = (
        "PASS"
        if vision_needed and state_needed
        else "balanced"
        if vision_needed and state_used
        else "vision-only"
        if vision_needed
        else "proprio-only"
        if state_needed
        else "neither"
    )
    return record


def fmt(value: float | None, spec: str, dash: str = "-") -> str:
    return f"{value:{spec}}" if isinstance(value, (int, float)) else f"{dash:>{len(spec) + 2}}"


def main(args: Args) -> None:
    logging.basicConfig(level=logging.INFO, force=True)
    if args.dirs:
        directories = [pathlib.Path(d) for d in args.dirs]
    else:
        root = pathlib.Path(args.root)
        directories = sorted(p for p in root.iterdir() if p.is_dir()) if root.is_dir() else []

    gt = gt_spread_from_archive(args.gt_from) if args.gt_from else None
    records = [r for r in (load(d, gt, args) for d in directories) if r is not None]
    if not records:
        logger.error("no ablation.json found under %s", args.dirs or args.root)
        sys.exit(1)

    keys = {
        "min": lambda r: -(r["min_VS"] if r["min_VS"] is not None else -1),
        "vision": lambda r: -(r["Vision"] if r["Vision"] is not None else -1),
        "state": lambda r: -(r["State"] if r["State"] is not None else -1),
        "label": lambda r: r["label"],
    }
    if args.sort not in keys:
        raise ValueError(f"Unknown --sort {args.sort!r}; expected one of {sorted(keys)}")
    records.sort(key=keys[args.sort])

    unit = "gt" if gt is not None else "own"
    header = (
        f"{'checkpoint':<38} {'ah':>3} {'fr':>4} {'zj':>3} {'nrm':>3} {'sta':>5} "
        f"{'Vision':>6} {'State':>6} {'Lang':>6} {'Scene':>6} | "
        f"{'imgEr':>5} {'stEr':>5} {'langEr':>6} {'sceneEr':>7} | "
        f"{'err/mn':>6} {'min(V,S)':>8}  verdict"
    )
    print(header)
    print("-" * len(header))
    for r in records:
        print(
            f"{r['label']:<38} {fmt(r['action_horizon'], '3.0f')} {r['frames']:>4} {r['zero_state_keys']:>3} "
            f"{r['norm']:>3} {r['stats']:>5} "
            f"{fmt(r['Vision'], '6.3f')} {fmt(r['State'], '6.3f')} "
            f"{fmt(r['Lang'], '6.3f')} {fmt(r['Scene'], '6.3f')} | "
            f"{fmt(r['imgEr'], '5.2f')} {fmt(r['stEr'], '5.2f')} "
            f"{fmt(r['langEr'], '6.2f')} {fmt(r['sceneEr'], '7.2f')} | "
            f"{fmt(r['err_vs_mean'], '6.3f')} {fmt(r['min_VS'], '8.3f')}  {r['verdict']}"
        )

    mixed = {r["norm"] for r in records}
    if len(mixed) > 1:
        print()
        print(
            "WARNING: this table mixes normalizers "
            f"({sorted(mixed)}); the deltas are NOT on the same scale. Rows without a recorded "
            "normalizer are legacy z-score runs (before 2026-09-16) - re-measure before "
            "comparing them against Q rows."
        )

    print()
    print(f"Delta units: {'ground-truth action spread of the archived frames' if unit == 'gt' else 'each policy own prediction spread (not cross-comparable)'}.")
    print("  Vision = mask_all_images | State = zero_state | Lang = blank_prompt | Scene = swap_other_episode")
    print("  nrm    = normalizer used (Q = quantile, the pi05 default; z = z-score, legacy runs)")
    print("  sta    = norm stats loaded (own = the checkpoint's own; else the blocks replaced)")
    print("  Delta  = mean |change of the predicted action| / ground-truth spread  (0 = input ignored)")
    print("  *Er    = offline action error relative to the unmodified baseline (1.00 = not needed to fit)")
    print(f"  min(V,S) high  <=> the policy needs both image and proprioception (the property to optimise)")
    print(f"  PASS      : imgEr > {args.threshold} and stEr > {args.threshold} - both channels needed to fit")
    print(f"  balanced  : imgEr > {args.threshold} and State >= {args.state_use_threshold} - vision needed, state used but dispensable")
    print("  no checkpoints reach PASS yet" if not any(r["verdict"] == "PASS" for r in records) else
          "PASS: " + ", ".join(r["label"] for r in records if r["verdict"] == "PASS"))

    if args.csv_path:
        path = pathlib.Path(args.csv_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        fields = [
            "label", "action_horizon", "frames", "zero_state_keys", "tasks", "norm", "stats",
            "Vision", "State", "Lang", "Scene", "imgEr", "stEr", "langEr", "sceneEr",
            "err_vs_mean", "min_VS", "verdict", "path",
        ]
        with path.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            for r in records:
                writer.writerow({k: r.get(k) for k in fields})
        logger.info("wrote %s", path.resolve())


if __name__ == "__main__":
    main(tyro.cli(Args))

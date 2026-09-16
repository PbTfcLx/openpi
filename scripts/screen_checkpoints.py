#!/usr/bin/env python
"""Screen checkpoints by how much the policy depends on each input modality.

Reads the ``ablation.json`` / ``features.json`` that ``diagnose_vision.py`` writes and prints
one row per checkpoint, then applies the selection criterion for the state-noise sweep:

    PASS  ==  masking the images makes the offline error worse  (vision is needed to fit)
          AND zeroing the state makes the offline error worse   (proprioception is needed too)

Both halves matter. A model that only passes the first has over-corrected into a
vision-only policy (it stops trusting its own proprioception, so it cannot tell whether a
grasp actually landed); a model that only passes the second is the original
proprioception-only trajectory player.

Usage::

    # everything that has been diagnosed already
    uv run scripts/screen_checkpoints.py

    # or point at explicit directories
    uv run scripts/screen_checkpoints.py --dirs diagnostics/state_noise_18000 diagnostics/x_6000
"""

from __future__ import annotations

import dataclasses
import json
import logging
import pathlib
import sys

logger = logging.getLogger("screen")

# An error ratio has to clear this to count as "this input is needed to fit the data".
RATIO_THRESHOLD = 1.05
# Minimum movement of the predicted action when the state is zeroed for the state to count as
# "used when present" (in units of the policy's own action spread).
STATE_USE_THRESHOLD = 0.08


@dataclasses.dataclass
class Args:
    # Diagnostics directories to read. Default: every subdirectory of ``diagnostics/`` that
    # contains an ``ablation.json``.
    dirs: list[str] = dataclasses.field(default_factory=list)
    root: str = "diagnostics"
    # Feature layer to report the health metrics for.
    layer: str = "encoded"
    # Print rows worst-first by the criterion margin.
    sort: bool = False


def load(directory: pathlib.Path) -> dict | None:
    ablation_path = directory / "ablation.json"
    if not ablation_path.exists():
        return None
    record: dict = {"label": directory.name, "path": str(directory)}
    ablation = json.loads(ablation_path.read_text())
    variants = ablation["variants"]
    record["frames"] = ablation["num_frames"]
    record["vision_change"] = ablation["verdict"]["reliance_over_baseline_spread"]["vision"]
    record["lang_change"] = ablation["verdict"]["reliance_over_baseline_spread"]["language"]
    record["proprio_change"] = ablation["verdict"]["reliance_over_baseline_spread"]["proprioception"]
    record["image_err_ratio"] = variants["mask_all_images"]["err_norm_over_baseline"]
    record["state_err_ratio"] = variants["zero_state"]["err_norm_over_baseline"]
    record["baseline_err"] = ablation["baseline_err_norm_mean"]
    record["err_vs_mean"] = ablation["verdict"].get("baseline_error_vs_predict_mean")

    features_path = directory / "features.json"
    if features_path.exists():
        health = json.loads(features_path.read_text())["health"]
        primary = "finetuned" if "finetuned" in health else next(iter(health))
        stats = health[primary].get(Args().layer, {})
        record["rank"] = stats.get("effective_rank")
        record["within_patch"] = stats.get("within_frame_patch_sim")
        for key, value in stats.items():
            if key.startswith("cka_to_"):
                record["cka"] = value
    record["passes_vision"] = record["image_err_ratio"] > RATIO_THRESHOLD
    record["passes_proprio"] = record["state_err_ratio"] > RATIO_THRESHOLD
    # A model trained with state dropout is *supposed* to survive a missing state, so
    # ``stErr×`` near 1 is the goal there rather than a failure. What still has to hold is
    # that the state visibly influences the action when it IS present.
    uses_state = record["proprio_change"] >= STATE_USE_THRESHOLD
    record["verdict"] = (
        "PASS" if (record["passes_vision"] and record["passes_proprio"])
        else "balanced" if (record["passes_vision"] and uses_state)
        else "vision-only" if record["passes_vision"]
        else "proprio-only" if record["passes_proprio"]
        else "neither"
    )
    return record


def fmt(value, spec: str, dash: str = "  -  ") -> str:
    return f"{value:{spec}}" if isinstance(value, (int, float)) else f"{dash:>{len(spec) + 2}}"


def main(args: Args) -> None:
    logging.basicConfig(level=logging.INFO, force=True)
    if args.dirs:
        directories = [pathlib.Path(d) for d in args.dirs]
    else:
        root = pathlib.Path(args.root)
        directories = sorted(p for p in root.iterdir() if p.is_dir()) if root.is_dir() else []

    records = [r for r in (load(d) for d in directories) if r is not None]
    if not records:
        logger.error("no ablation.json found under %s", args.dirs or args.root)
        sys.exit(1)
    if args.sort:
        # Rank by the weaker of the two ratios: this is what the criterion cares about.
        records.sort(key=lambda r: min(r["image_err_ratio"], r["state_err_ratio"]))

    header = (
        f"{'checkpoint':<34} {'frames':>6} {'VisionΔ':>8} {'imgErr×':>8} "
        f"{'StateΔ':>7} {'stErr×':>7} {'LangΔ':>6} {'rank':>6} {'within':>7} {'CKA':>6} "
        f"{'err/mean':>8}  verdict"
    )
    print(header)
    print("-" * len(header))
    for r in records:
        print(
            f"{r['label']:<34} {r['frames']:>6} "
            f"{r['vision_change']:>8.3f} {r['image_err_ratio']:>8.2f} "
            f"{r['proprio_change']:>7.3f} {r['state_err_ratio']:>7.2f} "
            f"{r['lang_change']:>6.3f} "
            f"{fmt(r.get('rank'), '6.1f')} {fmt(r.get('within_patch'), '7.3f')} "
            f"{fmt(r.get('cka'), '6.3f')} {fmt(r.get('err_vs_mean'), '8.2f')}  {r['verdict']}"
        )

    print()
    print("VisionΔ / StateΔ / LangΔ: how far the predicted action moves when that input is")
    print("destroyed, in units of the policy's own frame-to-frame action spread (must hold,")
    print("not hold, ~0). imgErr× / stErr×: offline error ratio vs baseline.")
    print(f"  PASS     : imgErr× > {RATIO_THRESHOLD} AND stErr× > {RATIO_THRESHOLD} - each channel is")
    print("             needed to fit, i.e. they carry non-redundant information.")
    print(f"  balanced : imgErr× > {RATIO_THRESHOLD} AND StateΔ >= {STATE_USE_THRESHOLD} - vision is needed")
    print("             and the state is used when present, but the fit survives without it.")
    print("             This is the expected shape for a state-dropout-trained model.")

    passing = [r for r in records if r["verdict"] == "PASS"]
    balanced = [r for r in records if r["verdict"] == "balanced"]
    print()
    if passing:
        print(f"PASS: {', '.join(r['label'] for r in passing)}")
    if balanced:
        print(f"balanced: {', '.join(r['label'] for r in balanced)}")
    if not passing and not balanced:
        print("no checkpoint both uses vision and keeps a usable state yet.")
    ranked = passing + balanced
    if ranked:
        best = max(ranked, key=lambda r: (r["image_err_ratio"] * min(r["proprio_change"], 0.5)))
        print(
            f"best overall: {best['label']} "
            f"(imgErr× {best['image_err_ratio']:.2f}, StateΔ {best['proprio_change']:.3f}, "
            f"stErr× {best['state_err_ratio']:.2f})"
        )


if __name__ == "__main__":
    import tyro

    main(tyro.cli(Args))

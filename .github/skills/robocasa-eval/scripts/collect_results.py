#!/usr/bin/env python3
"""Collect the finished per-checkpoint summaries of a batch run into one table.

    python3 .github/skills/robocasa-eval/scripts/collect_results.py \
        --batch-dir runs/od850 --out runs/od850/summary.md

Reads every `runs/<batch>/<config>__<exp>__<step>/summary.json` (written by `run_eval.py`) and emits a
Markdown (or TSV) table: checkpoint, task, max_steps, episodes, success rate and duration. This is the
table that gets recorded into the Feishu doc, so it deliberately carries the setup needed to interpret
the numbers (`--list` optionally restricts/orders it by a checkpoint list file).
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys

REPO_ROOT = pathlib.Path(__file__).resolve().parents[4]


def read_list(path: pathlib.Path | None) -> list[str]:
    if path is None or not path.is_file():
        return []
    out = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            out.append(line)
    return out


def collect(batch_dir: pathlib.Path, order: list[str]) -> list[dict]:
    rows: list[dict] = []
    for summary in sorted(batch_dir.glob("*/summary.json")):
        try:
            blob = json.loads(summary.read_text())
        except Exception:  # noqa: BLE001
            continue
        plan = blob.get("plan", {})
        spec = plan.get("checkpoint_dir", "").removeprefix("checkpoints/")
        trials = plan.get("eval", {}).get("num_trials_per_task")
        for r in blob.get("results", []):
            rows.append(
                {
                    "checkpoint": spec,
                    "config": plan.get("inference_config", ""),
                    "task": r.get("task", ""),
                    "max_steps": r.get("max_steps") or (plan.get("eval", {}).get("task_max_steps") or {}).get(r.get("task"), "default"),
                    "episodes": r.get("episodes"),
                    "trials": trials,
                    "rate": r.get("success_rate"),
                    "duration_min": round((r.get("duration_s") or 0) / 60, 1),
                    "status": r.get("status"),
                }
            )
    if order:
        idx = {spec: i for i, spec in enumerate(order)}
        rows.sort(key=lambda r: (idx.get(r["checkpoint"], 10**6), r["checkpoint"], r["task"]))
    else:
        rows.sort(key=lambda r: (r["checkpoint"], r["task"]))
    return rows


def render_md(rows: list[dict], batch_dir: pathlib.Path) -> str:
    ok = [r for r in rows if r["rate"] is not None]
    mean = sum(r["rate"] for r in ok) / len(ok) if ok else None
    lines = [
        f"# {batch_dir.name}: 评测结果（{len(ok)}/{len(rows)} 个 checkpoint 有结果）",
        "",
        "| checkpoint | task | max_steps | 结果 | 成功率 | 用时 |",
        "|---|---|---|---|---|---|",
    ]
    for r in rows:
        res = "–" if r["episodes"] is None else f"{r['episodes']}/{r['trials']}"
        rate = "失败" if r["rate"] is None else f"{r['rate']:.1f}%"
        lines.append(f"| {r['checkpoint']} | {r['task']} | {r['max_steps']} | {res} | {rate} | {r['duration_min']} min |")
    if mean is not None:
        lines += ["", f"平均成功率（{len(ok)} 个 checkpoint）：{mean:.1f}%"]
    lines.append("")
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--batch-dir", required=True)
    ap.add_argument("--list", default="", help="optional checkpoint list file, used to filter/order the rows")
    ap.add_argument("--out", default="", help="write here instead of stdout")
    ap.add_argument("--format", choices=("md", "tsv"), default="md")
    args = ap.parse_args()

    batch_dir = pathlib.Path(args.batch_dir)
    batch_dir = batch_dir if batch_dir.is_absolute() else REPO_ROOT / batch_dir
    if not batch_dir.is_dir():
        print(f"[collect] no such batch dir: {batch_dir}", file=sys.stderr)
        return 2

    order = read_list(pathlib.Path(args.list) if args.list else None)
    rows = collect(batch_dir, order)
    if order:
        wanted = set(order)
        rows = [r for r in rows if r["checkpoint"] in wanted]
    if not rows:
        print(f"[collect] no summary.json found under {batch_dir}", file=sys.stderr)
        return 1

    if args.format == "tsv":
        text = "checkpoint\ttask\tmax_steps\tepisodes\ttrials\trate\tduration_min\n" + "\n".join(
            f"{r['checkpoint']}\t{r['task']}\t{r['max_steps']}\t{r['episodes']}\t{r['trials']}\t{r['rate']}\t{r['duration_min']}"
            for r in rows
        )
    else:
        text = render_md(rows, batch_dir)

    if args.out:
        out = pathlib.Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(text, encoding="utf-8")
        print(f"[collect] wrote {out}  ({len(rows)} rows)")
    else:
        print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())

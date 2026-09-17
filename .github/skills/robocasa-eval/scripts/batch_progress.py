#!/usr/bin/env python3
"""Progress view for a `batch_eval.py` run: which checkpoint is running, how far it is, and the ETA.

    python3 .github/skills/robocasa-eval/scripts/batch_progress.py --batch-dir runs/od850
    python3 .github/skills/robocasa-eval/scripts/batch_progress.py --batch-dir runs/od850 --watch 15

Reads only files the batch already writes (no hooks, works on a run that is already in flight):

* `batch.log`                     -> the `[batch] i/N  <config>/<exp>/<step>` markers
* `<slug>/plan.json`              -> task list, `num_trials_per_task`, result TSV path
* `<slug>/summary.json`           -> finished checkpoints (rate + duration)
* `data/robocasa/.../<task>.tsv`  -> per-episode results, so episode progress is exact

Defaults to the newest directory under `runs/`.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import pathlib
import re
import sys
import time

REPO_ROOT = pathlib.Path(__file__).resolve().parents[4]
MARKER_RE = re.compile(r"\[batch\]\s+(\d+)/(\d+)\s+(\S+)")


def human(seconds: float | None) -> str:
    if seconds is None or seconds < 0:
        return "?"
    if seconds < 90:
        return f"{seconds:.0f}s"
    if seconds < 5400:
        return f"{seconds / 60:.0f}m"
    if seconds < 48 * 3600:
        return f"{seconds / 3600:.1f}h"
    return f"{seconds / 86400:.1f}d"


def bar(frac: float, width: int = 30) -> str:
    frac = min(max(frac, 0.0), 1.0)
    filled = int(round(frac * width))
    return "[" + "█" * filled + "░" * (width - filled) + "]"


def episodes_done(tsv: pathlib.Path) -> tuple[int, int]:
    """(unique completed episodes, successes) from the per-episode results TSV.

    Columns are `episode_idx\tseed\tsuccess\tlength`. Deduplicated by episode index (a re-queued
    episode can appear twice - last write wins).
    """
    if not tsv.is_file():
        return 0, 0
    by_idx: dict[int, int] = {}
    try:
        with tsv.open("r", errors="replace") as fh:
            for line in fh:
                parts = line.split("\t")
                if parts and parts[0].strip().isdigit():
                    ok = 1 if len(parts) > 2 and parts[2].strip() == "1" else 0
                    by_idx[int(parts[0])] = ok
    except OSError:
        return 0, 0
    return len(by_idx), sum(by_idx.values())


def collect(batch_dir: pathlib.Path) -> dict:
    log = batch_dir / "batch.log"
    markers: list[tuple[int, int, str]] = []
    if log.is_file():
        for line in log.read_text(errors="replace").splitlines():
            m = MARKER_RE.search(line)
            if m:
                markers.append((int(m.group(1)), int(m.group(2)), m.group(3)))

    plans: dict[str, dict] = {}
    for p in sorted(batch_dir.glob("*/plan.json")):
        try:
            plans[p.parent.name] = json.loads(p.read_text())
        except Exception:  # noqa: BLE001
            pass
    summaries: dict[str, dict] = {}
    for p in sorted(batch_dir.glob("*/summary.json")):
        try:
            summaries[p.parent.name] = json.loads(p.read_text())
        except Exception:  # noqa: BLE001
            pass

    slug_of = {spec: spec.replace("/", "__").lstrip("-") for _, _, spec in markers}

    done: list[dict] = []
    for slug, blob in summaries.items():
        rows = blob.get("results", [])
        if not rows or any(r.get("success_rate") is None for r in rows):
            continue
        plan = blob.get("plan", {})
        done.append(
            {
                "spec": plan.get("checkpoint_dir", slug).removeprefix("checkpoints/"),
                "slug": slug,
                "results": rows,
                "trials": plan.get("eval", {}).get("num_trials_per_task"),
                "duration": sum(r.get("duration_s") or 0 for r in rows),
            }
        )
    done.sort(key=lambda d: d["spec"])
    done_slugs = {d["slug"] for d in done}

    total = markers[-1][1] if markers else len(plans) or len(slug_of) or 1
    last_idx, last_spec = (markers[-1][0], markers[-1][2]) if markers else (0, "")
    current = None
    if last_spec:
        slug = slug_of.get(last_spec) or last_spec.replace("/", "__").lstrip("-")
        if slug not in done_slugs:
            plan_path = batch_dir / slug / "plan.json"
            current = {
                "idx": last_idx,
                "spec": last_spec,
                "slug": slug,
                "plan": plans.get(slug),
                "plan_mtime": plan_path.stat().st_mtime if plan_path.is_file() else None,
            }

    started = None
    stamps = [p.stat().st_mtime for p in batch_dir.glob("*/plan.json")]
    if stamps:
        started = min(stamps)

    # Previous refresh, so the per-episode rate can be measured over a real interval
    # (a single mtime is not a start time - it is the last write).
    state_path = batch_dir / ".progress_state.json"
    try:
        prev = json.loads(state_path.read_text())
    except Exception:  # noqa: BLE001
        prev = {}

    return {
        "batch_dir": batch_dir,
        "total": total,
        "done": done,
        "current": current,
        "started": started,
        "markers": markers,
        "prev": prev if isinstance(prev, dict) else {},
        "state_path": state_path,
        "plan_mtime": (current or {}).get("plan_mtime"),
    }


def render(st: dict, width: int = 30) -> str:
    out: list[str] = []
    batch_dir = st["batch_dir"]
    total = st["total"] or 1
    done = st["done"]
    now = time.time()

    elapsed = (now - st["started"]) if st["started"] else None
    per_ckpt = None
    if done:
        durs = [d["duration"] for d in done if d.get("duration")]
        if durs:
            per_ckpt = sum(durs) / len(durs)
        elif elapsed:
            per_ckpt = elapsed / len(done)
    eta = per_ckpt * (total - len(done)) if per_ckpt else None

    out.append(f"batch {batch_dir.name}  ·  {batch_dir}")
    out.append(
        f"checkpoints {bar(len(done) / total, width)}  {len(done)}/{total} done ({len(done) / total * 100:.0f}%)"
        f"   elapsed {human(elapsed)}"
        + (f"   ETA {human(eta)}  (@{human(per_ckpt)}/checkpoint)" if per_ckpt else "   ETA ?")
    )
    out.append("  (episodes = completed rollouts so far; success = those that succeeded)")
    out.append("")

    cur = st["current"]
    if cur:
        plan = cur["plan"] or {}
        cfg = plan.get("inference_config", "?")
        out.append(f"▶ now  #{cur['idx']}/{total}  {cur['spec']}    cfg={cfg}")
        any_progress = False
        new_state: dict = {}
        for task in plan.get("tasks", []):
            tsv = REPO_ROOT / task["results_tsv"]
            trials = plan.get("eval", {}).get("num_trials_per_task", 0) or 0
            n, ok = episodes_done(tsv)
            key = f"{cur['slug']}::{task['name']}"
            if n == 0:
                phase = "loading model / warming up envs" if (tsv.is_file() or plan) else "starting policy server"
                out.append(f"    {task['name']:<16} {phase}  (first episode usually lands 1-4 min in)")
                continue
            any_progress = True
            # Prefer the measured rate between refreshes. Episodes arrive in bursts (every worker is
            # retired and its env rebuilt after each episode), so keep the last known rate whenever the
            # count has not moved - do not overwrite the sample with an unchanged one.
            prev_entry = st["prev"].get(key) or {}
            rate = prev_entry.get("rate")
            if prev_entry.get("n") and n > prev_entry["n"] and now > prev_entry.get("t", 0):
                rate = (now - prev_entry["t"]) / (n - prev_entry["n"])
                new_state[key] = {"n": n, "t": now, "rate": rate}
            elif prev_entry:
                new_state[key] = prev_entry
            elif plan:
                plan_age = now - (st.get("plan_mtime") or now)
                if n and plan_age > 0:
                    rate = plan_age / n
                new_state[key] = {"n": n, "t": now, "rate": rate}
            frac = (n / trials) if trials else 0.0
            speed = f"{rate:.1f}s/ep" if rate else "?"
            eta_task = (rate * (trials - n)) if (rate and trials) else None
            rate_pct = f"{ok / n * 100:.1f}%" if n else "?"
            out.append(
                f"    {task['name']:<16} episodes {bar(frac, width)} {n}/{trials} ({frac * 100:.0f}%)"
                f"   success {ok}/{n} = {rate_pct}"
                f"   {speed}  ETA {human(eta_task)}   max_steps={task.get('max_steps') or 'default'}"
            )
        st["new_state"] = new_state
        if not any_progress:
            n_envs = plan.get("eval", {}).get("num_envs", "?")
            out.append(f"    (model load + {n_envs} env warm-up usually takes 1-3 min before the first result)")
        out.append("")

    if done:
        out.append("finished")
        out.append("    #   checkpoint                                            result   rate     用时")
        for i, d in enumerate(done, 1):
            rows = d["results"]
            cells = "  ".join(
                f"{r['task']} {r.get('episodes')}/{d.get('trials')} {r['success_rate']:.1f}%"
                if r.get("success_rate") is not None
                else f"{r['task']} failed"
                for r in rows
            )
            rate = sum(r["success_rate"] for r in rows) / len(rows)
            out.append(
                f"    {i:<3} {d['spec'][:52]:<52} {cells:<8} {rate:>5.1f}%   {human(d['duration'])}"
            )
        out.append("")

    known = {d["spec"] for d in done} | ({st["current"]["spec"]} if st["current"] else set())
    queued = [spec for _, _, spec in st["markers"] if spec not in known]
    if queued:
        preview = ", ".join(queued[:3]) + (" ..." if len(queued) > 3 else "")
        out.append(f"queued ({len(queued)}): {preview}")
    out.append(f"updated {dt.datetime.now().strftime('%H:%M:%S')}  (log: tail -f {batch_dir / 'batch.log'})")
    return "\n".join(out)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--batch-dir", default="", help="default: newest directory under runs/")
    ap.add_argument("--total", type=int, default=0, help="override the expected number of checkpoints (e.g. when the plan shrinks mid-run)")
    ap.add_argument("--watch", type=float, default=0.0, help="refresh every N seconds (0 = print once)")
    args = ap.parse_args()

    if args.batch_dir:
        batch_dir = pathlib.Path(args.batch_dir)
        batch_dir = batch_dir if batch_dir.is_absolute() else REPO_ROOT / batch_dir
    else:
        cands = [p for p in (REPO_ROOT / "runs").glob("*") if (p / "batch.log").is_file()]
        if not cands:
            print("no batch run found under runs/", file=sys.stderr)
            return 1
        batch_dir = max(cands, key=lambda p: (p / "batch.log").stat().st_mtime)

    while True:
        st = collect(batch_dir)
        if args.total:
            st["total"] = args.total
        text = render(st)
        if args.watch and sys.stdout.isatty():
            sys.stdout.write("\033[2J\033[H")
        print(text, flush=True)
        if st.get("new_state"):
            try:
                st["state_path"].write_text(json.dumps({**st["prev"], **st["new_state"]}))
            except OSError:
                pass
        if not args.watch:
            return 0
        time.sleep(args.watch)


if __name__ == "__main__":
    sys.exit(main())

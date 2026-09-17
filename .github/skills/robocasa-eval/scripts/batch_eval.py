#!/usr/bin/env python3
"""Run the same task(s) over many checkpoints - one policy server per checkpoint.

    python3 .github/skills/robocasa-eval/scripts/batch_eval.py \
        --checkpoints runs/od850/cps.txt --tasks OpenDoubleDoor --num-envs 16 --batch-id od850

Each non-empty, non-`#` line of the checkpoint file is `<config>/<exp>/<step>` (relative to
`checkpoints/`) or an absolute checkpoint dir; a value containing `*` is treated as a glob. For every
checkpoint the driver:

1. builds a plan with `plan_eval.py` (openpi env, derives the serve config from the path),
2. makes sure the policy-server port is free, so `run_eval.py` can never reuse the *previous*
   checkpoint's server (each checkpoint has its own `--policy.dir`),
3. runs `run_eval.py`, which starts the server for that checkpoint, runs the task, stops it again,
4. merges the result into `runs/<batch-id>/batch_summary.{json,md}` **after every checkpoint**, so the
   log is monitorable and the run is resumable with `--resume`.

Stdlib only. Launch it in `screen` (it runs for hours):

    screen -dmS eval_od850 bash -lc 'cd <repo> && python3 .github/skills/.../batch_eval.py ... \
        > runs/od850/batch.log 2>&1'
"""

from __future__ import annotations

import argparse
import datetime as dt
import glob
import json
import os
import pathlib
import shlex
import socket
import subprocess
import sys
import time

REPO_ROOT = pathlib.Path(__file__).resolve().parents[4]
SCRIPT_DIR = pathlib.Path(__file__).resolve().parent
PLAN_EVAL = SCRIPT_DIR / "plan_eval.py"
RUN_EVAL = SCRIPT_DIR / "run_eval.py"


def port_healthy(port: int, host: str = "127.0.0.1", timeout: float = 3.0) -> bool:
    """serve_policy answers HTTP 426 (upgrade required) to a plain GET when healthy."""
    try:
        with socket.create_connection((host, port), timeout=timeout) as s:
            s.sendall(b"GET / HTTP/1.1\r\nHost: 127.0.0.1\r\nConnection: close\r\n\r\n")
            head = s.recv(64)
    except OSError:
        return False
    return b"426" in head.split(b"\r\n", 1)[0]


def free_port(port: int, screen_name: str, timeout: float = 180.0) -> bool:
    """Stop a leftover server on `port` and wait until the port is actually free."""
    subprocess.run(["screen", "-S", screen_name, "-X", "quit"], check=False, timeout=15)
    deadline = time.time() + timeout
    while time.time() < deadline:
        if not port_healthy(port):
            return True
        time.sleep(3)
    return False


def stream(cmd: list[str], log_path: pathlib.Path) -> int:
    """Run `cmd` from the repo root, teeing stdout+stderr to `log_path` and our stdout."""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as log:
        proc = subprocess.Popen(
            cmd, cwd=REPO_ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1
        )
        assert proc.stdout is not None
        for line in proc.stdout:
            log.write(line)
            log.flush()
            sys.stdout.write(line)
            sys.stdout.flush()
        return proc.wait()


def load_checkpoints(spec: str) -> list[str]:
    p = pathlib.Path(spec)
    if p.is_file():
        raw = [line.strip() for line in p.read_text(encoding="utf-8").splitlines()]
    elif "*" in spec:
        raw = sorted(glob.glob(str(REPO_ROOT / spec)))
    else:
        sys.exit(f"[batch] --checkpoints must be a file or a glob, got: {spec}")
    out, seen = [], set()
    for line in raw:
        if not line or line.startswith("#"):
            continue
        line = line.rstrip("/")
        if line not in seen:
            seen.add(line)
            out.append(line)
    if not out:
        sys.exit(f"[batch] no checkpoints found in {spec}")
    return out


def ckpt_dir(spec: str) -> pathlib.Path:
    p = pathlib.Path(spec)
    if p.is_absolute():
        return p
    p = REPO_ROOT / p
    return p if (p / "params").is_dir() or (p / "_CHECKPOINT_METADATA").is_file() or (p.parent.name == "checkpoints") else REPO_ROOT / "checkpoints" / spec


def slug(spec: str) -> str:
    return spec.replace("/", "__").lstrip("-")


def write_batch_summary(batch_dir: pathlib.Path, rows: list[dict], started: float | None, total: int) -> None:
    rows_sorted = sorted(rows, key=lambda r: (r["checkpoint"]))
    done = len([r for r in rows_sorted if r["success_rate"] is not None])
    rates = [r["success_rate"] for r in rows_sorted if r["success_rate"] is not None]
    mean = sum(rates) / len(rates) if rates else None

    lines = [
        f"# Batch {batch_dir.name}",
        "",
        f"- 生成时间: {dt.datetime.now().strftime('%Y-%m-%d %H:%M')}",
        f"- 进度: {done}/{total} 个 checkpoint 有结果",
    ]
    if started:
        elapsed = time.time() - started
        lines.append(f"- 已用时间: {elapsed / 3600:.2f} h" + (f"，预计剩余 {elapsed / max(done, 1) * (total - done) / 3600:.2f} h" if done else ""))
    if mean is not None:
        lines.append(f"- OpenDoubleDoor 平均成功率（已完成的）: {mean:.1f}%")
    lines += ["", "| checkpoint | task | max_steps | Result | Rate |", "|---|---|---|---|---|"]
    for r in rows_sorted:
        res = "–" if r["success_rate"] is None else f"{r['episodes']}"
        rate = "失败 / 未跑" if r["success_rate"] is None else f"{r['success_rate']:.1f}%"
        lines.append(f"| {r['checkpoint']} | {r['task']} | {r['max_steps'] or 'default'} | {res} | {rate} |")
    lines.append("")
    (batch_dir / "batch_summary.md").write_text("\n".join(lines), encoding="utf-8")
    (batch_dir / "batch_summary.json").write_text(
        json.dumps({"rows": rows_sorted, "mean_success_rate": mean, "done": done, "total": total}, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def already_done(ckpt_summary: pathlib.Path, tasks: list[str]) -> bool:
    if not ckpt_summary.is_file():
        return False
    try:
        blob = json.loads(ckpt_summary.read_text())
    except Exception:  # noqa: BLE001
        return False
    got = {r["task"]: r.get("success_rate") for r in blob.get("results", [])}
    return all(got.get(t) is not None for t in tasks)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoints", required=True, help="file with one <config>/<exp>/<step> per line, or a glob")
    ap.add_argument("--tasks", default="OpenDoubleDoor")
    ap.add_argument("--batch-id", default="", help="default: batch-<YYYYmmdd-HHMM>")
    ap.add_argument("--out-root", default="runs", help="where per-checkpoint dirs and the summary go")
    ap.add_argument("--num-envs", type=int, default=16)
    ap.add_argument("--num-trials", type=int, default=100)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--replan-steps", type=int, default=5)
    ap.add_argument("--max-steps", type=int, default=0, help="0 = per-task budget from examples/robocasa/main.py")
    ap.add_argument("--env-gb", type=float, default=0.0)
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--resume", action="store_true", help="skip checkpoints that already have results")
    ap.add_argument("--force", action="store_true", help="re-run even if results exist")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    checkpoints = load_checkpoints(args.checkpoints)
    tasks = [t.strip() for t in args.tasks.split(",") if t.strip()]
    batch_id = args.batch_id or f"batch-{dt.datetime.now().strftime('%Y%m%d-%H%M')}"
    batch_dir = REPO_ROOT / args.out_root / batch_id
    batch_dir.mkdir(parents=True, exist_ok=True)
    screen_name = f"openpi_serve_{args.port}"
    started = time.time()

    print(f"[batch] {batch_id}: {len(checkpoints)} checkpoint(s) x {tasks} "
          f"(num_envs={args.num_envs}, trials={args.num_trials}, seed={args.seed}, max_steps={args.max_steps or 'per-task'})",
          flush=True)
    print(f"[batch] outputs under {batch_dir}", flush=True)

    rows: list[dict] = []
    for i, spec in enumerate(checkpoints, 1):
        ckpt_slug = slug(spec)
        ckpt_dir_out = batch_dir / ckpt_slug
        summary_json = ckpt_dir_out / "summary.json"
        print(f"\n{'#' * 78}\n[batch] {i}/{len(checkpoints)}  {spec}\n{'#' * 78}", flush=True)

        if args.resume and not args.force and already_done(summary_json, tasks):
            print("[batch] already done, skipping (--resume)", flush=True)
            blob = json.loads(summary_json.read_text())
            rows += [{"checkpoint": spec, **{k: r.get(k) for k in ("task", "max_steps", "success_rate", "episodes", "status")}} for r in blob["results"]]
            continue

        plan_cmd = [
            "uv", "run", "--directory", str(REPO_ROOT), "python", str(PLAN_EVAL),
            "--ckpt", str(ckpt_dir(spec)), "--tasks", ",".join(tasks),
            "--num-envs", str(args.num_envs), "--num-trials", str(args.num_trials),
            "--seed", str(args.seed), "--replan-steps", str(args.replan_steps),
            "--port", str(args.port), "--out", str(ckpt_dir_out / "plan.json"),
        ]
        if args.max_steps:
            plan_cmd += ["--max-steps", str(args.max_steps)]
        if args.env_gb:
            plan_cmd += ["--env-gb", str(args.env_gb)]
        if args.dry_run:
            print("[batch] plan:", " ".join(shlex.quote(c) for c in plan_cmd), flush=True)
            print("[batch] eval:", f"python3 {RUN_EVAL} --plan {ckpt_dir_out / 'plan.json'}", flush=True)
            continue

        rc = stream(plan_cmd, ckpt_dir_out / "plan.log")
        if rc != 0 or not (ckpt_dir_out / "plan.json").is_file():
            print(f"[batch] !! plan failed (rc={rc}) for {spec}", file=sys.stderr, flush=True)
            rows += [{"checkpoint": spec, "task": t, "max_steps": None, "success_rate": None, "episodes": None, "status": f"plan failed rc={rc}"} for t in tasks]
            write_batch_summary(batch_dir, rows, started, len(checkpoints))
            continue

        if not free_port(args.port, screen_name):
            print(f"[batch] !! port {args.port} still busy after timeout; not starting a second server for {spec}", file=sys.stderr, flush=True)
            rows += [{"checkpoint": spec, "task": t, "max_steps": None, "success_rate": None, "episodes": None, "status": "port busy"} for t in tasks]
            write_batch_summary(batch_dir, rows, started, len(checkpoints))
            continue

        plan = json.loads((ckpt_dir_out / "plan.json").read_text())
        print(f"[batch] serve config = {plan['inference_config']}  |  tasks = "
              f"{[(t['name'], t['max_steps']) for t in plan['tasks']]}", flush=True)
        rc = stream([sys.executable, str(RUN_EVAL), "--plan", str(ckpt_dir_out / "plan.json")], ckpt_dir_out / "eval.log")

        if summary_json.is_file():
            blob = json.loads(summary_json.read_text())
            rows += [{"checkpoint": spec, **{k: r.get(k) for k in ("task", "max_steps", "success_rate", "episodes", "status")}} for r in blob["results"]]
        else:
            print(f"[batch] !! no summary written (rc={rc}) for {spec}", file=sys.stderr, flush=True)
            rows += [{"checkpoint": spec, "task": t, "max_steps": None, "success_rate": None, "episodes": None, "status": f"eval failed rc={rc}"} for t in tasks]
        write_batch_summary(batch_dir, rows, started, len(checkpoints))

    if not args.dry_run:
        write_batch_summary(batch_dir, rows, started, len(checkpoints))
        ok = len([r for r in rows if r["success_rate"] is not None])
        print(f"\n[batch] DONE: {ok}/{len(checkpoints)} checkpoints produced results in {(time.time() - started) / 3600:.2f} h", flush=True)
        print(f"[batch] summary: {batch_dir / 'batch_summary.md'}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())

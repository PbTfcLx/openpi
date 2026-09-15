#!/usr/bin/env python3
"""Run the RoboCasa eval described by a plan produced by `plan_eval.py`.

    python3 .github/skills/robocasa-eval/scripts/run_eval.py --plan runs/<run-id>/plan.json

* checks the policy server on `plan.eval.port` (a plain GET must answer HTTP 426) and, with
  `--manage-server` (default), starts it in `screen` and stops it again afterwards;
* runs `examples/robocasa/main.py` once per task, sequentially, with the eval flags from the plan;
* streams every task's output to `data/robocasa/<exp>/checkpoint-<step>/logs/<task>.log` while echoing
  it, and parses the harness's `Total success rate: X% (n/N)` line;
* writes `summary.json` + `summary.md` next to the plan and prints the markdown table.

Stdlib only - run it with any python3 (it does not import openpi). Long runs: start it in `screen` and
poll the log file.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import pathlib
import re
import shlex
import socket
import subprocess
import sys
import time

RATE_RE = re.compile(r"Total success rate:\s*([\d.]+)%\s*\((\d+)/(\d+)\)")
EPISODES_RE = re.compile(r"Total episodes:\s*(\d+)")
ABANDONED_RE = re.compile(r"(\d+) episode\(s\) were abandoned")


def port_healthy(port: int, host: str = "127.0.0.1", timeout: float = 3.0) -> bool:
    """serve_policy answers HTTP 426 (upgrade required) to a plain GET when healthy."""
    try:
        with socket.create_connection((host, port), timeout=timeout) as s:
            s.sendall(b"GET / HTTP/1.1\r\nHost: 127.0.0.1\r\nConnection: close\r\n\r\n")
            head = s.recv(64)
    except OSError:
        return False
    return b"426" in head.split(b"\r\n", 1)[0]


def screen_session_exists(name: str) -> bool:
    try:
        out = subprocess.run(["screen", "-ls"], capture_output=True, text=True, timeout=10).stdout
    except Exception:  # noqa: BLE001
        return False
    return name in out


def start_server(plan: dict, *, log_path: pathlib.Path, screen_name: str) -> list[str]:
    repo = plan["repo_root"]
    cmd = (
        f"cd {shlex.quote(repo)} && "
        'XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_FLAGS="--xla_gpu_deterministic_ops=true" '
        f"uv run python scripts/serve_policy.py --port={plan['eval']['port']} policy:checkpoint "
        f"--policy.config={shlex.quote(plan['inference_config'])} "
        f"--policy.dir={shlex.quote(plan['checkpoint_dir'])} "
        f"> {shlex.quote(str(log_path))} 2>&1"
    )
    argv = ["screen", "-dmS", screen_name, "bash", "-lc", cmd]
    subprocess.run(argv, check=True, timeout=30)
    return argv


def stop_server(screen_name: str) -> None:
    subprocess.run(["screen", "-S", screen_name, "-X", "quit"], check=False, timeout=15)


def wait_healthy(port: int, timeout_s: float, *, log_path: pathlib.Path | None = None) -> bool:
    deadline = time.time() + timeout_s
    started = time.time()
    while time.time() < deadline:
        if port_healthy(port):
            return True
        if time.time() - started > 30 and int(time.time() - started) % 30 < 5 and log_path is not None:
            print(f"[server] still not up after {int(time.time() - started)}s ({log_path}):", flush=True)
            try:
                for line in log_path.read_text(errors="replace").splitlines()[-5:]:
                    print(f"    | {line}", flush=True)
            except OSError:
                pass
        time.sleep(5)
    return False


def build_eval_argv(plan: dict, task: dict) -> list[str]:
    ev = plan["eval"]
    argv = [
        plan["robocasa_python"],
        str(pathlib.Path(plan["repo_root"]) / "examples" / "robocasa" / "main.py"),
        "--args.env-name",
        task["env_name"],
        "--args.num-trials-per-task",
        str(ev["num_trials_per_task"]),
        "--args.num-envs",
        str(ev["num_envs"]),
        "--args.seed",
        str(ev["seed"]),
        "--args.replan-steps",
        str(ev["replan_steps"]),
        "--args.max-steps",
        str(ev["max_steps"]),
        "--args.video-out-path",
        task["video_dir"],
        "--args.results-out-path",
        task["results_tsv"],
    ]
    if not ev.get("save_video", True):
        argv.append("--args.no-save-video")
    return argv


def run_task(plan: dict, task: dict) -> dict:
    repo = pathlib.Path(plan["repo_root"])
    log_path = repo / task["log"]
    log_path.parent.mkdir(parents=True, exist_ok=True)
    argv = build_eval_argv(plan, task)

    env = dict(os.environ)
    env["PYTHONPATH"] = str(repo / "packages" / "openpi-client" / "src")
    env.setdefault("OPENPI_EVAL_THREADS", "1")

    print(f"\n{'=' * 78}\n[task] {task['name']}  ({task['env_name']})\n[cmd ] {' '.join(shlex.quote(a) for a in argv)}"
          f"\n[log ] {log_path}\n{'=' * 78}", flush=True)

    t0 = time.time()
    rate = None
    episodes = None
    abandoned = 0
    with log_path.open("w", encoding="utf-8") as log:
        proc = subprocess.Popen(argv, cwd=repo, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
        assert proc.stdout is not None
        for line in proc.stdout:
            log.write(line)
            log.flush()
            print(f"[{task['name']}] {line.rstrip()}", flush=True)
            m = RATE_RE.search(line)
            if m:
                rate = float(m.group(1))
                successes, episodes = int(m.group(2)), int(m.group(3))
            m = EPISODES_RE.search(line)
            if m:
                episodes = int(m.group(1))
            m = ABANDONED_RE.search(line)
            if m:
                abandoned = int(m.group(1))
        code = proc.wait()
    duration = time.time() - t0

    status = "ok" if rate is not None else "failed"
    if status == "ok" and episodes is not None and episodes < plan["eval"]["num_trials_per_task"]:
        status = "partial"
    return {
        "task": task["name"],
        "env_name": task["env_name"],
        "status": status,
        "exit_code": code,
        "success_rate": rate,
        "episodes": episodes,
        "abandoned": abandoned,
        "log": task["log"],
        "duration_s": round(duration, 1),
    }


def write_summary(plan: dict, results: list[dict], out_dir: pathlib.Path) -> tuple[pathlib.Path, pathlib.Path]:
    ev = plan["eval"]
    ok = [r for r in results if r["success_rate"] is not None]
    mean = sum(r["success_rate"] for r in ok) / len(ok) if ok else None
    total_n = sum(r["episodes"] or 0 for r in ok)
    total_ok = sum(round((r["success_rate"] / 100) * (r["episodes"] or 0)) for r in ok)

    stamp = dt.datetime.now().strftime("%Y-%m-%d %H:%M")
    lines = [
        f"### {plan['exp_name']} / checkpoint-{plan['step']}",
        "",
        f"- 模型: `{plan['checkpoint_dir']}`",
        f"- 训练 config: `{plan['train_config']}` | 推理 config: `{plan['inference_config']}`",
        f"- 评测参数: {ev['num_trials_per_task']} trials × {ev['num_envs']} envs, seed={ev['seed']}, "
        f"replan={ev['replan_steps']}, max_steps={ev['max_steps']}, save_video={ev['save_video']}",
        f"- 时间: {stamp}",
        "",
        "| Task | Success | Rate |",
        "|---|---|---|",
    ]
    for r in results:
        if r["success_rate"] is None:
            lines.append(f"| {r['task']} | – | 失败 (exit {r['exit_code']}) |")
        else:
            flag = "" if r["status"] == "ok" else " ⚠️"
            lines.append(f"| {r['task']} | {r['episodes']}/{plan['eval']['num_trials_per_task']} | {r['success_rate']:.1f}%{flag} |")
    if mean is not None:
        lines.append(f"| **平均** | {total_ok}/{total_n} | **{mean:.1f}%** |")
    lines += ["", f"- 运行目录: `{plan['run_dir']}`", ""]
    md = "\n".join(lines)

    md_path = out_dir / "summary.md"
    md_path.write_text(md, encoding="utf-8")
    js_path = out_dir / "summary.json"
    js_path.write_text(
        json.dumps({"plan": plan, "results": results, "mean_success_rate": mean, "generated": stamp}, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return md_path, js_path


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--plan", required=True)
    ap.add_argument("--only", default="", help="comma separated subset of tasks to (re)run")
    ap.add_argument("--manage-server", dest="manage_server", action="store_true", default=True)
    ap.add_argument("--no-manage-server", dest="manage_server", action="store_false")
    ap.add_argument("--keep-server", action="store_true", help="do not stop a server that we started")
    ap.add_argument("--server-timeout", type=float, default=900.0, help="seconds to wait for the server to come up")
    ap.add_argument("--dry-run", action="store_true", help="print commands only")
    args = ap.parse_args()

    plan_path = pathlib.Path(args.plan)
    plan_path = plan_path if plan_path.is_absolute() else pathlib.Path.cwd() / plan_path
    plan = json.loads(plan_path.read_text())
    out_dir = plan_path.parent
    repo = pathlib.Path(plan["repo_root"])

    tasks = plan["tasks"]
    if args.only:
        want = {t.strip().lower() for t in args.only.split(",") if t.strip()}
        tasks = [t for t in tasks if t["name"].lower() in want]
        if not tasks:
            print(f"[run] --only {args.only!r} matched no task in the plan", file=sys.stderr)
            return 2

    if not pathlib.Path(plan["robocasa_python"]).exists():
        print(f"[run] robocasa python not found: {plan['robocasa_python']}\n"
              f"      set OPENPI_ROBOCASA_PYTHON or --robocasa-python when building the plan", file=sys.stderr)
        return 2

    if args.dry_run:
        for t in tasks:
            print(" ".join(shlex.quote(a) for a in build_eval_argv(plan, t)))
        return 0

    port = plan["eval"]["port"]
    screen_name = f"openpi_serve_{port}"
    serve_log = repo / "logs" / f"serve_{port}.log"
    serve_log.parent.mkdir(parents=True, exist_ok=True)
    started_by_us = False

    healthy = port_healthy(port)
    print(f"[server] port {port} healthy: {healthy}", flush=True)
    if not healthy:
        if not args.manage_server:
            print(f"[server] not reachable and --no-manage-server given; aborting", file=sys.stderr)
            return 2
        print(f"[server] starting {screen_name} (config={plan['inference_config']}, dir={plan['checkpoint_dir']})", flush=True)
        start_server(plan, log_path=serve_log, screen_name=screen_name)
        started_by_us = True
        if not wait_healthy(port, args.server_timeout, log_path=serve_log):
            print(f"[server] did not become healthy within {args.server_timeout:.0f}s - see {serve_log}", file=sys.stderr)
            return 2
        print("[server] healthy (HTTP 426)", flush=True)

    results: list[dict] = []
    try:
        for t in tasks:
            (repo / t["video_dir"]).mkdir(parents=True, exist_ok=True)
            (repo / t["results_tsv"]).parent.mkdir(parents=True, exist_ok=True)
            results.append(run_task(plan, t))
    finally:
        if started_by_us and not args.keep_server:
            print(f"[server] stopping {screen_name}", flush=True)
            stop_server(screen_name)

    md_path, js_path = write_summary(plan, results, out_dir)
    print(f"\n[summary] {md_path}\n[summary] {js_path}\n")
    print(md_path.read_text(encoding="utf-8"))
    print("记录到飞书: 见 .github/skills/robocasa-eval/references/feishu-recording.md")
    return 0 if all(r["success_rate"] is not None for r in results) else 1


if __name__ == "__main__":
    sys.exit(main())

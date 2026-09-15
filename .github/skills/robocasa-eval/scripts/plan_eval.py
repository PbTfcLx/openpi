#!/usr/bin/env python3
"""Resolve a RoboCasa evaluation plan for an openpi checkpoint.

Run from the repo root with the openpi env (needs `openpi.training.config`):

    uv run python .github/skills/robocasa-eval/scripts/plan_eval.py \
        --ckpt checkpoints/pi05_robocasa_state_noise/test_state_noise/18000 \
        --tasks OpenDrawer,CloseDrawer,TurnOnStove \
        --out runs/<run-id>/plan.json

It *does not run anything*. It resolves and validates:

* the step directory (accepts an experiment dir and picks the newest numeric step),
* the training config (explicit, else inferred from `checkpoints/<config>/<exp>/<step>`),
* the config to hand to `serve_policy.py` (same name by default) and the architecture fields that must
  match the checkpoint, plus any training-only data augmentation found in the config,
* the persisted `assets/norm_stats.json`,
* `num_envs` from the CPU/memory of this box,
* all output paths (`data/robocasa/<exp>/checkpoint-<step>/{videos,results,logs}`).

The plan JSON is consumed by `run_eval.py`.
"""

from __future__ import annotations

import argparse
import difflib
import json
import os
import pathlib
import re
import sys

REPO_ROOT = pathlib.Path(__file__).resolve().parents[4]

DEFAULT_ROBOCASA_PYTHON = pathlib.Path(
    "/root/autodl-tmp/Isaac-GR00T/gr00t/eval/sim/robocasa/robocasa_uv/.venv/bin/python"
)

# Task classes used so far. Any other class registered with robocasa also works: the env id is
# always `robocasa_panda_omron/<TaskClass>_PandaOmron_Env`.
DEFAULT_TASKS = [
    "OpenDrawer",
    "CloseDrawer",
    "OpenDoubleDoor",
    "CloseDoubleDoor",
    "CoffeeSetupMug",
    "TurnOnStove",
]

# Architecture fields that must describe the same model the checkpoint was trained with.
ARCH_FIELDS = (
    "pi05",
    "action_horizon",
    "action_dim",
    "max_token_len",
    "discrete_state_input",
    "paligemma_variant",
    "action_expert_variant",
)

# Training-only data augmentation: must never be applied at inference time. (Safe to keep the same
# config name anyway - policy_config.create_trained_policy does not add these transforms.)
TRAIN_ONLY_DATA_FLAGS = ("state_noise", "prompt_drop_p", "state_noise_beta_a", "state_noise_beta_b")


def env_id(task: str) -> str:
    return f"robocasa_panda_omron/{task}_PandaOmron_Env"


def parse_tasks(spec: str) -> list[str]:
    raw = [t.strip() for t in re.split(r"[,\s]+", spec or "") if t.strip()]
    if not raw:
        return list(DEFAULT_TASKS)
    lower = {t.lower(): t for t in DEFAULT_TASKS}
    out = []
    for t in raw:
        if t in DEFAULT_TASKS:
            out.append(t)
            continue
        # tolerate "opendoubledoor" / "open_double_door" style input
        key = t.lower().replace("_", "")
        match = next((v for k, v in lower.items() if k.replace("_", "") == key), None)
        out.append(match or t)
    seen, uniq = set(), []
    for t in out:
        if t not in seen:
            seen.add(t)
            uniq.append(t)
    return uniq


def is_step_dir(d: pathlib.Path) -> bool:
    return (d / "params").is_dir() or (d / "_CHECKPOINT_METADATA").is_file()


def resolve_checkpoint(spec: str) -> pathlib.Path:
    p = pathlib.Path(spec)
    p = p if p.is_absolute() else (REPO_ROOT / p)
    p = p.resolve()
    if not p.is_dir():
        sys.exit(f"[plan] checkpoint directory not found: {p}")
    if is_step_dir(p):
        return p
    steps = [c for c in p.iterdir() if c.is_dir() and re.fullmatch(r"\d+", c.name)]
    if not steps:
        sys.exit(f"[plan] {p} is neither a step dir (no params/ or _CHECKPOINT_METADATA) nor contains numeric step dirs")
    return max(steps, key=lambda c: int(c.name))


def infer_names_from_path(ckpt: pathlib.Path) -> tuple[str, str]:
    """`checkpoints/<config>/<exp>/<step>` -> (config, exp)."""
    try:
        rel = ckpt.resolve().relative_to((REPO_ROOT / "checkpoints").resolve())
    except ValueError:
        return "", ""
    parts = rel.parts
    return (parts[0] if parts else ""), (parts[1] if len(parts) > 1 else "")


def read_norm_stats(step_dir: pathlib.Path) -> dict:
    f = step_dir / "assets" / "norm_stats.json"
    if not f.is_file():
        return {"present": False, "path": str(f)}
    try:
        blob = json.loads(f.read_text())
    except Exception as e:  # noqa: BLE001
        return {"present": True, "path": str(f), "error": str(e)}
    ns = blob.get("norm_stats", blob)
    info: dict = {"present": True, "path": str(f), "keys": sorted(ns.keys()) if isinstance(ns, dict) else []}
    if isinstance(ns, dict):
        for k, v in ns.items():
            if isinstance(v, dict) and isinstance(v.get("mean"), list):
                info[f"{k}_dim"] = len(v["mean"])
    return info


def all_config_names() -> list[str]:
    try:
        from openpi.training import config as _config  # noqa: PLC0415
    except Exception:  # noqa: BLE001
        return []
    names: list[str] = []
    for attr in ("_CONFIGS", "_CONFIGS_DICT"):
        obj = getattr(_config, attr, None)
        if isinstance(obj, dict):
            names += [str(k) for k in obj]
        elif obj:
            names += [getattr(c, "name", "") for c in obj]
    return sorted({n for n in names if n})


def inspect_config(name: str) -> dict:
    """Architecture + training-only flags for a training config name."""
    out: dict = {"name": name, "ok": False}
    if not name:
        out["error"] = "no config name given or inferable"
        return out
    try:
        from openpi.training import config as _config  # noqa: PLC0415
    except Exception as e:  # noqa: BLE001
        out["error"] = f"cannot import openpi.training.config: {e}"
        return out
    try:
        cfg = _config.get_config(name)
    except Exception as e:  # noqa: BLE001
        out["error"] = str(e)
        close = difflib.get_close_matches(name, all_config_names(), n=6, cutoff=0.3)
        if close:
            out["did_you_mean"] = close
        return out

    model = cfg.model
    out["ok"] = True
    out["arch"] = {f: getattr(model, f, None) for f in ARCH_FIELDS}
    out["model_class"] = type(model).__name__
    data = cfg.data
    out["train_only_data_flags"] = {
        f: getattr(data, f, None) for f in TRAIN_ONLY_DATA_FLAGS if getattr(data, f, None) not in (None, 0.0, False)
    }
    # Fields that describe the training run, not the architecture (useful context in the record).
    for f in ("num_train_steps", "batch_size", "freeze_filter"):
        if hasattr(cfg, f):
            out.setdefault("train_config_fields", {})[f] = "set" if getattr(cfg, f) is not None else None
    out["data_class"] = type(data).__name__
    return out


def usable_cores() -> tuple[int, str]:
    """Cores this container may actually use.

    `os.cpu_count()` reports the *host* (208 on this box) and `os.sched_getaffinity` can too, while
    `nproc` / `/sys/fs/cgroup/cpu.max` reflect the container's quota (25 here). Prefer the quota.
    """
    try:
        quota, period = pathlib.Path("/sys/fs/cgroup/cpu.max").read_text().split()
        if quota != "max":
            return max(1, int(float(quota) / float(period))), "cgroup cpu.max"
    except Exception:  # noqa: BLE001
        pass
    try:
        n = len(os.sched_getaffinity(0))
        if n:
            return n, "sched_getaffinity"
    except Exception:  # noqa: BLE001
        pass
    return os.cpu_count() or 4, "os.cpu_count"


def resources(env_gb: float | None, save_video: bool) -> tuple[int, str, float | None]:
    cores, cores_src = usable_cores()
    mem_max = mem_used = mem_anon = None
    try:
        mem_max = int(pathlib.Path("/sys/fs/cgroup/memory.max").read_text().strip())
        current = int(pathlib.Path("/sys/fs/cgroup/memory.current").read_text().strip())
        inactive = anon = 0
        for line in pathlib.Path("/sys/fs/cgroup/memory.stat").read_text().splitlines():
            key, _, value = line.partition(" ")  # exact key: `anon_thp` must not clobber `anon`
            if key == "inactive_file":
                inactive = int(value)
            elif key == "anon":
                anon = int(value)
        # page cache is reclaimable - same rule as stats_cpu_mem.py in the repo root
        mem_used = current - inactive
        mem_anon = anon
    except Exception:  # noqa: BLE001
        pass

    avail_gb = None if (mem_max is None or mem_used is None) else (mem_max - mem_used) / 1e9
    per_env = env_gb if env_gb else (4.0 if save_video else 2.5)
    by_cpu = max(2, cores // 4)  # oversubscription is a known SIGKILL cause here
    by_mem = 99
    if avail_gb is not None:
        by_mem = max(1, int(avail_gb // per_env))
    n = max(2, min(12, by_cpu, by_mem)) if by_mem >= 2 else max(1, min(12, by_cpu, by_mem))

    why = f"cores={cores} ({cores_src}) // 4 -> {by_cpu}"
    if avail_gb is not None:
        why += (
            f"; avail_mem={avail_gb:.0f}GB (max {(mem_max or 0) / 1e9:.0f} - used {(mem_used or 0) / 1e9:.0f}"
            f", anon={(mem_anon or 0) / 1e9:.0f}GB) / {per_env:g}GB per env -> {by_mem}"
        )
    why += f" => num_envs={n} (cap 12, override with --num-envs)"
    return n, why, avail_gb


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", required=True, help="step dir, or exp dir (latest numeric step is used)")
    ap.add_argument("--tasks", default="", help="comma/space separated task classes (default: the 6 robocasa tasks)")
    ap.add_argument("--train-config", default="", help="default: inferred from checkpoints/<config>/<exp>/<step>")
    ap.add_argument("--inference-config", default="", help="default: same as the training config")
    ap.add_argument("--exp-name", default="", help="default: inferred from the checkpoint path")
    ap.add_argument("--run-id", default="", help="default: <exp>-<config>-<step>")
    ap.add_argument("--out", default="", help="where to write plan.json (default: runs/<run-id>/plan.json)")
    ap.add_argument("--num-trials", type=int, default=100)
    ap.add_argument("--num-envs", type=int, default=0, help="0 = derive from CPU/memory")
    ap.add_argument("--env-gb", type=float, default=0.0, help="memory assumed per eval env when deriving num_envs (default 4 with video, 2.5 without)")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--replan-steps", type=int, default=5)
    ap.add_argument("--max-steps", type=int, default=720)
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--no-save-video", action="store_true", help="faster, much lower memory: writes --args.no-save-video")
    ap.add_argument("--robocasa-python", default=os.environ.get("OPENPI_ROBOCASA_PYTHON", str(DEFAULT_ROBOCASA_PYTHON)))
    args = ap.parse_args()

    ckpt = resolve_checkpoint(args.ckpt)
    step = int(ckpt.name)
    path_config, path_exp = infer_names_from_path(ckpt)
    train_config = args.train_config or path_config
    exp_name = args.exp_name or path_exp or train_config or "unknown"
    inference_config = args.inference_config or train_config
    run_id = args.run_id or f"{exp_name}-{train_config or 'noconfig'}-{step}"

    tasks = parse_tasks(args.tasks)
    num_envs, num_envs_why, avail_gb = resources(args.env_gb or None, save_video=not args.no_save_video)

    run_dir = REPO_ROOT / "data" / "robocasa" / exp_name / f"checkpoint-{step}"
    task_entries = []
    for t in tasks:
        slug = t.lower()
        task_entries.append(
            {
                "name": t,
                "env_name": env_id(t),
                "video_dir": str((run_dir / "videos" / slug).relative_to(REPO_ROOT)),
                "results_tsv": str((run_dir / "results" / f"{slug}.tsv").relative_to(REPO_ROOT)),
                "log": str((run_dir / "logs" / f"{slug}.log").relative_to(REPO_ROOT)),
            }
        )

    plan = {
        "run_id": run_id,
        "repo_root": str(REPO_ROOT),
        "exp_name": exp_name,
        "step": step,
        "checkpoint_dir": str(ckpt.relative_to(REPO_ROOT)) if ckpt.is_relative_to(REPO_ROOT) else str(ckpt),
        "train_config": train_config,
        "inference_config": inference_config,
        "task_spec_in": args.tasks,
        "tasks": task_entries,
        "eval": {
            "num_trials_per_task": args.num_trials,
            "num_envs": args.num_envs or num_envs,
            "num_envs_reason": f"explicit --num-envs" if args.num_envs else num_envs_why,
            "seed": args.seed,
            "replan_steps": args.replan_steps,
            "max_steps": args.max_steps,
            "save_video": not args.no_save_video,
            "port": args.port,
        },
        "robocasa_python": args.robocasa_python,
        "run_dir": str(run_dir.relative_to(REPO_ROOT)),
        "norm_stats": read_norm_stats(ckpt),
        "training_config": inspect_config(train_config),
        "inference_config_info": inspect_config(inference_config),
        "host": num_envs_why,
    }

    out_path = pathlib.Path(args.out) if args.out else REPO_ROOT / "runs" / run_id / "plan.json"
    out_path = out_path if out_path.is_absolute() else REPO_ROOT / out_path
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(plan, indent=2, ensure_ascii=False) + "\n")

    # ---- human summary -------------------------------------------------------------------
    print("=" * 78)
    print(f"plan            : {out_path}")
    print(f"checkpoint      : {plan['checkpoint_dir']}  (step {step}, exp '{exp_name}')")
    print(f"training config : {train_config or '<unknown>'}")
    print(f"INFERENCE config: {inference_config or '<unknown>'}   <- pass to serve_policy --policy.config")
    tc = plan["training_config"]
    if tc.get("ok"):
        print(f"  arch          : {tc['arch']}")
        if tc.get("train_only_data_flags"):
            print(f"  train-only aug: {tc['train_only_data_flags']}")
            print("                  (ignored by create_trained_policy at inference - do NOT copy into a serve config)")
    else:
        print(f"  !! could not inspect config: {tc.get('error')}")
        if tc.get("did_you_mean"):
            print(f"     did you mean: {tc['did_you_mean']}")
    ns = plan["norm_stats"]
    dims = {k: v for k, v in ns.items() if k.endswith("_dim")}
    print(f"norm stats      : {ns['path']} present={ns['present']} {dims}")
    print(f"eval            : {plan['eval']['num_trials_per_task']} trials x {plan['eval']['num_envs']} envs"
          f", seed={args.seed}, replan={args.replan_steps}, save_video={plan['eval']['save_video']}")
    print(f"  num_envs why  : {plan['eval']['num_envs_reason']}")
    print(f"run dir         : {plan['run_dir']}  (videos/, results/, logs/)")
    print("tasks           : " + ", ".join(f"{t['name']}" for t in task_entries))
    print("-" * 78)
    print("serve:")
    print(f"  cd {REPO_ROOT} && screen -dmS openpi_serve_{args.port} bash -lc \\")
    print(f"   'XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_FLAGS=\"--xla_gpu_deterministic_ops=true\" \\")
    print(f"    uv run python scripts/serve_policy.py --port={args.port} policy:checkpoint \\")
    print(f"    --policy.config={inference_config} --policy.dir={plan['checkpoint_dir']} \\")
    print(f"    > logs/serve_{args.port}.log 2>&1'")
    print("eval:")
    print(f"  cd {REPO_ROOT} && python3 .github/skills/robocasa-eval/scripts/run_eval.py --plan {out_path}")
    print("=" * 78)
    print(f"robocasa python : {args.robocasa_python} (exists: {pathlib.Path(args.robocasa_python).exists()})")


if __name__ == "__main__":
    main()

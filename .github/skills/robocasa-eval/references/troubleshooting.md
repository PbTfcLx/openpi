---
# Reference: RoboCasa eval failure modes (measured in this container)
---

# Troubleshooting RoboCasa eval runs

## "All worker processes died before completing all episodes"

A worker was hard-killed (SIGSEGV/SIGKILL — no Python exception, nothing queued). The usual cause here
is **container memory pressure**: cgroup `memory.max` ≈ 128.8 GB, and 12 envs with `--args.save-video`
push the peak into the limit, so 2–4 workers get SIGKILLed per run (exit code `-9`, often with no
`oom_kill` counter recorded).

Mitigations, in order:

1. lower `--args.num-envs` (this is why `plan_eval.py` starts from `cores // 4`, not `cores`),
2. `--args.no-save-video` (each in-flight episode buffers ≈ 0.4 GB of frames),
3. check real usage as `memory.current - inactive_file` (page cache is reclaimable) — `stats_cpu_mem.py`
   in the repo root uses the same definition.

With `--args.save-video`, a **missing** `rollout_XXXX_*.mp4` in the video dir names the lost episode:

```bash
ls data/robocasa/<exp>/checkpoint-<step>/videos/<task>/ \
 | sed -E 's/rollout_0*([0-9]+)_.*/\1/' | sort -n > /tmp/have.txt
seq 0 <N-1> | comm -13 /tmp/have.txt -
```

The harness re-queues a lost episode into a fresh worker (bounded by `--args.max-episode-restarts` /
`--args.max-worker-restarts`) and still reports the success rate; abandoned episodes are logged and
counted as failures.

## Thread oversubscription

`/etc/profile.d/autodl.env.sh` exports `OMP_NUM_THREADS=MKL_NUM_THREADS=$(nproc)` (=25 here). Every
worker inherits it, so 12 workers would run 12×25 OpenMP threads on 25 cores. `main.py` now forces 1
thread per worker before numpy/MuJoCo import; override with `OPENPI_EVAL_THREADS=n`.

## Reproducibility / comparing checkpoints

* Scene generation is reproducible; the policy is deterministic given `(seed, replan_idx)`.
* `--args.one-episode-per-worker` (default true) retires process + env after each episode. A **reused
  env** makes an episode's outcome depend on which episodes ran before it — `robocasa`/`robosuite`
  `reset()` does not restore everything (it mutates the MuJoCo *model*; controller internals persist).
  Only disable it for speed.
* Compare runs with the same `--args.seed` and `--args.num-trials-per-task`.
* **mp4 md5s are not a determinism check** (x264 is multithreaded).
* Residual: 2/4 episodes bit-identical across runs, the others wobble ±1 step in the success step
  (success flags stable). Suspect GPU/XLA nondeterminism under memory pressure — hence
  `XLA_FLAGS=--xla_gpu_deterministic_ops=true` on the server.

## Server-side checks

* `ss -tlnp` prints **nothing** in this container (no socket visibility). Use
  `grep -i 1F40 /proc/net/tcp` for port 8000, or a real `socket.create_connection`, or the 426 check in
  `run_eval.py`.
* `serve_policy.py` on `:8000` answering **HTTP 426** to a plain `GET` is healthy (websocket endpoint).
* A missing checkpoint dir / wrong `--policy.config` fails at load time; grep the server log
  (`logs/serve_<port>.log`) for the traceback instead of guessing from the eval side.

## Process management

* Long jobs **must** run inside `screen` (`tmux` is not installed). VS Code integrated terminals die with
  the ptyHost on remote reconnect and are unrecoverable: no `CAP_SYS_PTRACE`, `ptrace_scope=1`, and a
  seccomp filter block `reptyr` / `gdb -p` / `pidfd_getfd`.
* Reattach with `screen -r`; if it says "(Attached)" with no real client, run `screen -d <pid.tty.host>`
  then `screen -r`. `screen -ls` shows the spec.

## Do not touch while running

* `--args.one-episode-per-worker` and `--args.deterministic` change what the numbers mean.
* Deleting/rewriting the video dir mid-run breaks the leftover-video heuristic above; `main.py` uses
  `--args.run-start-epoch` internally to tell this run's videos from earlier leftovers.

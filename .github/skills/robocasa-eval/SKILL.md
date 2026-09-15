---
name: robocasa-eval
description: "Run a RoboCasa policy evaluation for an openpi checkpoint and record the results. Use when the user gives a checkpoint/model path plus the tasks to run and (optionally) the training config, and asks to evaluate / benchmark / score / 跑评测 / 记录结果. Covers resolving the matching inference (serve_policy) config, starting and stopping the policy server, running examples/robocasa/main.py per task, aggregating per-task success rates, and appending the summary table to the Feishu results doc."
argument-hint: '<checkpoint dir> ; <tasks> ; [training config]'
---

# RoboCasa 评测 + 结果记录

Given **a checkpoint path + which tasks to run + the training config**, this skill derives the matching
inference config, runs the eval, aggregates success rates, and records them.

## Inputs

| Input | Example | Required |
|---|---|---|
| checkpoint | `checkpoints/pi05_robocasa_state_noise/test_state_noise/18000` (step dir, or the exp dir to auto-pick the latest step) | yes |
| tasks | `OpenDrawer,CloseDrawer,OpenDoubleDoor,CloseDoubleDoor,CoffeeSetupMug,TurnOnStove` | yes |
| training config | `pi05_robocasa_state_noise` | optional — inferred from `checkpoints/<config>/<exp>/<step>` |
| scale overrides | `--num-trials`, `--num-envs`, `--seed`, `--replan-steps` | optional |

Env id for a task is `robocasa_panda_omron/<Task>_PandaOmron_Env` (141 task classes are registered; the
6 robocasa tasks used so far are OpenDrawer, CloseDrawer, OpenDoubleDoor, CloseDoubleDoor,
CoffeeSetupMug, TurnOnStove).

## Procedure

### 1. Build the plan (do not skip — this is where the inference config is derived)

```bash
cd <repo-root> && uv run python .github/skills/robocasa-eval/scripts/plan_eval.py \
  --ckpt <checkpoint dir> --tasks <t1,t2,...> --out runs/<run-id>/plan.json
```

The script resolves the training config, validates that the checkpoint exists, reads the persisted
`assets/norm_stats.json`, picks `num_envs` from the CPU/memory of the box, and prints the **serve
config** it will use plus the exact serve/eval commands. Read its output before continuing.

The `num_envs` rule is `min(12, cores // 4, available_memory // per_env_gb)` with `cores` from
`/sys/fs/cgroup/cpu.max` (25 here — **not** `os.cpu_count()`, which reports the 208 host cores) and
available memory as `memory.current - inactive_file`. If other jobs are hogging the box the plan can
come out as low as 1 env, which makes a 100-trial run drag; check the printed reason and either free
memory / `screen -X quit` the leftovers, or override explicitly with `--num-envs N` (and `--env-gb` if
your per-env memory assumption differs, e.g. 4 GB with video vs 2.5 GB with `--no-save-video`).

### 2. Derive the inference (serve) config — the important rule

`scripts/serve_policy.py --policy.config=<NAME>` must describe the *same architecture* the checkpoint
was trained with. Default: **reuse the training config name**. But check the plan output for:

* **Training-only augmentations** (`state_noise`, `prompt_drop_p`, `state_noise_beta_*`) — these must
  never apply at inference. Verified: `policies/policy_config.create_trained_policy` builds the
  inference transform list from `data_transforms` + `Normalize` + `model_transforms` only, so
  `RandomPromptDrop` / `InterpolatedStateNoise` (which live in `training/data_loader.transform_dataset`)
  are already skipped at serving time. Reusing the same config name is therefore safe — **but do not
  hand these flags to a hand-written inference config, and do not "fix" a config by copying them.** If
  the user asks for a derived inference config, drop them.
* **Architecture fields that must match the checkpoint**: `pi05`, `action_horizon`, `action_dim`,
  `max_token_len`, `discrete_state_input`, `paligemma_variant`, `action_expert_variant`. If the user
  hands you a config whose arch fields differ from the training config, stop and ask — a mismatch
  either fails to load or silently changes what the eval measures.

### 3. Start the policy server (in `screen` — never in a foreground terminal)

```bash
cd <repo-root>
screen -dmS openpi_serve_8000 bash -lc \
 'XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_FLAGS="--xla_gpu_deterministic_ops=true" \
  uv run python scripts/serve_policy.py --port=8000 policy:checkpoint \
  --policy.config=<CONFIG> --policy.dir=<CKPT> > logs/serve_8000.log 2>&1'
```

Health check: a plain `GET /` must answer **HTTP 426** (websocket endpoint). `ss -tlnp` prints nothing
in this container — check with `grep -i 1F40 /proc/net/tcp` or a real socket connect.
`run_eval.py --manage-server` does both automatically.

### 4. Run the eval

`run_eval.py` runs one task per `main.py` invocation, sequentially, and parses the harness's
`Total success rate: X% (n/N)` line.

```bash
cd <repo-root> && python3 .github/skills/robocasa-eval/scripts/run_eval.py --plan runs/<run-id>/plan.json
```

Notes that matter:

* The eval runs in the **robocasa venv**, not the openpi env:
  `/root/autodl-tmp/Isaac-GR00T/gr00t/eval/sim/robocasa/robocasa_uv/.venv/bin/python`.
* Every flag needs the `--args.` prefix (`--args.num-trials-per-task`, `--args.no-save-video`, ...).
* Long runs must live in `screen` (`screen -dmS eval_<id> bash -lc '... > logs/eval_<id>.log 2>&1'`),
  then poll the log / the summary file.
* `--args.one-episode-per-worker` (default true) and `--args.deterministic` (default true) must stay on
  when comparing checkpoints — scene generation is reproducible and action noise is pinned to
  `(episode seed, replan index)`, so **same seed + same trial count = comparable numbers**.
* Records land in `data/robocasa/<exp>/checkpoint-<step>/`: `videos/<task>/`, `results/<task>.tsv`,
  `logs/<task>.log`.

### 5. Record the results (Feishu)

Append a new section to the results wiki doc: **setup** (model path, training config, inference config,
eval params) + **results table** (per-task success counts + mean). Follow
[references/feishu-recording.md](./references/feishu-recording.md) — it has the verified browser
procedure (login requirement, virtualized-doc scrolling, clipboard paste, and the fact that a pasted
HTML table becomes a `blockType=sheet` embedded spreadsheet).

Always confirm with the user before writing to the shared doc.

### 6. Clean up

If `run_eval.py` started the server, it stops it (`screen -S <name> -X quit`). If you started it
manually, stop it yourself and say so.

## Outputs of one run

```
runs/<run-id>/plan.json          # resolved config/ckpt/tasks/paths/params
runs/<run-id>/summary.json       # machine-readable per-task results
runs/<run-id>/summary.md         # table to paste into the doc
data/robocasa/<exp>/checkpoint-<step>/{videos,results,logs}/
```

## Related

* Failure modes (workers SIGKILLed, "All worker processes died", NaN, video-less runs) →
  [references/troubleshooting.md](./references/troubleshooting.md)
* Background/design notes for this repo → `/memories/repo/openpi-groot.md`

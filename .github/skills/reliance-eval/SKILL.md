---
name: reliance-eval
description: "Measure how much an openpi RoboCasa checkpoint depends on each input modality (vision / proprioception / language / scene) and how much each input is needed to fit the data, then compare checkpoints on the joint vision+state reliance. Use when the user gives a checkpoint or experiment directory and asks to 评测依赖度 / 看依赖率 / 视觉依赖 / state依赖 / 联合依赖率 / 消融 / ablation / 判断模型到底有没有用视觉 / 是不是走了捷径 / compare checkpoints on reliance. Covers fixing the evaluation conditions (training config, action_horizon, state width) from the checkpoint itself, running scripts/diagnose_vision.py against a frame archive, and collating rows with scripts/reliance_table.py."
argument-hint: '<checkpoint dir> ; [training config] ; [frames archive]'
---

# 依赖度评测（vision / state / language / scene）

Given **a checkpoint**, this skill produces one row of evidence answering two different questions.

| Question | Columns | How to read |
|---|---|---|
| **Does the input matter at all?** Destroy it and see how far the predicted action moves. | `Vision` `State` `Lang` `Scene` | 0 = ignored, high = the action is driven by it. In units of the ground-truth action spread, so rows are comparable. |
| **Is the input needed to fit the data?** Destroy it and see how much worse the offline error gets. | `imgEr` `stEr` `langEr` `sceneEr` | 1.00 = the policy fits the data just as well without it (a shortcut); > 1.05 = load-bearing. |
| **Is the policy forced to use both channels?** | `min(V, S)` | High only when the policy needs *both* the image and its own proprioception. This is the property worth optimising. |

The two questions disagree, and that disagreement is the point: a policy can swing hard when
an input is removed (high `Vision`) and still fit the data without it (`imgEr` ≈ 1.00) — that is
a shortcut, not perception. Report both, never one alone.

Output = one row in a table (see `--csv`) plus `diagnostics/<label>/ablation.json` for the raw
per-dimension numbers. Same Feishu results doc as the `robocasa-eval` skill.

## Inputs

| Input | Example | Required |
|---|---|---|
| checkpoint | `checkpoints/pi05_robocasa/fix_checkpoint/15000` (step dir, or the exp dir to list steps) | yes |
| training config | `pi05_robocasa` — **usually readable from the path**, wrong config = the params do not load | no |
| frames archive | `diagnostics/frames_robocasa_7tasks_84frames.npz` (84 frames = 7 tasks x 2 episodes x 6 steps) | no — build one, see below |
| label | `fix_checkpoint_15000_ah50_dim16` — becomes the diagnostics dir name | yes |

The archive is what makes checkpoints comparable: every checkpoint sees **bit-identical frames**,
so a difference between rows is a difference between models. It is also why the ablation needs no
LeRobot dataset access (`HF_LEROBOT_HOME=/nonexistent` works, but the var must be set because
`training/config.py` reads it at import time).

## Procedure

### 0. Prerequisites

```bash
cd <repo-root>
export XLA_PYTHON_CLIENT_PREALLOCATE=false   # before importing jax, else it grabs all VRAM
export HF_LEROBOT_HOME=/nonexistent          # only read for --dump-frames, but must be set
```

Free VRAM no longer needs hand-tuning: **`--batch-size 0` (the default) sizes itself** from the
free memory at start-up and, if the GPU still runs out, halves the batch and retries the same
frames — so co-tenancy with a training job needs no guesswork. The rule is deliberately
pessimistic (weights = 3.22B params in bf16 = 6.4 GiB, plus ~0.8 GiB per frame at ah=50);
measured on this box: a batch of 8 raised `RESOURCE_EXHAUSTED` with 9.7 GiB free, a batch of 2
fitted in 8.9 GiB free. The effective batch size is recorded in `summary.json` and
`ablation.json`. A CUDA OOM kills only this process, never the co-tenant (and no longer the run).
Set an explicit `--batch-size N` only to cap the peak.

### 1. Fix the conditions (do not skip — two of the three has to come from the checkpoint)

```bash
uv run .github/skills/reliance-eval/scripts/ckpt_info.py --checkpoint <ckpt dir> \
  [--zero-state-keys joint_position]
```

It prints, from the run's own wandb config and the persisted `assets/norm_stats.json`:

- available steps, and the training config inferred from the path,
- `action_horizon`, `max_token_len`, the trained task list, and any state-noise / prompt-drop
  settings — **`action_horizon` must be passed as `--action-horizon`**,
- which state slots hold the **padding identity** (mean 0, std 1) — those are keys the model
  never saw, and the cause of a state-layout mismatch,
- a ready-to-run command.

### 2. Run the ablation

```bash
uv run scripts/diagnose_vision.py \
  --config-name pi05_robocasa \
  --checkpoint-dir <ckpt dir> \
  --action-horizon <training horizon> \
  [--zero-state-keys joint_position] [--state-dim-limit 16] \
  --frames-from diagnostics/frames_robocasa_7tasks_84frames.npz \
  [--frames-tasks <the tasks the checkpoint trained on>] \
  --batch-size 0 --no-run-features --no-run-occlusion \
  --output-dir diagnostics/<label>
```

Nine variants run in one pass (baseline, mask_all_images, mask_base, mask_wrist,
swap_other_episode, stale_image, blank_prompt, zero_state, predict_mean_reference). One shot of
flow-matching noise is shared by all variants with a fixed seed, so the comparison is paired.
84 frames x 9 variants takes ~5 min at batch 2.

Choose the input rewriting deliberately:

| Situation | Flag | Why |
|---|---|---|
| Checkpoint trained **without** a key (its stats slot is the padding identity) | `--zero-state-keys joint_position` | sends a constant, matching what the model saw |
| Same, and you want the **exact** training width | add `--state-dim-limit 16` | `Normalize` slices the stats to the given state length, so 16 dims = 16 stats = 16 prompt numbers, exactly as in training; zeroing alone adds one constant number per dropped slot |
| Checkpoint trained on fewer tasks than the archive | `--frames-tasks t1 t2 ...` | frames from an unseen task are a trap: the policy must guess the instruction from the image, which inflates `Vision` for the wrong reason |

### 3. Collate

```bash
uv run scripts/reliance_table.py                       # every diagnostics/*/ablation.json
uv run scripts/reliance_table.py --csv runs/reliance.csv
uv run scripts/reliance_table.py --gt-from ''          # fall back to each policy's own spread
```

`scripts/reliance_table.py` is the canonical table (ground-truth units, four deltas, four error
ratios, `min(V,S)`, verdict). `scripts/screen_checkpoints.py` is the older screener: same
ablations in units of each policy's **own** prediction spread, plus the SigLIP tower health
(`rank`, `within`, `CKA`) and the `PASS`/`balanced` verdicts. Use the first to compare
checkpoints, the second to inspect one.

### 4. Record

- `diagnostics/<label>/ablation.json` — per-dimension deltas and errors, the verdict, and (new
  runs only) `zero_state_keys` / `tasks_scored` / `action_horizon`.
- `diagnostics/<label>/summary.json` — the exact run conditions, including `state_dim_limit`.
- Append the row to the project's results/handoff doc. Keep `ah`, the task subset, the state
  conditioning and the frame count next to the numbers; a row without them cannot be compared.

## How to read the result

- **`min(V, S)` is the headline.** Across the 5-task RoboCasa family the two channels trade off
  against each other ($State \approx 0.33 - 0.60\,Vision$, $R^2 \approx 0.6$, Spearman ≈ −0.9):
  every recipe that raises one lowers the other. A row is interesting only if it sits *above*
  that line, i.e. keeps `State` while `Vision` is high.
- **A `State` of ~0.02 with `Vision` ~0.6** is the "pure vision" end (state-noise runs). **A
  `State` of ~0.45 with `Vision` ~0.04** is the original proprioception-only trajectory player.
- **`langEr` is nearly constant (1.03–1.16) across every checkpoint measured**, so it rarely
  discriminates; `imgEr` / `stEr` span 1.0–1.6 and do.
- **`Scene`** (swap in another episode's pixels, same task/state/prompt) is the object-perception
  proxy: 0.002 means "the pixels are wallpaper", 0.2 means the scene content drives the action.
- Fit quality: `err/mn` ≤ 0.95 means the action head actually fitted the behaviour (> 0.95: the
  verdicts are noise, fix the fit first).
- **Reproducibility floor ≈ 0.1 %** on the deltas (bf16 kernel scheduling differs run to run:
  the same command gave `Vision` 0.4878 and 0.4884). Never interpret differences below that.

## Pitfalls (all of these actually happened)

1. **Wrong `action_horizon`** silently degrades the fit: the same checkpoint at ah 20 vs its
   trained ah 50 went from `err/mn` 1.032 to 1.000 with `Vision` 0.037 → 0.040. Always pass the
   training value, and re-run a reference checkpoint at the same horizon if you are comparing
   across horizons.
2. **Ground-truth horizon ≠ prediction horizon** used to crash the script (`broadcast (84,50,21)
   vs (84,20,21)`); it now scores the shared prefix. If the archive was dumped at horizon 20 and
   the model predicts 50, the error columns only cover the first 20 steps.
3. **State layout**: never assume the standard key order. Read the padding identity out of the
   checkpoint's own `norm_stats.json` (`ckpt_info.py` does it). A checkpoint trained with a
   narrower state will read garbage in the slots it never saw.
4. **Task subset**: compare a checkpoint against a control measured on the *same* tasks. A
   checkpoint trained on 5 of the archive's 7 tasks must be scored with `--frames-tasks`.
5. **Ancestry**: two checkpoints where one is a fine-tune of the other are **not** two
   independent observations of a recipe. Check the run's `weight_loader` before treating rows as
   independent.
6. **Prompt truncation**: the pi05 prompt includes the state as text and can exceed
   `max_token_len` (`Token length (117) exceeds max length (112)`). The state is at the end, so
   truncation silently drops the tail of the state. A checkpoint trained at a smaller
   `max_token_len` saw a shorter state than you are feeding it.
7. **`--frames-from` ignores** `--tasks`, `--num-episodes-per-task`, `--num-frames-per-episode`
   and `--stale-offset`: sampling is baked into the archive. Task subsetting is `--frames-tasks`.
8. **Build a new archive only when the data changes**:

   ```bash
   uv run scripts/diagnose_vision.py --dump-frames diagnostics/frames_<desc>.npz \
     --tasks <t1 t2 ...> --num-episodes-per-task 2 --num-frames-per-episode 6   # pure CPU, ~1 min
   sha256sum diagnostics/frames_<desc>.npz    # record it; the file is the comparability contract
   ```

## What this does NOT measure

Simulator success rate (`robocasa-eval` skill) and closed-loop error correction. The ablation is
offline: it shows what the policy's action distribution depends on, not whether the robot
recovers when a grasp slips. A checkpoint can score well here and still fail in the environment,
and vice versa — report both.

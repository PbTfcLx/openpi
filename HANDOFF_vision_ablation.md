# Handoff: settle whether the vision collapse is a learning-rate problem

## If you are the agent receiving this

You have been given this file (or its text). Your job is **§4 Commands** + **§5 Decision
rule**: measure the counterfactual vision reliance (`VisionΔ`) of two already-trained
checkpoints and report whether a 3× lower learning-rate schedule preserves vision *use*.
Nothing here requires training. Read §1's boxed warning and §2's subtleties before you
interpret anything, and obey §5's decision table literally — the tower-health metrics
(`effective_rank`, `CKA`) are explicitly **not** valid evidence for this question.
Ask for the transfer in §4.1 before starting; do not attempt to re-download the checkpoints
from `gs://` (see §6).

## TL;DR — the one job

Run a **counterfactual input ablation** (measure `VisionΔ`) on **two already-trained
checkpoints**, and report two numbers. No training. Minutes of GPU, not hours.

- `checkpoints/pi05_robocasa/test_action_horizon/15000`  (peak_lr = 5e-5)
- `checkpoints/pi05_robocasa/test_new_lr/15000`          (peak_lr = 1.5e-4)

The second one is a **re-measurement of a known number** (should come out ≈ 0.036) and
exists only to make the comparison internally consistent on the new machine. The **new**
information is the first one.

If you want to skip all context, jump to **§4 Commands** and **§5 Decision rule**.

---

## 1. Background in five lines

- Model: `pi05` (PaliGemma `gemma_2b` + SigLIP So400m/14, plus a `gemma_300m` action expert),
  flow matching, 10 denoising steps, `action_dim=32`, `action_horizon=20`.
- Fine-tuned on RoboCasa LeRobot data. **pi05 puts the discretized 23-dim state into the
  prompt text**, and RoboCasa actions are **deltas** (`"absolute": false`). So the recorded
  behaviour is reproducible from proprioception alone → there is no gradient pressure to keep
  the vision tower informative → it decays.
- Measured symptom on `test_new_lr/15000`: masking all images moves the action by only
  **3.6 %** of the natural action spread, while zeroing the state moves it by **43 %**.
  The SigLIP patch features have collapsed (effective rank 14.9 vs 115.6 for `pi05_base`).
- A competitor explanation exists and has NOT been ruled out: peak_lr = 1.5e-4 may simply be
  too large for the fully-unfrozen SigLIP tower (NB: `Pi0Config.get_freeze_filter()` only
  matches `.*llm.*`, so the tower is **not** frozen even in the `*_lora` configs), and large
  updates may have destroyed the pretrained visual features.
- These two hypotheses are **H1 = learning rate** and **H2 = incentive/shortcut**. This
  handoff settles it.

### Why the tower's health cannot settle it (read this before interpreting anything)

H1 and H2 **both** predict "lower LR ⇒ healthier tower". A low-LR model can look healthy
**by inertia**: it has not yet taken enough steps to overwrite the pretrained features, while
the task never actually *used* them. Therefore:

> `effective_rank` / `CKA` / `within_frame_patch_sim` **cannot** discriminate H1 from H2.
> Only the counterfactual ablation can: **mask the images and see whether the action moves.**

---

## 2. Why these two checkpoints (verified, not assumed)

Their wandb configs (`wandb/run-*7wvgr2up` = `test_new_lr`, `wandb/run-*hawvr47z` =
`test_action_horizon`) were diffed field by field. **The only differences are the learning
rate schedule:**

| field | `test_new_lr` | `test_action_horizon` |
|---|---|---|
| `lr_schedule.peak_lr` | 1.5e-4 | **5e-5** |
| `lr_schedule.warmup_steps` | 600 | 2000 |
| `lr_schedule.decay_lr` | 1.5e-5 | 5e-6 |
| `lr_schedule.decay_steps` | 17000 | 58000 |

Identical: `model.action_horizon = 20`, `model.max_token_len = 112`, `batch_size = 64`,
no state noise, no prompt dropout, and **the same five datasets**. This is the cleanest
single-variable LR contrast available in this project.

### A subtlety you must not get wrong

At step 15000 the two runs' **instantaneous** LRs have **crossed over**, because
`test_new_lr` decays 3.4× faster:

| | instantaneous LR @ step 15000 | cumulative Σ LR (×1e-4·step) |
|---|---|---|
| `test_new_lr` (peak 1.5e-4) | 1.99e-5 | **13647** |
| `test_action_horizon` (peak 5e-5) | 4.43e-5 | **6748** |

So "step-matched" means **the high-peak run has accumulated 2.02× the total update**, not
"one trains at a higher LR at step 15000". Describe the contrast that way.

### Do NOT use `test_token_len/24000` for this

It is **confounded** and must not be substituted: vs `test_new_lr` it also differs in
`model.action_horizon` (50 vs 20) and `model.max_token_len` (104 vs 112). `action_horizon=50`
is a *directional* confound — predicting a 50-step action chunk is harder from the state
alone than 20 steps, so it should raise vision reliance for reasons unrelated to the LR.
Also its diagnostic run (`diagnostics/token_len_24000/`) contains **no `ablation.json`**,
so its `VisionΔ` has never been measured.

---

## 3. The 2×2 this fills in

|  | peak_lr = 1.5e-4 | peak_lr = 5e-5 |
|---|---|---|
| **no state noise** | `test_new_lr/15000`: VisionΔ **0.036**, rank 14.9 *(measured)* | **`test_action_horizon/15000` ← the empty cell, this job** |
| **state noise** | `test_state_noise/18000`: VisionΔ **0.753**, rank 35.2 *(measured)* | (not needed) |

(`pi05_robocasa` and `pi05_robocasa_state_noise` were diffed and differ **only** in
`data.state_noise`; `action_horizon`/`max_token_len` are 20/112 in both, so the bottom-left
cell is a clean noise-only contrast.)

---

## 4. Commands

### 4.1 What has to be on the new machine

The 84 frames the ablation needs have been dumped to a **53 MB `.npz`**, so the 78 GiB of
LeRobot datasets do **not** have to be transferred at all.

| item | size (measured) | note |
|---|---|---|
| this repo source tree, minus `checkpoints/ data/ wandb/ diagnostics/ .venv/` | **913 MB** | includes `third_party/` (524 MB). Do **not** copy `.venv/` (9.4 GB, machine-specific) — recreate it with `uv sync`. |
| `diagnostics/frames_robocasa_7tasks_84frames.npz` | **53 MB** | the frame archive, see below. sha256 `bb9a853116ab6a576ce3c81ef8653a86c25adde146c83a2a869bfffaf5040883` |
| `checkpoints/pi05_robocasa/test_action_horizon/15000` | **12 GiB** | the new measurement |
| `checkpoints/pi05_robocasa/test_new_lr/15000` | **12 GiB** | the re-measurement (see §5) |
| the LeRobot datasets | **not needed** | only if you want to re-dump frames |

Each checkpoint is 12 GiB of **fp32 params** (`params/`) plus a tiny `assets/norm_stats.json`.
**Both checkpoints contain `assets/norm_stats.json` — verified**, and the two files are
**byte-identical** (sha256 both start `f33aec5d40fdd6b9`, 0 numeric differences), which is why
one archive serves both checkpoints. The diagnostic does not need the optimizer state, so
`train_state/` is not needed.

**Contents of the archive** (verified by round-trip against the datasets — 0 mismatches over
all 84 frames × 3 image sets × 3 cameras, key/state/prompt/gt_chunk/probe included):

- `images` / `swap_images` / `stale_images`: `(84, 3, 256, 256, 3)` **uint8 RGB**, camera order
  `[base_0_rgb, base_1_rgb, left_wrist_0_rgb]`. Note these are the **raw 256×256** frames; the
  resize to 224 happens downstream in the model transforms.
- `state` `(84, 23)` float32, `prompt` `(84,)` str, `gt_chunk` `(84, 20, 21)` float32,
  `probe` / `probe_names`, `key` / `task` / `episode` / `step`, and a `meta_json` manifest.
- 7 tasks × 12 frames (2 episodes × 6 frames each) = **84**, matching the archived
  `diagnostics/test_new_lr_15000/ablation.json` (`num_frames: 84`).

How it was produced (already done, listed for provenance — needs the datasets):

```bash
export HF_LEROBOT_HOME=/root/autodl-tmp/g00t/datasets/PhysicalAI-Robotics-GR00T-X-Embodiment-Sim
uv run python scripts/diagnose_vision.py \
  --config-name pi05_robocasa \
  --dump-frames diagnostics/frames_robocasa_7tasks_84frames.npz
```

That takes ~43 s of **CPU only** and loads no model (it also runs fine on a box whose GPU is
busy). Do not pass `--tasks`: the default discovery of all 7 datasets is what produced 84.

### 4.2 Run it

```bash
# ``openpi/training/config.py`` reads os.environ["HF_LEROBOT_HOME"] AT IMPORT TIME, so it must
# be set — but the archive path never touches the files, so a placeholder is fine.
export HF_LEROBOT_HOME=/nonexistent/placeholder
export XLA_PYTHON_CLIENT_PREALLOCATE=false          # MUST be set before python imports jax
cd /path/to/openpi

for CKPT in test_action_horizon/15000 test_new_lr/15000; do
  uv run python scripts/diagnose_vision.py \
    --config-name pi05_robocasa \
    --checkpoint-dir checkpoints/pi05_robocasa/$CKPT \
    --frames-from diagnostics/frames_robocasa_7tasks_84frames.npz \
    --batch-size 8 \
    --no-run-features \
    --no-run-occlusion \
    --output-dir diagnostics/$(echo $CKPT | tr / _)
done
```

Notes:
- `--run-ablation` defaults to **True**; `--run-features`/`--run-occlusion` default to
  True/False. The `--no-run-*` forms are tyro's bool negation. Skipping `--run-features`
  also means you do **not** need the 12 GiB `pi05_base` reference checkpoint.
- `--frames-from` **ignores** `--tasks`, `--num-episodes-per-task`, `--num-frames-per-episode`
  and `--stale-offset` (it warns about this) — the sampling is baked into the archive. It is
  deliberately read through the normal pipeline, so the checkpoint's own `Normalize` is still
  applied and every variant (including `mask_*`, `zero_state`, and `--run-occlusion`) still
  works, and new variants can still be added.
- Verified without a GPU: `--frames-from` runs to `summary.json` with a **non-existent**
  dataset root and never calls task discovery. The only unverified part is the model forward
  pass itself, which needs the GPU.

### 4.3 Read the numbers

```bash
uv run python -c "
import json
d = json.load(open('diagnostics/test_action_horizon_15000/ablation.json'))
r = d['verdict']['reliance_over_baseline_spread']
print('frames=%d  VisionD=%.3f  StateD=%.3f  LangD=%.3f  err/mean=%.2f' % (
    d['num_frames'], r['vision'], r['proprioception'], r['language'],
    d['baseline_error_vs_predict_mean']))
print('excluded_action_dims =', d['excluded_action_dims'],  # expected [7, 8, 9, 10, 11]
      '(these are identically zero in the data; do NOT \"fix\" this)')
print(d['verdict']['vision'])
"
```

Key names: `verdict.reliance_over_baseline_spread.vision` = **VisionΔ** (action movement
when all images are masked, as a fraction of the natural action spread);
`.proprioception` = **StateΔ**; `.language`; `baseline_error_vs_predict_mean` = offline error
÷ the error of always predicting the training mean (1.0 = the model has fitted nothing).
`variants.mask_all_images.delta_phys_over_spread` is the same VisionΔ number.

---

## 5. Decision rule (the point of the whole exercise)

Compare **VisionΔ** of the two runs, and read the tower's `effective_rank` as a *secondary*
signal only.

| outcome | interpretation |
|---|---|
| VisionΔ(`test_action_horizon/15000`) ≈ **0.4 – 0.75** | **H1 confirmed**: the learning rate was destroying vision. A ~3× lower peak LR / 2× smaller cumulative update preserves vision *use*. Fix = lower the LR (and re-tune the LR schedule). |
| VisionΔ ≈ **0.03 – 0.08**, while `effective_rank` is clearly **> 14.9** (i.e. the tower looks much healthier than the collapsed one) | **H2 confirmed**: a **healthy but unused** tower — health by inertia. The LR explanation is **excluded**, and the incentive/shortcut diagnosis stands. This is the predicted outcome. |
| VisionΔ in between (~0.1–0.3) | Both effects are present; report it as such and do not over-claim. |

Report, for each run: `VisionΔ`, `StateΔ`, `LangΔ`, `err/mean`, `effective_rank`, and
`excluded_action_dims`. A bare VisionΔ without the rank is not interpretable.

### Validity warning — this is why both runs must happen on the same machine

The archived `diagnostics/test_new_lr_15000/ablation.json` (VisionΔ = 0.036, 84 frames) was
produced by sampling `2 episodes × 6 frames per task` **spread evenly across each dataset**.
A dataset with different episode counts yields different frames, so archival numbers are not
comparable across machines in general.

**The archive removes this risk for this job**: `--frames-from` feeds both runs bit-identical
inputs, so the two numbers are directly comparable whatever is on disk. Still run **both**
checkpoints in one sitting and compare those two numbers; use the archived 0.036 only as a
sanity check on the `test_new_lr/15000` re-run — if it does not land near 0.036 something is
wrong upstream, not merely a different dataset.

---

## 6. Environment gotchas (learned the hard way)

- **`XLA_PYTHON_CLIENT_PREALLOCATE=false` must be exported before Python imports JAX**,
  otherwise the script OOMs/starves any other job sharing the GPU.
- `HF_LEROBOT_HOME` must be **set** (to anything, e.g. `/nonexistent`) because
  `openpi/training/config.py` reads it **at import time** via module-level `get_ds_meta` calls.
  That function only concatenates a path string and never touches the files, so a placeholder
  is fine when using `--frames-from`; only `collect_samples` needs the real datasets, and it
  globs `single_panda_gripper.*` under that root.
- `OPENPI_DATA_HOME` is only needed to resolve `gs://` paths (`pi05_base` assets). Not needed
  for this job if you skip `--compare-checkpoint-dir`.
- Datasets are read directly (pyarrow for the parquet, **cv2 for the mp4s — cv2 returns BGR
  and the script flips it to RGB**). Prompt text comes from
  `tasks.jsonl[parquet annotation.human.action.task_description]` (that column stores an
  **int index**). GT action ordering must match `GrootOpenpiSingleDataset.__getitem__`.
- Raw decoded frames are **256×256**; the 224×224 resize happens later, in the model
  transforms. (Only the synthetic fallback for a missing camera is 224.)
- `excluded_action_dims` = `[7, 8, 9, 10, 11]`: `base_motion` (7:11) and `control_mode` (11)
  are **identically zero** in every dataset, so their norm-stats std is 0 and normalized
  error is `0/(0+1e-6)` = meaningless. The script drops them automatically. With them
  included, baseline error looked like 2.11× the mean predictor instead of 1.03×.
- Probe numerics (only relevant with `--run-features`): use **robot-frame** targets
  (`joint_position`, `gripper_qpos`), not world-frame ones — RoboCasa randomises the layout
  per episode, so `end_effector_position_absolute`/`base_position` are out of range on a
  held-out episode and score R² ≈ −1e7. Use a cosine k-NN decoder, not ridge (D ≫ N).
- PyTorch/optimizer state is irrelevant here; `params/` (12 GiB fp32) is all that is read.

---

## 7. What this does NOT measure

`VisionΔ` is **behavioural reliance on the recorded frames**. It is not task success and not
object awareness. A model can have high VisionΔ and still mechanically replay a trajectory.
Do not report it as a success-rate proxy.

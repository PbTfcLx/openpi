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

**VisionΔ is the decision variable. `effective_rank` is corroborating evidence only** — it
cannot pick between H1 and H2 on its own, because both predict a healthier tower at a lower
LR. Do not gate the verdict on the rank.

| outcome | interpretation |
|---|---|
| VisionΔ(`test_action_horizon/15000`) ≈ **0.4 – 0.75** | **H1 confirmed**: the learning rate was destroying vision. A ~3× lower peak LR / 2× smaller cumulative update preserves vision *use*. Fix = lower the LR (and re-tune the LR schedule). |
| VisionΔ ≈ **0.03 – 0.10**, with a *partially preserved* tower | **H2 confirmed, and more strongly than a fully collapsed tower would**: feature directions survive (2.8× the collapsed value at `encoded`) and are **still not used**. That decouples representation health from behavioural reliance, which is exactly the H2 claim. The LR explanation is **excluded**; the incentive/shortcut diagnosis stands. This is the predicted outcome. |
| VisionΔ in between (~0.1–0.4) | Both effects are present; report it as such and do not over-claim. |

### Which layer, and the reference numbers (asked more than once)

**The headline `effective_rank` figures are the `encoded` layer**: the SigLIP tower's output
after all 27 encoder blocks, *before* the projection into the LLM's token space. Camera
`base_0_rgb`, 84 frames. Do **not** quote `with_posemb` — the write-up's 14.9 / 115.6 are
`encoded` (exactly 14.86 / 115.64).

Measured on **the same 84 frames in the archive**, so these reference values may be compared
against a run on any machine (the archive guarantees identical frames; `effective_rank`
depends only on the frames and the weights):

| layer | `test_new_lr/15000` (collapsed) | ratio to base | `pi05_base` | what it is |
|---|---|---|---|---|
| `with_posemb` | 6.91 | 0.68× | 10.15 | tower input (stem + positional embedding) |
| `block13` | 13.14 | **0.077×** | 169.62 | encoder block 13 output (`"+mlp"`) |
| `block20` | 12.70 | **0.068×** | 186.22 | encoder block 20 output |
| **`encoded`** | **14.86** | **0.129×** | **115.64** | tower output, pre-projection ← the headline |
| `tokens` | 6.52 | 0.094× | 69.61 | projected token sequence the LLM consumes |

`block20` shows the *largest* relative collapse (14.7×) and `encoded` the most
interpretable one (7.8×); quote whichever, but never mix layers between checkpoints.

Rank is a sample-dependent *absolute* quantity, so always report it as a **ratio to a
`pi05_base` measured in the same session** rather than against a hard-coded threshold — the
old "> 14.9" gate was too lax to discriminate anything. If `pi05_base` was not run on that
machine, the column above is a valid reference **provided `--frames-from` was used**.

`effective_rank` = participation ratio of the centred patch-feature covariance: reshape
`(B, 256, C)` → `(256B, C)`, centre across that axis, SVD, square the singular values,
normalise to a distribution `p`, return `exp(-Σ p log p)`. It therefore measures how many
directions the patch features span **across the whole frame set**, not within one frame — for
the spatial-within-frame question use `within_frame_patch_sim` (mean pairwise cosine
similarity of the 256 patches inside a frame, averaged over ≤32 frames): 0.9998 collapsed vs
0.4634 for `pi05_base` at `encoded`.

Note in `features.json` the CKA values live on the **reference** entry as
`cka_to_reference` (the fine-tuned entry shows `-`). CKA is symmetric, so they are the
fine-tuned-vs-base numbers either way.

Report, for each run: `VisionΔ`, `StateΔ`, `LangΔ`, `err/mean`, `effective_rank` at all
layers, `num_frames` (must be 84 when using the archive), and `excluded_action_dims`.

### Proving two runs are comparable, without sharing a machine

The archive pins the frames, so numbers from different machines **are** comparable — but
prove it rather than assume it. `ablation.json` contains two quantities that depend only on
the ground truth and **not on the model**, so they are a bit-exact fingerprint of the frame
set:

- `predict_mean_err_norm_mean` — must equal `0.6628513781264901`
- `baseline_action_spread` — must equal
  `[0.255950003862381, 0.21372392773628235, 0.17233122885227203, 0.06084089353680611,`
  `0.07598678767681122, 0.07035878300666809, 0.5723248720169067, 4.37e-09, ...]`
  (12 values; `action_horizon=20`, `num_steps=10`, `num_frames=84`)

If both match, the two runs saw identical inputs and every number is directly comparable. If
they do not, stop and re-dump — something differs in the sampling.

There is also **no need to transfer a second checkpoint just to get a baseline**: the
`test_new_lr/15000` reference (VisionΔ 0.036) was measured on exactly these archived frames.

### The landscape so far (all on the same 84 frames)

**Ancestry matters — these are not independent points.** Every row is fine-tuned directly from
`pi05_base` except the child row: `lora_new_lr/11999` is a LoRA fine-tune **of
`test_token_len/24000`** and *inherits* its tower, so it must not be counted as a separate
observation (a mistake that was made once and corrected).

| checkpoint | lr | noise | ah | step | err/mean | VisionΔ | StateΔ | LangΔ | rank @ `encoded` |
|---|---|---|---|---|---|---|---|---|---|
| `test_new_lr/15000` | 1.5e-4 | – | 20 | 15000 | 1.032 | 0.036 | 0.432 | 0.188 | 14.86 |
| `test_action_horizon/15000` | 5e-5 | – | 20 | 15000 | 0.919 | 0.181 | 0.327 | 0.116 | 42.0 |
| `test_token_len/24000` | 5e-5 | – | **50** | 24000 | 0.947 | 0.203 | 0.318 | 0.118 | 52.56 |
| `w020/3000` | 1.5e-4 | ✓ | 20 | 3000 | 0.851 | 0.323 | 0.115 | 0.154 | 25.9 |
| `w020/6000` | 1.5e-4 | ✓ | 20 | 6000 | 0.866 | 0.390 | 0.100 | 0.156 | 24.4 |
| `state_noise/18000` | 1.5e-4 | ✓ | 20 | 18000 | 0.827 | 0.753 | 0.028 | 0.171 | 35.2 |
| ↳ `lora_new_lr/11999` *(child of the row 3 above)* | LoRA | | 50 | 11999 | 0.856 | 0.213 | 0.327 | 0.156 | 44.02 |

`test_token_len/24000` was measured and settles the question: it has the **healthiest tower of
any fine-tuned model here (52.56)** yet VisionΔ is only **0.203**. Note it was *trained* with
`action_horizon=50, max_token_len=104` but is *evaluated* at horizon 20 — forced, not chosen:
the archive's `gt_chunk` is `(84, 20, 21)` and `run_ablation` only broadcasts at
`horizon == 20`. Confirmed that `action_horizon`/`max_token_len` appear in no parameter shape
(`pi0.py` uses them only for runtime tensors), and this checkpoint had already loaded under
`--config-name pi05_robocasa` before. Two consequences: its `err/mean` is inflated by the
train/eval mismatch, and the mismatch is *conservative* for H2 — a 50-step objective should
need vision **more** than a 20-step one, yet the reliance is still low.

1. **Tower health has a real but small and saturating effect; the incentive dominates.** Do not
   quote a correlation coefficient — it swings +0.09/+0.30/+0.40 depending on which points are
   kept, and n ≤ 6 makes all of them meaningless. The honest picture is three-tiered:

   - *Within the three "clean" (no state corruption) runs*, VisionΔ **is** monotone in rank:
     `14.86 → 0.036` < `42.0 → 0.181` < `52.56 → 0.203`. So health does contribute.
     **But this trio also varies in LR, step count and `action_horizon` simultaneously**, so it
     does not establish that *health* is the cause — "gentler/longer training" moves both
     quantities together. Treat it as consistent with H1's mechanism, not as proof of it.
   - *The trend saturates and flattens*: +0.145 for the first 27 rank units, then only +0.022
     for the next 10.5. Extrapolating to a fully healthy tower (rank ≈ 115, i.e. `pi05_base`)
     still lands well under 0.3 — an extrapolation, not a measurement, but it puts the health
     knob's *ceiling* below what the incentive knob already delivers.
   - *The two knobs' reach, at the same starting point*: from `test_new_lr/15000` (rank 14.9,
     0.036), the **incentive** knob (add state noise, same LR) reaches 0.753 — a **×20.9**
     change; the **health/LR** knob reaches 0.203 — **×5.6**. The incentive lever reaches ~3.7×
     further, and it does so while *lowering* tower health.

   And the pairwise inversion still holds and is ancestry-clean: `test_action_horizon/15000`
   (rank 42.0) → 0.181 versus `state_noise/18000` (rank 35.2) → **0.753**, i.e. 4.2× more
   reliance from the *worse* tower. Likewise the parent→child pair
   (`test_token_len/24000` 52.56 → 0.203 versus its LoRA child 44.02 → 0.213) shows a **16 %
   rank drop with no change in reliance** (Δ = 0.010, inside the noise).

   Correct conclusion: **"health has a positive influence but is far from sufficient; the
   dominant factor is the incentive/shortcut"** — not "health is irrelevant".

2. **Vision and proprioception reliance trace one curve.** Spearman(VisionΔ, StateΔ) = **−1.00**
   across the six ancestry-clean points (−0.96 including the child, whose single 0.009 StateΔ
   inversion is well inside noise). Turning the LR down and corrupting the state both slide the
   model along the *same* state↔vision axis. Two independent knobs, one curve ⇒ the curve is a
   property of the task and its incentive structure, not of any hyper-parameter.

3. **Language reliance is a negative control — and it is never the dominant channel.** On the
   GT-normalised scale LangΔ sits in 0.100–0.208, against VisionΔ 0.037–0.642 (17×) and StateΔ
   0.021–0.466 (22×). **Refinement:** that 2.1× LangΔ spread is driven entirely by the floor
   model — among the seven checkpoints with VisionΔ > 0.1, LangΔ spans **0.100–0.133 (1.3 %)**, so
   it really is ~invariant to the knobs. The floor model is the outlier at 0.208, its *highest*
   value: the model that ignores images leans **more** on the prompt. Ranked within each
   checkpoint, language is #2 in 4 of 8 and #3 in 4 of 8 — **never #1**, but never negligible
   either (10–21 % of the GT action spread).

   **Caveat that must be stated: `pi05_base` has never been ablated.** It appears in
   `features.json` only as the CKA/PCA *reference*, so every LangΔ number here describes a
   **fine-tuned** RoboCasa checkpoint. There is no "before fine-tuning" measurement, and hence
   **no evidence in this project that full-weight fine-tuning reduced language reliance** — the
   question simply has not been measured.

   Measuring it is cheap but needs care: `openpi-assets/checkpoints/pi05_base/` holds only
   `params` and `params.lock`, so
   `diagnose_vision.py` would log "`no assets/ directory; normalization will be a no-op`" and
   **silently run with `Normalize` disabled**, making `zero_state` meaningless. Fix by symlinking
   `params` into a directory alongside a fine-tuned checkpoint's `assets/norm_stats.json` (all
   eight are byte-identical, verified), so the base model and the fine-tunes go through the same
   normalisation. The reading is still limited: pi05_base never saw RoboCasa, so its images are
   out-of-distribution and its predicted actions are off-task — the number is a *sensitivity*
   baseline, not a capability one.

   Note `blank_prompt` removes only the task description — the prompt is assembled as
   `f"Task: {cleaned_text}, State: {state_str};\n"` (`tokenizer.py`), so the state text survives
   being blanked and `LangΔ` is not contaminated by `StateΔ`.

   Note this axis is the robust finding; the *position* of any single point on it is worth
   about ±0.7 % on the rank side (the same `pi05_base` reference came out 115.64 in one run and
   114.88 in another, from batching/float differences), so only differences well above that
   should be read.

So a mid-range VisionΔ is not an inconclusive result; read it as **a position on the curve plus
a health reading**, and note that LR is a second-order lever: 3× lower peak LR bought a 5.0×
increase in VisionΔ (0.036 → 0.181) and 2.8× in rank, but recovered only ~1/4 of the distance
to the vision-reliant end (0.753) and left the model still state-dominated (StateΔ 0.327 >
VisionΔ 0.181).

### Interpreting `err/mean` (the tool says so itself)

`fit_quality` in `ablation.json` switches bands at ratio 1.0 and 0.7: `> 1.0` → "no better than
predicting the mean, any modality question is premature"; `> 0.7` → "only slightly better than
a constant, expect the modality verdicts to be noisy"; otherwise trustworthy.

Note that **every** run in the landscape above except the first sits in the 0.7–1.0 band
(0.827–0.919). That caps the resolution of this comparison: treat differences of a few × and
the *inversions* (rank vs use, knob vs curve) as the signal, and do not over-read gaps like
0.323 vs 0.390. A more converged checkpoint (`test_action_horizon/21000` and `27000` exist)
would raise the fit and is worth measuring if a tighter comparison is needed — and if
VisionΔ *falls* with more training at the same LR, that is direct evidence for the
incentive mechanism: the model progressively abandons vision as it exploits the
proprioceptive shortcut.

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

## 7. What each number means, and why the scales differ so much

The four families are **different kinds of quantity**, which is the whole reason some sit below
1 and some above 100.

| quantity | what it is | units | sensible range |
|---|---|---|---|
| `VisionΔ` / `StateΔ` / `LangΔ` (= `delta_phys_over_spread`) | **ratio**: how far the action moves when that input is destroyed, divided by how much the policy's own output varies naturally | dimensionless | 0 – ~1.5 |
| `err/mean` (= `baseline_error_vs_predict_mean`) | **ratio of errors**: the model's offline error ÷ the error of always predicting the training mean. 1.0 = has fitted nothing | dimensionless | ~0.8 – 1.1 |
| `effective_rank` | **a count of directions**: participation ratio `exp(−Σ p log p)` of the centred patch-feature covariance's eigenvalue spectrum | dimensions | 1 … `min(N_frames·256, width)` = 1152 |
| `within_frame_patch_sim` / `same_episode_sim` / `cross_task_sim` | **cosine similarities** between feature vectors | dimensionless | −1 … 1 |

So an `effective_rank` of 186 is not "186×" anything — it says the patch features spread over
≈186 directions out of a 1152-dimensional space (16 % of the available capacity). A collapsed
tower uses 15. Nothing exceeds 1152 however healthy the tower, because that is the feature
width; and the ceiling is really `min(84·256, 1152) = 1152`.

The reliance ratio is defined as
`mean_frames,steps |a_variant − a_baseline| / std_frames,steps(a_baseline)`, per action
dimension, then averaged over dimensions — so its natural "large effect" reference point is
**1.0**, not 0.5.

> **Scale caveat, and the recommended fix — two separate problems.**
>
> *(a) A counting bug.* `delta_phys_over_spread` is computed as
> `ratio = np.zeros(12); ratio[informative] = …; np.mean(ratio)`, i.e. it sums **7** informative
> ratios but divides by **12**, diluting by `7/12 = 0.583`. The sibling statistics in the same
> function (`delta_phys_mean`, `err_norm_mean`) *are* informative-only. So this is an
> inconsistency in the tool, not in the physics.
>
> *(b) A model-dependent denominator.* The normaliser is `std` of the **baseline predictions**
> (across frames × horizon) — a property of the *model*, so it is not the same yardstick for
> every checkpoint. Measured: `test_new_lr/15000`'s own prediction spread is 1.2–1.3× larger
> than every other checkpoint's. The principled reference is the **ground-truth action spread**,
> which is a property of the data and identical for all checkpoints (computed from the archive,
> no model needed): `[0.426, 0.441, 0.332, 0.127, 0.123, 0.156, 0.390]` for the 7 signal dims.
>
> **Do not "fix" (a) by multiplying by `12/7`** — that over-corrects, because (b) pushes the
> other way. The three quantities are:
>
> | | definition | `test_new_lr` | `w020/6000` | `state_noise/18000` |
> |---|---|---|---|---|
> | as reported | `mean_12(Δ / own_spread)` | 0.036 | 0.390 | 0.753 |
> | dims-only fix | `mean_7(Δ / own_spread)` = reported × 12/7 | 0.062 | 0.669 | 1.291 |
> | **GT-normalised (use this)** | `mean_7(Δ / gt_spread)` | **0.037** | **0.391** | **0.642** |
>
> Remarkably the two errors **nearly cancel** for a typical checkpoint, so the as-reported
> numbers land within 0–17 % of the principled ones (`state_noise/18000` is the worst case for
> the vision channel: 0.753 reported vs 0.642). Ordering and every conclusion are unchanged —
> Spearman(VisionΔ, StateΔ) = −0.94 on GT-normalised values versus ≈ −1.00 on reported ones.
>
> Practical guidance: for the write-up prefer **`mean_7(Δ / gt_spread)`**, and if you quote the
> as-reported numbers instead, say that the normaliser is the model's own output spread. Adding
> `delta_phys_over_gt_spread` to the tool (keeping the existing key) is the clean fix; until
> then a checkpoint's GT-normalised value can be recovered from its `ablation.json` plus the GT
> spread vector above, since `delta_phys_per_dim` is stored per variant.
>
> **Full GT-normalised table — all 8 checkpoints, same 84 frames** (transcription of the two
> off-machine runs verified against their own reported means to ~1e-7 relative):
>
> | checkpoint | VisionΔ | StateΔ | LangΔ | (reported VisionΔ) | err/mean |
> |---|---|---|---|---|---|
> | `test_new_lr/15000` | 0.037 | 0.466 | 0.208 | 0.036 | 1.032 |
> | `test_action_horizon/15000` | 0.185 | 0.300 | 0.100 | 0.181 | 0.947 |
> | `lora_new_lr/11999` *(child of the row below)* | 0.204 | 0.297 | 0.133 | 0.213 | 0.856 |
> | `test_token_len/24000` | 0.215 | 0.322 | 0.118 | 0.203 | 0.947 |
> | `w020/3000` | 0.317 | 0.080 | 0.109 | 0.323 | 0.851 |
> | `w020/6000` | 0.391 | 0.072 | 0.115 | 0.390 | 0.866 |
> | `state_noise/12000` | 0.585 | 0.020 | 0.110 | 0.708 | 0.841 |
> | `state_noise/18000` | 0.642 | 0.021 | 0.128 | 0.753 | 0.827 |
>
> Two consequences of moving to the principled scale:
>
> * **The trade-off curve is strong but no longer perfectly monotone.**
>   Spearman(VisionΔ, StateΔ) = **−0.90** on GT-normalised values, versus **−1.00** on the
>   reported ones (six ancestry-clean points). The small inversions are of the kind you would
>   expect to be noise — `test_token_len/24000` (0.215 → 0.322) against its own LoRA child
>   `lora_new_lr/11999` (0.204 → 0.297), and `state_noise/18000` (0.642 → 0.021) against an
>   earlier step of the *same* run (0.585 → 0.020) — but at least part of the reported −1.00
>   was an artifact of the model-dependent denominator. **Do not present the reported −1.00 as
>   a law.** The trend is still very strong: VisionΔ spans 0.037–0.642 (17×) while StateΔ moves
>   monotonically opposite.
> * **The denominator distortion is dimension-dependent, not a global factor.** Per-dimension
>   `own_spread / gt_spread` is **0.32–0.65 on the six arm dims but 1.16–1.47 on dim 6
>   (`gripper_close`), for every one of the eight checkpoints.** So on the arm the models are
>   *under*-dispersed relative to the demos, and on the gripper they are *over*-dispersed —
>   which is exactly why a single "multiply by 12/7" style correction cannot be right. It also
>   means the gripper channel is where the models' own variability exceeds the data's, so the
>   arm dims and the gripper dim get different weights under the two scalings.
>
> Two-knob comparison on the corrected scale, from `test_new_lr/15000` (VisionΔ 0.037):
> the **incentive** knob reaches 0.642 (**×17.4**) while the **health/LR** knob reaches
> 0.215 (**×5.8**) — the incentive lever reaches ~3× further, and it does so while *lowering*
> tower health. The language control is unchanged and still clean: LangΔ spans 0.100–0.208
> (2.1×) against VisionΔ's 17×.

>
> Note the normaliser is **not** removable stochasticity. The flow-matching noise is already
> deterministic *and shared across variants* — `noise = [noise_for(s.key, horizon, action_dim,
> seed) for s in samples]` is built once and passed to `sample_actions` identically for the
> baseline and every counterfactual, with a single fixed `rng = jax.random.key(args.seed)`. So
> the baseline/variant comparison is already perfectly paired; the "spread" being divided by is
> the policy's output variation *across different frames and scenes*, i.e. the scale of the
> task's action space — the unit, not noise. Dividing by it is what makes a bare Δ meaningful.


## 8. Does VisionΔ actually matter? (task success, 5 tasks × 100 episodes each)

**Ancestry discipline applies here too.** `test_lora_new_lr/*` is a LoRA child of
`test_token_len/24000`, so it is **not an independent point**: its VisionΔ (0.204) is its
parent's (0.215) inherited rather than produced by the LoRA run. (Three separate times an
inherited checkpoint has been mistaken for an independent observation in this project — always
check `checkpoint_dirs` and the parent run before adding a row.) Now that the parent's success
rate exists, the pair can be read as a **paired comparison** instead of a new row, and it is a
clean null: parent 61.0 % → child 62.2 % (**+1.2 pts, 0.39 σ**) with VisionΔ 0.215 → 0.204.
Fine-tuning with LoRA changed neither measurable quantity.

**Four independent runs** have both a measured VisionΔ and a measured success rate:

| checkpoint | intervention | VisionΔ (GT-norm) | success | err/mean | per-task (5 × 100 eps) |
|---|---|---|---|---|---|
| `test_new_lr/15000` | peak_lr 1.5e-4 | **0.037** | **48.0 %** | 1.032 | 52, 18, 69, 99, **2** |
| `test_action_horizon/15000` | peak_lr 5e-5, **no noise** | **0.185** | **60.6 %** | 0.947 | 66, 28, 80, 100, 29 |
| `test_token_len/24000` | peak_lr 5e-5, **no noise**, ah 50 | **0.215** | **61.0 %** | 0.947 | 78, 28, 79, 100, 20 |
| `test_state_noise/18000` | + state noise | **0.642** | **63.2 %** | 0.827 | 81, 27, 82, 100, 26 |

(Also measured, same run as the first row: `test_new_lr/18000` = 51.0 % — a within-run
consistency check, not a new point.) `Spearman(VisionΔ, success) = +1.00` across the four, and
the pre-registered prediction of ≈61–62 % for `test_token_len/24000` was confirmed at 61.0 %.

**The state-noise confound is removed twice over.** *Two* of the four reach their VisionΔ with
**no state noise at all** — `test_action_horizon` (60.6 %) and `test_token_len` (61.0 %), which
differ in `action_horizon` (20 vs 50) and step (15000 vs 24000) yet land 0.1 σ apart. So the
relation is not "state noise happens to act as a regulariser".

**The relationship is a threshold, not a gradient.** Pairwise differences, 500 episodes per arm,
unpaired (SE of a difference ≈ 3.1 pts):

| comparison | Δ success | significance |
|---|---|---|
| 0.037 → 0.185 | **+12.6 pts** | **4.0 σ** |
| 0.037 → 0.215 | +13.0 pts | 4.2 σ |
| 0.037 → 0.642 | +15.2 pts | 4.9 σ |
| 0.185 → 0.215 | +0.4 pts | 0.1 σ |
| 0.185 → 0.642 | +2.6 pts | 0.8 σ |
| 0.215 → 0.642 | +2.2 pts | 0.7 σ |

So crossing off the floor is solid (p < 1e-4), while **a 3.5× range of VisionΔ above the floor
buys only 2.6 success points and none of those three models is resolvable from another**. They
also trade places task-by-task (best on task 1 is `state_noise` at 81 %, worst on task 5 is also
`state_noise` at 26 %), which is what "no real difference" looks like.

**Consequence, and it reverses the earlier recommendation.** Of the total +15.2 pts available,
**the LR intervention alone delivers +12.6 pts (83 %)**, with no state noise at all. State noise
buys a 3.5× larger VisionΔ and only +2.6 further points (0.8 σ, not resolvable). So:

> VisionΔ is a useful *diagnostic* — a model at the floor (≈0.04) loses ~13 success points
> versus one that merely clears ≈0.19 — but pushing VisionΔ toward the state-noise ceiling does
> **not** buy further performance. **Lowering the peak LR is the efficient fix; state corruption
> is not**, despite moving VisionΔ 17×.

**Critical caveat for the write-up: VisionΔ is confounded with offline fit quality.** Across
these four rows, `err/mean` is *equally* monotone with success (1.032 → 48.0 %, 0.947 → 60.6 %,
0.947 → 61.0 %, 0.827 → 63.2 %), and across all eight checkpoints VisionΔ and `err/mean` improve
together. So the +13 points cannot be attributed to *seeing* rather than to *fitting the
demonstrations better* (or to stronger regularisation). The floor row is the extreme case:
VisionΔ 0.037 coincides with `err/mean` 1.03, i.e. a model that has not fitted at all.

**The step sweep came back null — the model is stationary by step 15000.**
`test_action_horizon` 15000 → 21000, same run / same LR schedule / same `action_horizon`, +40 %
steps:

| quantity | 15000 | 21000 | change |
|---|---|---|---|
| VisionΔ (reported) | 0.1806 | 0.1787 | −1.1 % |
| StateΔ | 0.3266 | 0.3280 | +0.4 % |
| LangΔ | 0.1158 | 0.1090 | −5.9 % |
| `err/mean` | 0.919 | 0.927 | **+0.9 %** |
| `effective_rank` @ `encoded` | 42.00 | 41.16 | −2.0 % |

Two consequences. First, **the "progressive abandonment of vision" prediction is not supported**
in this window (VisionΔ does not fall) — but the test had **no power**, because *nothing* moved,
so it neither confirms nor refutes the mechanism. Second, **the experiment failed to break the
`err/mean` confound below**, which needs the two quantities to move in *opposite* directions;
here they both stood still.

Note also that `err/mean` did not improve with 40 % more training (0.919 → 0.927). Across all
checkpoints `err/mean` sits in 0.827–1.03, so **this metric appears to bottom out near ≈0.83** in
this setup — worth stating in the write-up, since it means the above-floor models may all be at
an irreducible floor rather than differing in "how well fitted" they are.

**Correction, after `w020/6000` (49.0 %) and three p50 checkpoints were measured: none of the
candidate variables predicts success.** Updated table (success = 5 tasks × 100 episodes):

| checkpoint | lr | noise E[u²] | steps | err/mean | VisionΔ | success |
|---|---|---|---|---|---|---|
| `test_new_lr/15000` | 1.5e-4 | – | 15000 | 1.032 | 0.037 | 48.0 % |
| `test_new_lr/18000` | 1.5e-4 | – | 18000 | – | – | 51.0 % |
| **`w020/6000`** | 1.5e-4 | 0.067 | **6000** | **0.866** | **0.391** | **49.0 %** |
| `test_action_horizon/15000` | 5e-5 | – | 15000 | 0.947 | 0.185 | 60.6 % |
| `test_token_len/24000` | 5e-5 | – | 24000 | 0.947 | 0.215 | 61.0 % |
| `lora_new_lr/11999` | LoRA of token_len | – | 11999 | 0.856 | 0.204 | 62.2 % |
| `state_noise/18000` | 1.5e-4 | 0.167 | 18000 | 0.827 | 0.642 | 63.2 % |

`w020/6000` breaks **both** proposed explanations at once: it has the 2nd-highest VisionΔ yet the
2nd-lowest success, and its `err/mean` (0.866) is *better* than three models that beat it by
11–12 points. Rank correlations over all seven rows:

* Spearman(VisionΔ, success) = **+0.60** (n = 6) — was +1.00 before this point was added;
* Spearman(err/mean, success) = **−0.68** (n = 7) — the best of the three, still weak;
* Spearman(steps, success) = +0.29 (n = 7).

**So do not claim VisionΔ predicts success.** What survives are the two *step-matched* pairs,
where only one factor differs:

| matched pair | Δ success |
|---|---|
| 15000 steps: lr 1.5e-4 (48.0 %) → lr 5e-5 (60.6 %) | **+12.6 pts** |
| 18000 steps: no noise (51.0 %) → noise E=0.167 (63.2 %) | **+12.2 pts** |

The pattern in the table — every success ≥ 60 % has **both** ≥ 12000 steps **and** an
intervention, while `w020/6000` has the intervention but only 6000 steps — suggests
"enough training **and** an intervention" is the requirement, with VisionΔ merely tracking the
intervention. That is a hypothesis, not a result, and the cheapest test is to evaluate
**`state_noise/12000`** (intervention, 12000 steps, VisionΔ already 0.585): if it lands near
62 %, the step threshold sits between 6000 and 12000.

### The p50 run: its design purpose is refuted, but it yields a clean training trajectory

`pi05_robocasa_state_noise_p50` (Beta(1,2) with p = 0.5, i.e. half the samples keep a **clean**
state; effective noise energy 0.083, between w020's 0.067 and `state_noise`'s 0.167):

| checkpoint | err/mean | VisionΔ | StateΔ | LangΔ |
|---|---|---|---|---|
| p50/1000 | 0.878 | **0.175** | 0.115 | 0.085 |
| p50/2000 | 0.849 | **0.291** | 0.074 | 0.130 |
| p50/3000 | 0.854 | **0.305** | 0.058 | 0.136 |

* **The "calibration" hypothesis fails.** The idea was that never seeing a clean state (p = 1.0)
  permanently depresses the state trust-gain, so p = 0.5 should reach a *higher* StateΔ at the
  same VisionΔ. At matched VisionΔ, p50/3000 (0.305 → StateΔ 0.058) sits *below* w020/3000
  (0.317 → StateΔ 0.080). It does not leave the trade-off curve; it is slightly under it.
* **A third independent run showing VisionΔ grows during training under the incentive:**
  0.175 → 0.291 → 0.305 over 1000 → 2000 → 3000 steps, with StateΔ falling 0.115 → 0.074 →
  0.058. Most of the growth happens in the first 2000 steps. This corroborates the w020
  (+23 % over 3000→6000) and `state_noise` (+9.7 % over 12000→18000) trajectories.
* **`LangΔ` is *not* invariant early in training**: p50/1000 reads 0.085, below every converged
  checkpoint (which span 0.100–0.208), rising to 0.130/0.136 by 2000–3000 steps. So the
  invariance claim holds only among converged models.
* These three checkpoints have too few steps for their success rates to be informative, so
  evaluating them is not worth the simulator time.

### `max_token_len` is not a confounder (checked, no effect)

The existing ablations used mixed token budgets (`pi05_robocasa` / `..._state_noise` = 112,
`..._state_noise_w020` = 128) and mt = 112 truncates the `CoffeeSetupMug` prompt — and since the
state text follows the task description, the tail that gets cut is part of the **state** string.
Measured directly on p50/3000 at both settings: VisionΔ **+0.0 %**, LangΔ **+0.0 %**,
err/mean −0.0 %, StateΔ −1.9 %. So the mixed mt across the eight ablations is a non-issue.

**How to break the confound (unchanged): evaluate `w020/6000`.** The requirement is a pair with *matched fit
quality but different VisionΔ*. Exactly one such pair exists in the checkpoints at hand, and one
half of it is already measured:

| | `err/mean` | VisionΔ | success |
|---|---|---|---|
| `lora_new_lr/11999` | 0.856 | 0.204 | **62.2 %** (known) |
| `w020/6000` | 0.866 | **0.391** | **?** |

Fit differs by 1.2 %, VisionΔ by 1.92×. So one eval run settles it:

* success clearly above 62 % ⇒ at matched fit, more VisionΔ means more success ⇒ **VisionΔ is the
  operative variable and the confound is broken**;
* success ≈ 62 % ⇒ 1.9× VisionΔ at matched fit buys nothing ⇒ the threshold story holds and
  `err/mean` is the better predictor.

Caveat: `w020/6000` has only 6000 steps versus `lora_new_lr`'s 11999, so step is not matched —
but `err/mean` is, which is the quantity at issue. (A second, weaker pair exists —
`w020/3000` at err 0.851 / VisionΔ 0.317 versus `state_noise/12000` at err 0.841 / VisionΔ 0.585,
1.8× — but it needs *two* evals and has a worse step mismatch: 3000 vs 12000.)

**The floor effect rests on a single checkpoint, and that is the sharpest form of the problem.**
`test_new_lr/15000` is the *only* checkpoint with `err/mean > 1` (1.032) **and** the only one at
the VisionΔ floor (0.037); every other checkpoint lies in 0.827–0.947. So the 4 σ floor effect is
equally consistent with "**a model that has not fitted performs badly**" as with "a model that
does not see performs badly". State it as a limitation, or buy the `w020/6000` point above.

Limits to state in the write-up: n = 4, and the four rows must share the **same 5 tasks** (the
LoRA row was annotated "old tasks" — the others need confirming); the eval harness changed
during this project, so confirm a common protocol; and the floor row's VisionΔ coincides with
`err/mean ≈ 1.03`.

## 9. What this does NOT measure

`VisionΔ` is **behavioural reliance on the recorded frames**. It is not task success and not
object awareness. A model can have high VisionΔ and still mechanically replay a trajectory.
Do not report it as a success-rate proxy.

# reliance-eval：pristine clone 需要哪些文件

本文件回答一个问题：**只为了在新机器/新 clone 上跑通这个 skill，必须 push 哪些文件。**

结论：**7 个文件**（4 个 skill 文件——含本文件——+ 2 个脚本 + 1 个数据归档），其中归档被
`.gitignore` 挡住，必须 `git add -f`。另外建议一起 push 2 个文件。

---

## 1. 必须（缺任何一个都会失败）

| # | 文件 | git 状态 | 作用 | 缺了会怎样 |
|---|---|---|---|---|
| 1 | `.github/skills/reliance-eval/SKILL.md` | 新增（未跟踪） | skill 本体：流程、判读规则、踩坑 | skill 不存在 |
| 2 | `.github/skills/reliance-eval/references/metrics.md` | 新增 | 指标定义、GT 归一化公式、已测基线表 | 能跑但无法判读（没有基线可比） |
| 3 | `.github/skills/reliance-eval/scripts/ckpt_info.py` | 新增 | 从 checkpoint 反推配置 / `action_horizon` / state 宽度 | 条件只能靠猜 → 数字不可信 |
| 4 | `scripts/diagnose_vision.py` | **修改**（已跟踪） | 唯一的测量入口：消融 + 特征健康 | 没有测量工具 |
| 5 | `scripts/reliance_table.py` | 新增 | 汇总表（4 个 Δ + 4 个误差比 + `min(V,S)`） | 只能手工算 |
| 6 | `diagnostics/frames_robocasa_7tasks_84frames.npz` | 新增，**被 `.gitignore` 忽略** | 84 帧输入归档：保证各 checkpoint 看到**逐位相同**的输入 | 需要重跑 `--dump-frames`，而那需要 ~78 GiB 的 5 个 LeRobot 数据集 |

> `git add .github/skills/reliance-eval` 会把该目录下 4 个文件一并纳入（含本清单
> `references/files-to-push.md`），不需要单独处理。

### 归档必须显式强制添加

```
.gitignore:7:diagnostics/    diagnostics/frames_robocasa_7tasks_84frames.npz
```

因为 `diagnostics/` 整目录被忽略（里面的 `*/ablation.json`、`*.png` 是**生成物**，不该进仓库），
但归档是**输入**，必须进。两种做法二选一：

```
# 做法 A：只强制加这一个文件（推荐，改动面最小）
git add -f diagnostics/frames_robocasa_7tasks_84frames.npz

# 做法 B：在 .gitignore 里开一个例外（之后 dump 新归档会自动被跟踪）
printf '!diagnostics/frames_*.npz\n' >> .gitignore
```

归档指纹（push 后在另一台机器上核对）：

```
size  52.7 MiB (55,222,759 B)
sha256 bb9a853116ab6a576ce3c81ef8653a86c25adde146c83a2a869bfffaf5040883
内容  7 tasks x 2 episodes x 6 steps = 84 帧；state (84,23)、gt_chunk (84,20,21)、
      3 个 variant 的 3 路 256x256 原始帧、prompt、probe 标签
```

注意 git 仓库里放 52.7 MiB 二进制是可以接受的，但如果远端有体积限制，唯一替代是让对方
自己 `--dump-frames`（需要数据集）。**不要**改小 `--num-frames-per-episode` 重新 dump —— 那样
出来的帧和已有 `ablation.json` 不再可比，整个基线表作废。

---

## 2. 建议一起 push（非必须）

| 文件 | git 状态 | 为什么建议 | 缺了会怎样 |
|---|---|---|---|
| `scripts/screen_checkpoints.py` | 新增 | `SKILL.md` 把它列为"旧口径筛查器"（按策略自身 spread 归一，附带 `rank`/`within`/`CKA` 塔健康 + `PASS`/`balanced` 判定） | 只能用 `reliance_table.py`，失去塔健康与 PASS 判定 |
| `HANDOFF_vision_ablation.md` | **修改** | 项目的完整结论/背景记录，另一台机器就是靠它接手的 | 结论脉络丢失 |

---

## 3. 明确不需要（对本 skill 零依赖）

| 文件 | 状态 | 为什么不需 |
|---|---|---|
| `src/openpi/training/config.py` | 修改（6 行） | 改动全在 `pi05_robocasa_state_noise_p50` 那个 config 里（mt 112→128、`beta_b` 2.0→4.0、`peak_lr` 1.5e-4→5e-5、`num_train_steps` 6000→21000）。skill 全程用 `--config-name pi05_robocasa`，且它调用的 `data.create(..., load_norm_stats=False)` **在 HEAD 里已经存在**（见下面的审计） |
| `src/openpi/training/groot_openpi_dataset.py` | 修改（16 行） | `compute_overall_statistics` 的 q01/q99 由加权平均改为 `np.min`/`np.max`。只影响**新算**的 norm stats；skill 用的是 checkpoint 自带 `assets/norm_stats.json` |
| `src/openpi/training/checkpoints.py` | 修改（1 行） | `# max_to_keep=1` 取消注释成 `max_to_keep=3`，只是保留策略 |
| `examples/robocasa/main.py` | 修改（+347 行） | 属于 **robocasa-eval** 那个 skill（成功率评测）。要 push 就单独审，与依赖度无关 |
| `train.sh` / `server.sh` | 修改 | 本机路径 |
| `scripts/compute_per_task_norm_stats.py`、`scripts/trajectory_shortcut_check.py`、`scripts/lr_range_test.py`、`stats_cpu_mem.py`、`test_page_cache.py`、`package.json`、`package-lock.json`、`docs/pi05_robocasa_experiments_summary.md` | 未跟踪 | 临时分析脚本 / 无关产物 |
| `diagnostics/*/`（除归档） | 未跟踪、已忽略 | 生成物：`ablation.json`、`ablation.png`、`features.json`、`summary.json`。另一台机器重新跑自己的 |

---

## 4. 依赖审计（结论已核实，不是推测）

- `scripts/reliance_table.py`、`.github/skills/reliance-eval/scripts/ckpt_info.py`：只 import
  **标准库 + numpy + tyro**，不 import `openpi`。→ 任何环境下都能跑。
- `scripts/diagnose_vision.py` 对未提交的 src 改动**没有硬依赖**：
  - 它唯一的 `config` 调用是 `config.data.create(config.assets_dirs, config.model, load_norm_stats=False)`，
    而 `load_norm_stats` 参数 **HEAD 版本已有**（`git show HEAD:src/openpi/training/config.py:421`）。
  - 它不直接 import `groot_openpi_dataset`（只在注释里引用），只通过 `config.py` 的 fallback 路径间接触达，
    而 `--frames-from` 路径根本不会走到那里。
  - `state_noise_p` 的训练侧管线（`config.py` 7 处、`data_loader.py` 2 处、`transforms.py` 的
    `InterpolatedStateNoise.p`）**都已在 HEAD**，所以即便顺手把 `config.py` 的改动一起 push，
    也不需要额外带 `data_loader.py` / `transforms.py`。
- 因此：**只 push 第 1 节的 6 个文件，在一棵处于 HEAD 状态的树上就能跑通。**

---

## 5. 建议的 push 步骤

```bash
cd <repo-root>

# 1) 必须的 6 个（归档要 -f）
git add .github/skills/reliance-eval \
        scripts/diagnose_vision.py \
        scripts/reliance_table.py
git add -f diagnostics/frames_robocasa_7tasks_84frames.npz

# 2) 建议的 2 个
git add scripts/screen_checkpoints.py HANDOFF_vision_ablation.md

# 3) 看一眼列表：应恰好是上面 8 个（+ 你可选的其它文件）
git status --short

# 4) 提交并推送
git commit -m "add reliance-eval skill, its table tool and the 84-frame archive"
git push
```

## 6. push 后在另一台机器上做冒烟检查（不需要 GPU / 不需要数据集）

```bash
export HF_LEROBOT_HOME=/nonexistent        # config.py 在 import 时读它，必须存在（值不重要）

sha256sum diagnostics/frames_robocasa_7tasks_84frames.npz
#   期望 bb9a853116ab6a576ce3c81ef8653a86c25adde146c83a2a869bfffaf5040883

uv run .github/skills/reliance-eval/scripts/ckpt_info.py \
  --checkpoint checkpoints/<config>/<exp>/<step>          # 应打印 ah / 训练任务 / padding slots / 现成命令

uv run scripts/diagnose_vision.py --help | grep -E "zero-state-keys|state-dim-limit|frames-tasks|action-horizon"
```

`reliance_table.py` 在另一台机器上会报 `no ablation.json found` —— 这是**正常的**：`ablation.json`
是生成物、不进仓库，必须先跑 `diagnose_vision.py` 产出，然后再汇总。

---

## 7. 一条与本 skill 无关但会影响结果的提醒

`src/openpi/training/config.py` 里 `pi05_robocasa_state_noise_p50` 的**当前**取值
（mt=128、`beta_b=4.0`、`peak_lr=5e-5`、`num_train_steps=21000`）与已经训练出来的
`checkpoints/pi05_robocasa_state_noise_p50/p50/` 那些 checkpoint **使用的配方不一致**
（后者是 mt=112、`beta_b=2.0`、`peak_lr=1.5e-4`）。做对比时以 checkpoint 自己的
wandb 配置（`<exp>/wandb_id.txt` → `wandb/run-*/files/config.yaml`）为准，
`ckpt_info.py` 打印的就是它。

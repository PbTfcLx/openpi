# 依赖度指标：定义、口径与已测基线

## 0. ⚠️ 归一化口径修正（2026-09-16）：**下面所有 `nrm = z` 的行全部作废**

发现并修复了一个影响**全部历史行**的口径错误：

- 训练 (`training/data_loader.py:225,271`) 和推理 (`policies/policy_config.py:110,115`) 都用
  `Normalize(norm_stats, use_quantiles=data_config.use_quantile_norm)`，而 pi05 的 config
  `use_quantile_norm=True`（训练 run 自己 dump 的 config 也写着 `use_quantile_norm=True`）。
- 但 `scripts/diagnose_vision.py` 之前写的是 `_transforms.Normalize(norm_stats)`，即
  **默认的 z-score**。于是消融一直在用 `(x-mean)/(std+1e-6)` 编码 state、并用同一个 z-score
  把模型的输出还原成物理量，而 checkpoint 是在 **quantile** 空间 `(x-q01)/(q99-q01)*2-1` 上训练的。
- 两个映射不是常数倍关系：每维的换算因子 `std / (q01..q99 半宽)` 在 0.19–0.76 之间（arm 维
  0.19–0.52，gripper 维 0.76），所以历史 Δ **既整体偏小约 2x，又逐维加权不同**。
- 误差指标的后果更严重：z-score 空间里拿 quantile 空间的预测去比 z-score 空间的 GT，
  **一个完美模型也只能拿 0.708**（解析计算：`mean|q(x)-z(x)| / mean|z(x)|`，用归档 GT + 该
  checkpoint 的 stats）。这正好覆盖了历史表里 err/mn 的 0.827–1.032 —— 所以"所有模型都只比
  预测均值好一点点"是口径假象，不是模型结论。

修好之后（`--use-quantiles auto`，默认即跟随 config）：

| checkpoint | 帧 | nrm | Vision | State | Lang | Scene | imgEr | stEr | langEr | sceneEr | err/mn | min(V,S) | verdict |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| p50_w020/12000（7 任务） | 84 | **Q** | 0.415 | 0.150 | 0.162 | 0.341 | 1.41 | **1.08** | 1.13 | 1.33 | **0.457** | **0.150** | **PASS** |
| p50_w020/12000（5 训练任务） | 60 | **Q** | 0.441 | 0.119 | 0.074 | 0.370 | **2.07** | **1.11** | 1.06 | 1.82 | **0.287** | 0.119 | **PASS** |
| p50_w020/12000（同一 run，对照） | 84 | z | 0.296 | 0.022 | 0.067 | 0.168 | 1.21 | 1.00 | 1.04 | 1.09 | 0.889 | 0.022 | vision-only |
| test_new_lr/15000 | 84 | **Q** | 0.034 | **0.872** | 0.168 | 0.002 | 1.03 | **2.35** | 1.10 | 1.00 | **0.492** | 0.034 | proprio-only |
| test_new_lr/15000（对照） | 84 | z | 0.037 | 0.466 | 0.205 | 0.001 | 1.00 | 1.13 | 1.12 | 1.00 | 1.032 | 0.037 | proprio-only |

（`--use-quantiles no` 精确复现了历史行：Vision 0.296 / State 0.022 / err 0.888 vs 历史 0.889，
说明代码改动是两者之间**唯一**的差别。）

结论：

1. **`err/mn` 的可信区间整体失效重估**：真值 0.29–0.51，即这些模型其实拟合得不错（都在
   script 的 `<0.7 = trustworthy` 带内），"拟合太差所以结论不可信"的免责声明作废。
2. **`stEr` 跨过 1.05 阈值**：p50_w020/12000 的 state 是承重的（1.08/1.11），verdict 由
   `vision-only` 翻成 **PASS**（这是全项目第一个 PASS 行）；`test_new_lr` 的 `stEr` 从 1.13 涨到
   2.35。proprioception 的承重程度被系统性低估。
3. **Δ 的方向一致偏大**：Vision 0.296→0.415（84 帧）/ 0.338→0.441（5 任务），State 0.022→0.150
   （6.8x），Scene 翻倍，Lang 也变。
4. **trade-off 曲线的系数（`State = 0.332 - 0.605*Vision`）不能再引用**，它是用错空间的行拟合的。
   Q 口径下两个已测点都在老直线**上方**（V=0.415 → 老线预测 S 0.08，实测 0.150；V=0.034 → 预测
   0.31，实测 0.872）。

**待办**：表里其余 8 个 checkpoint 全部需要用 `--use-quantiles auto` 重测；在重测完成前，
任何 Q 行与 z 行的比较都是错的（`scripts/reliance_table.py` 现在会打印 `nrm` / `sta` 两列，并在
混用时告警）。

**同 run 步数扫描也跟着变了**（p50_w020，84 帧）：

| step | Vision | State | imgEr | stEr | err/mn | verdict |
|---|---|---|---|---|---|---|
| 6000 | 0.380 | 0.167 | 1.27 | 1.03 | 0.512 | balanced |
| 9000 | 0.409 | 0.158 | 1.32 | 1.02 | 0.498 | balanced |
| 12000 | 0.415 | 0.150 | 1.41 | **1.08** | 0.457 | **PASS** |

z 口径下这个扫描看起来是"Vision 平、State 跌 35%"；Q 口径下实际是 **Vision +9%、State 只跌 10%**，
而 `stEr` 到 12000 步才越过 1.05。即"本体感觉被逐步放弃"的幅度被错口径放大了 3 倍。

## 1. 消融挂在哪一步

`scripts/diagnose_vision.py` 重建 `create_trained_policy` 的输入变换链，并在 **`Normalize` 之后、
模型变换之前** 插入 `VariantTransform`：

```
repack -> InjectDefaultPrompt -> data_transforms (RobocasaInputs)
       -> Normalize -> VariantTransform(variant) -> model_transforms (ResizeImages/Tokenize/Pad)
```

这个位置只有一处：`state` 已是**归一化后的**向量（置零 = 回到训练均值），图像还是 uint8 HWC，
`image_mask` 可以逐相机切掉（比"把像素涂黑"更强，被 mask 的相机连 256 个 patch token 都不进
attention）。pi05 还会把 state 变成文本塞进 prompt，**这段文本在任何 variant 下都存在**，所以
`Lang` 的消融不会顺带改变 state 的可见性。

## 2. Δ（动了多少）

每个 variant 与 `baseline` 共用**同一份 flow-matching 噪声**（`noise_for(key, ...)`，按帧 key
fold_in 固定 seed），因此是严格配对比较。

```
delta_phys_per_dim[d] = mean_over_frames,steps | physical[variant][d] - physical[baseline][d] |
spread[d]             = std_over_frames,steps( physical[baseline][d] )        # 策略自身抖动
ratio[d]              = delta_phys_per_dim[d] / spread[d]                     # 仅对 informative 维
```

- `physical` 是**反归一化后**的物理量（`output_transform`），所以 Δ 与 GT 同量纲。
- **只统计 informative 维**：`base_motion`(7:11) 与 `control_mode`(11) 在数据里恒为 0，归一化
  std=0，归一化误差无定义 → 12 个环境维中只有 **7 维**（ee_pos 3 + ee_rot 3 + gripper_close 1）
  参与统计。`scored_action_dims` 记录在 JSON 里。
- `spread` 是**策略自己的**帧间抖动，`delta_phys_over_spread` 是 `screen_checkpoints.py` 用的
  口径（自洽但依赖于该策略，跨 checkpoint 不完全可比）。

### GT 归一化（本项目表格实际使用的口径）

```
gt_spread[d] = std_over_frames,steps( archive["gt_chunk"][d] )   # 数据本身的动作幅度
Delta        = mean_over_scored_dims( delta_phys_per_dim[d] / gt_spread[d] )
```

`gt_spread`（7 维，来自 `frames_robocasa_7tasks_84frames.npz`，**与 checkpoint 无关**）：
`[0.4261, 0.4413, 0.3319, 0.1268, 0.1230, 0.1563, 0.3898]`。
**换归档文件必须重算这 7 个数**，否则新行和旧行不在同一把尺子上。

读法：`Vision = 0.5` 意思是"拿掉图像后，预测的动作平均移动了数据动作幅度的 50%"。

## 3. Er（拟合是否还需要它）

```
err_norm_mean          = mean | prediction[variant] - gt_norm |        # 归一化空间
err_norm_over_baseline = err_norm_mean / err_norm_mean[baseline]
```

1.00 = 破坏这个输入后离线误差**没有变差** → 该输入对拟合数据不是必需的（捷径）；
> 1.05 = 该输入是承重的。注意误差只在**预测与 GT 共有的步数**上取（archive 是 20 步时，
  ah=50 的模型只用前 20 步），所以跨 horizon 的 Er 仍可比、Δ 需要同 horizon 对照。

参考量：`predict_mean_reference` = 永远输出训练均值的误差，`err/mn = err[baseline]/err[predict_mean]`
是该模型的拟合质量。判读带：`> 0.95` 拟合不足（结论不可信）、`0.7–0.95` 偏弱（结论有噪声）、
`< 0.7` 可信。已测区间 0.750–1.032。

## 4. 四个 variant 的语义

| variant | 破坏什么 | 列名 | 它回答的问题 |
|---|---|---|---|
| `mask_all_images` | 三路相机 token 全部移出 attention | `Vision` / `imgEr` | 有没有用视觉 |
| `zero_state` | state 置为归一化零点 | `State` / `stEr` | 有没有用本体感觉（当前关节/夹爪） |
| `blank_prompt` | 指令文本清空（state 文本保留） | `Lang` / `langEr` | 有没有用语言 |
| `swap_other_episode` | 换成**同任务另一个 episode**的像素（state/prompt 不变） | `Scene` / `sceneEr` | 看的是"有图"还是"画面内容"（物体感知代理） |

## 5. 已测基线（84 帧 / 60 帧，GT 归一化）

| checkpoint | ah | frames | Vision | State | Lang | Scene | imgEr | stEr | langEr | sceneEr | err/mn | min(V,S) | verdict |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| lora_new_lr/11999 | 50 | 84 | 0.204 | 0.297 | 0.133 | 0.108 | 1.12 | 1.09 | 1.06 | 1.04 | 0.856 | **0.204** | PASS |
| p50/1000 | 20 | 84 | 0.175 | 0.115 | 0.085 | 0.112 | 1.11 | 1.02 | 1.07 | 1.06 | 0.878 | 0.115 | balanced |
| **fix_checkpoint/15000（state 截断到 16 维，精确对齐）** | 50 | 84 | 0.440 | 0.098 | 0.094 | 0.201 | 1.33 | 0.98 | 1.04 | 1.10 | 0.881 | 0.098 | balanced |
| **fix_checkpoint/15000（16 维 + 5 训练任务）** | 50 | 60 | 0.421 | 0.088 | 0.065 | 0.218 | 1.55 | 1.05 | 1.04 | 1.21 | **0.738** | 0.088 | PASS |
| fix_checkpoint/15000（旧：置 0 但送 23 维，5任务） | 50 | 60 | 0.465 | 0.084 | 0.067 | 0.212 | 1.64 | 1.05 | 1.03 | 1.19 | 0.750 | 0.084 | PASS |
| fix_checkpoint/15000（旧：置 0 但送 23 维） | 50 | 84 | 0.505 | 0.090 | 0.096 | 0.194 | 1.45 | 0.99 | 1.04 | 1.10 | 0.878 | 0.090 | balanced |
| w020/3000 | 20 | 84 | 0.317 | 0.080 | 0.109 | 0.129 | 1.25 | 1.00 | 1.06 | 1.08 | 0.851 | 0.080 | balanced |
| p50/2000 | 20 | 84 | 0.291 | 0.074 | 0.130 | 0.126 | 1.20 | 1.02 | 1.09 | 1.09 | 0.849 | 0.074 | vision-only |
| w020/6000 | 20 | 84 | 0.391 | 0.072 | 0.115 | 0.157 | 1.30 | 1.00 | 1.07 | 1.08 | 0.866 | 0.072 | vision-only |
| p50/3000 | 20 | 84 | 0.305 | 0.058 | 0.136 | 0.136 | 1.20 | 1.01 | 1.08 | 1.08 | 0.854 | 0.058 | vision-only |
| p50/3000 (mt128) | 20 | 84 | 0.305 | 0.057 | 0.136 | 0.136 | 1.20 | 1.01 | 1.08 | 1.08 | 0.854 | 0.057 | vision-only |
| **p50_w020/6000** | 20 | 84 | 0.286 | 0.034 | 0.055 | 0.150 | 1.19 | 0.99 | 1.03 | 1.08 | 0.908 | 0.034 | vision-only |
| **p50_w020/9000** | 20 | 84 | 0.278 | 0.026 | 0.063 | 0.158 | 1.20 | 0.99 | 1.04 | 1.09 | 0.897 | 0.026 | vision-only |
| **p50_w020/12000** | 20 | 84 | 0.296 | 0.022 | 0.067 | 0.168 | 1.21 | 1.00 | 1.04 | 1.09 | 0.889 | 0.022 | vision-only |
| **p50_w020/12000（5 训练任务）** | 20 | 60 | 0.338 | 0.017 | 0.037 | 0.189 | 1.37 | 1.00 | 1.02 | 1.17 | 0.787 | 0.017 | vision-only |
| test_new_lr/15000（对照 ah50） | 50 | 84 | 0.040 | 0.388 | 0.185 | 0.001 | 1.01 | 1.12 | 1.12 | 1.00 | 1.000 | 0.040 | proprio-only |
| test_new_lr/15000 | 20 | 84 | 0.037 | 0.466 | 0.208 | 0.002 | 1.00 | 1.13 | 1.12 | 1.00 | 1.032 | 0.037 | proprio-only |
| test_new_lr/15000（对照 5任务） | 50 | 60 | 0.036 | 0.419 | 0.215 | 0.001 | 1.00 | 1.22 | 1.16 | 1.00 | 0.914 | 0.036 | proprio-only |
| state_noise/18000 | 20 | 84 | 0.642 | 0.021 | 0.128 | 0.215 | 1.57 | 1.00 | 1.08 | 1.16 | 0.827 | 0.021 | vision-only |
| state_noise/12000 | 20 | 84 | 0.585 | 0.020 | 0.110 | 0.189 | 1.49 | 1.00 | 1.05 | 1.13 | 0.841 | 0.020 | vision-only |

（`test_token_len/24000` 的 0.215 / 0.322 是早期在另一台机器上算的，本机只留了 `features.json`，
无法复现；本机可复现的最好 min 是 lora_new_lr 的 0.204。）

### p50_w020 这一个 run（`pi05_robocasa_state_noise_p50` / exp `test_p50_w020`，wandb `4fbkfxq8`）

条件：ah 20、`max_token_len` 128、beta (1.0, 4.0)、`state_noise_p` 0.5（E[u²] = 0.5×0.0667 = 0.033）、
peak_lr 5e-5、batch 64、5 任务；norm stats 的 padding 单位阵在 slot 23–31 ⇒ **state 是完整 23 维，
不需要 `--zero-state-keys`/`--state-dim-limit`**（与 `fix_checkpoint` 的 16 维不同）。
2026-09-16 评测时 run 还在训练（已存 6000/9000/12000，`num_train_steps` 21000）。

> **⚠️ 下面这一段是 z 口径的结论，已被 §0 推翻（保留作为"错口径会得出什么结论"的记录）。**
> 正确口径下同一个 checkpoint 是 Vision 0.441 / State 0.119 / stEr 1.11 / err 0.287 / **PASS**。

- **结论：极小 `min(V,S)` = 0.022（84 帧）/ 0.017（5 任务），是纯视觉端，proprioception 完全冗余**
  （`stEr` = 1.00：把 state 置零离线误差一点都不变差）。
- **它落在 trade-off 曲线下方**：Vision 0.296 代入曲线得 State ≈ 0.153，实测 0.022（0.14×）；
  5 任务行同向（0.338 → 预测 0.127，实测 0.017，0.13×）。即 p=0.5 这个配方**丢掉了 state 依赖，
  却没有换到曲线该给的视觉依赖**，不是沿曲线滑动。
- **同步数对比**（都与 12000 步、同 5 任务族）：`state_noise/12000`（beta 1,2 / p=1.0 / lr 1.5e-4）
  Vision 0.585 / State 0.020 / min 0.020；p50_w020/12000 Vision 0.296 / State 0.022 / min 0.022。
  min(V,S) 没有改善（0.020 vs 0.022），视觉依赖只有一半。**注意这不是单因子对照**：beta_b 2→4、
  p 1.0→0.5、peak_lr 1.5e-4→5e-5、mt 112→128 同时变了，只能当"另一条配方"读。
- **run 内步数扫描（84 帧，无 confound）**：6000→9000→12000，Vision 0.286/0.278/0.296（平），
  State 0.034/0.026/0.022（单调下降 -35%），`err/mn` 0.908/0.897/0.889（拟合变好）。
  即拟合继续变好的同时本体感觉继续被放弃，而视觉没有跟着涨（对比 `w020/6000` 的 State 0.072）。
  与 `p50/1000→3000`（Vision 0.175→0.305 上升）不矛盾：那是另一条 run（beta 1,2 / mt 112→128）。

**trade-off 曲线**（用上表 9 个非 fix_checkpoint 点线性拟合）——**⚠️ 见 §0：这条线是 z 口径的
产物，已作废，不要再用**：
$State = 0.3318 - 0.6047 \cdot Vision$（$R^2 = 0.601$，残差 ±0.06~0.16），Spearman ≈ −0.9。
用精确对齐后的行代入：84 帧那行 Vision=0.440 → 曲线预测 State≈0.066，实测 0.098（1.49×）；
5 任务那行 Vision=0.421 → 预测 0.077，实测 0.088（1.14×）。
**即 `fix_checkpoint` 基本落在已有 trade-off 曲线上**（略高 1.1–1.5×，超出量远小于既有散点的
±0.06~0.16 残差），它是沿曲线滑到视觉端，不是打破曲线。早期用"置 0 但送 23 维"那行
（Vision 0.505）算出的 3.4× 是口径不准造成的，已作废。

`fix_checkpoint` 精确对齐版本的 `err/mn = 0.738` 是全表**最好**的拟合（其余 0.750–1.032），
所以它的"视觉承重、state 冗余"不是拟合失败造成的假象。

其他已确认的事实：
- `langEr` 在全部 13 行里只有 **1.03–1.16**，且**最高的两行是 proprioception-only 模型**
  （它必须靠指令知道该干什么）。所以语言通道既不是瓶颈也不是捷径。
- **没有任何一行 `stEr` 明显 > 1.13**：在这套数据上，破坏 state 从没让误差显著变差过，
  唯一接近的是 test_new_lr（1.12–1.22，同时 `Vision` 0.04）。
- **horizon 影响的可比性**：同一 checkpoint（test_new_lr/15000）ah20 vs ah50 得
  `Vision` 0.037→0.040、`State` 0.466→0.388、`err/mn` 1.032→1.000。Δ 大约 ±10%，
  所以跨 horizon 比较要谨慎，最好带同 horizon 对照。
- **重现性下限 ≈ 0.1%**：同一条命令跑两次，`Vision` 得 0.4878 / 0.4884（bf16 kernel 调度差异）。
  低于该量级的差异不要解读。
- `p50_w020/*` 三个 84 帧行的原始输出在 `diagnostics/p50_w020_{6000,9000,12000}/`，5 任务行在
  `diagnostics/p50_w020_12000_5task/`；`batch_size` 自动降到 1（同机有训练任务占 88 GiB 显存）。

## 6. 为什么早期表格里只有 `Lang` 没有 `LangErr`

`ablation.json` 里**四个 variant 都有** `err_norm_mean` / `err_norm_over_baseline`（`blank_prompt`
在 test_new_lr/15000 上是 1.1221）。早期人工整理表格时只挑了 `imgEr` / `stEr`，因为
`screen_checkpoints.py` 的 PASS 判据只建立在 vision 与 proprioception 两条上；`scripts/reliance_table.py`
把四个误差比一起列出来了。这不是数据缺失，是当时的选择。

## 7. 已知未决项
- **全部历史行需要用 §0 的口径重测**（8 个 checkpoint，每个约 10 分钟）。在重测完成前，
  只有 §0 表里那 4 个新行可以互相比较。
- **部分 checkpoint 的 state 语义无法完全确认**。`fix_checkpoint` 的 `norm_stats.json` 显示
  slot 16–31 是 padding 单位阵（即模型只读 16 个 state 值，`joint_position` 不在其中），但它的
  slot 0–2 数值与同样布局的 `test_new_lr` 差很多，怀疑该 run 的 norm stats 由
  `scripts/compute_per_task_norm_stats.py` 用不同索引表/单任务数据算出。已按"截断到 16 维"
  做了精确对齐；这条只影响该 checkpoint 的绝对数值，不影响跨 checkpoint 的结论。
- **quantile 统计本身有两套写法**：`p50_w020/12000` 的 actions `q01/q99` 在 dim 0–2 是硬裁剪的
  ±1、gripper 维是 0/1；`test_new_lr` / `test_action_horizon` 则是真实分位数（半宽 0.29–0.91）。
  两者都是"checkpoint 自己的 stats"，各自内部自洽，但**不能跨 checkpoint 比较 Δ 的绝对大小**。
- `pi05_base` 从未做过消融（只当 CKA/PCA 参照），因为 RoboCasa 图像对它是 OOD。

## 8. proprioception 的四种探针（2026-09-16 新增）

`zero_state` 单独一个探针无法区分"state 没有信息"与"恰好这个常数无害"，所以加了三种：

| variant | 改什么 | 回答什么 |
|---|---|---|
| `zero_state` | 归一化空间置 0（quantile 下 = q01..q99 区间的**中点**，不是样本均值） | 老口径，保留以便和历史行对照 |
| `state_low` / `state_high` | 置为**原始** q01 / q99 的归一化像（quantile 下正好是 −1 / +1，即离散化的两端 bin） | 极端但**真实存在**的关节构型是否改变动作 |
| `swap_state` | 换成**同任务另一个 episode** 的 state（图像的 `swap_other_episode` 的本体感觉版） | state 的**内容**（而非它的存在）是否被读 |
| `state_other_norm` | 原始 state 不变，用**另一种归一化**重新编码（quantile ⇄ z-score） | 不破坏任何信息；有反应 = 模型真的在读 state |

判定逻辑写在 `ablation_verdict` 里：若 `zero_state` 很小但 `swap_state` / `state_other_norm`
不小 → 提示"模型对 prompt 里的数字敏感，而不是对它们的意思敏感"，会在 verdict 里给出专门的
说明文本。

**p50_w020/12000 实测（`own-spread` 单位，见 `ablation.json.verdict.state_probes`；括号内是
归一化空间 Δ）**：

| 口径 | zero | q01 | q99 | swap | other_norm |
|---|---|---|---|---|---|
| **Q，84 帧** | 0.150 | 0.229 | 0.262 | **0.233** | 0.143 |
| Q，60 帧（5 任务） | 0.119 | 0.189 | 0.225 | 0.210 | 0.112 |
| **Q，test_new_lr/15000** | 0.872 | 0.897 | **1.163** | 0.494 | 0.637 |
| z，84 帧（旧口径） | 0.022 | 0.032 | 0.031 | 0.022 | 0.046 |
| z，test_new_lr/15000（旧口径） | 0.466 | 0.530 | 0.638 | 0.322 | 0.337 |

（GT 归一化单位；括号内是同一次运行的归一化空间 Δ 见 `ablation.json`。）

- **Q 口径下四个探针一致**：都 ≥ `zero_state`，误差比都 > 1.05（`zero_state` 1.084 /
  `state_low` 1.063 / `state_high` 1.121 / `swap_state` **1.131** / `state_other_norm` 1.096）。
  ⇒ state 真的在被读、也真的承重。**`zero_state` 是最弱的探针**（q01/q99 比它强 1.5–1.8 倍，
  q99 在 proprioception-only 的 test_new_lr 上甚至给 1.163 vs `zero_state` 0.872），所以新版默认
  探针集建议带上 q01/q99/swap（`swap_state` 在"内容"意义上最接近图像的 `swap_other_episode`）。
- **z 口径下三个探针也一致**（0.008–0.014，误差比 0.99–1.00）⇒ 那个世界是**自洽**的，只是它是
  一个被错误归一化的模型；两个世界结论相反，原因在服务端的编码，不在探针设计。

## 9. 换 norm_stats 会怎样（`--norm-stats-from` / `--norm-stats-parts`）

```bash
# 用别的 stats 服务同一个 checkpoint，只换 state 块：
uv run scripts/diagnose_vision.py --config-name ... --checkpoint-dir ... \
  --norm-stats-from diagnostics/norm_stats_per_task/_merged/norm_stats.json \
  --norm-stats-parts state
```

p50_w020/12000，84 帧，quantile 口径：

| stats | Vision | State | Lang | Scene | imgEr | stEr | err/mn | verdict |
|---|---|---|---|---|---|---|---|---|
| 自己的（own） | 0.415 | 0.150 | 0.162 | 0.341 | 1.41 | 1.08 | 0.457 | **PASS** |
| `_merged` per-task，只换 state | 0.664 | 0.206 | 0.222 | 0.407 | 1.68 | **0.97** | 0.512 | balanced |
| `_merged`，state+actions 都换 | 0.664 | 0.205 | 0.222 | 0.407 | 1.68 | **0.97** | 0.512 | balanced |
| 自己的（用 `--norm-stats-from` 指向同一个文件） | 0.415 | 0.150 | 0.162 | 0.341 | 1.41 | 1.08 | 0.457 | PASS |

- **换 stats 影响很大**：`stEr` 从 1.08 掉回 0.97（跨回"不承重"），Vision 0.415→0.664。原因不是
  消融方法，而是模型对 state 编码高度敏感（见 §8 的 `state_other_norm`）。
- **同文件控制组误差 1.9e-4**（bf16 重现性量级），证明差异来自 stats 本身而不是新代码路径。
- **`state` 块与 `actions` 块可以分开换**，而且这一对文件里**只有 state 块真的起作用**：
  `_merged` 与 ckpt 自己的 stats 在 actions 上只差 dim 12–20，而那些维在 metric 里被排除；
  所以"只换 state"与"两块都换"给出到小数点后三位一致的结果（0.206 vs 0.205）。
- **实操规则**：评测一个 checkpoint 必须用**它自己的** `assets/norm_stats.json`；重算的 stats
  （索引表/裁剪方式都可能不同）会改变所有指标，包括 verdict。

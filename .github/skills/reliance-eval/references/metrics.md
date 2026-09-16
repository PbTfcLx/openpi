# 依赖度指标：定义、口径与已测基线

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
| test_new_lr/15000（对照 ah50） | 50 | 84 | 0.040 | 0.388 | 0.185 | 0.001 | 1.01 | 1.12 | 1.12 | 1.00 | 1.000 | 0.040 | proprio-only |
| test_new_lr/15000 | 20 | 84 | 0.037 | 0.466 | 0.208 | 0.002 | 1.00 | 1.13 | 1.12 | 1.00 | 1.032 | 0.037 | proprio-only |
| test_new_lr/15000（对照 5任务） | 50 | 60 | 0.036 | 0.419 | 0.215 | 0.001 | 1.00 | 1.22 | 1.16 | 1.00 | 0.914 | 0.036 | proprio-only |
| state_noise/18000 | 20 | 84 | 0.642 | 0.021 | 0.128 | 0.215 | 1.57 | 1.00 | 1.08 | 1.16 | 0.827 | 0.021 | vision-only |
| state_noise/12000 | 20 | 84 | 0.585 | 0.020 | 0.110 | 0.189 | 1.49 | 1.00 | 1.05 | 1.13 | 0.841 | 0.020 | vision-only |

（`test_token_len/24000` 的 0.215 / 0.322 是早期在另一台机器上算的，本机只留了 `features.json`，
无法复现；本机可复现的最好 min 是 lora_new_lr 的 0.204。）

**trade-off 曲线**（用上表 9 个非 fix_checkpoint 点线性拟合）：
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

## 6. 为什么早期表格里只有 `Lang` 没有 `LangErr`

`ablation.json` 里**四个 variant 都有** `err_norm_mean` / `err_norm_over_baseline`（`blank_prompt`
在 test_new_lr/15000 上是 1.1221）。早期人工整理表格时只挑了 `imgEr` / `stEr`，因为
`screen_checkpoints.py` 的 PASS 判据只建立在 vision 与 proprioception 两条上；`scripts/reliance_table.py`
把四个误差比一起列出来了。这不是数据缺失，是当时的选择。

## 7. 已知未决项

- **部分 checkpoint 的 state 语义无法完全确认**。`fix_checkpoint` 的 `norm_stats.json` 显示
  slot 16–31 是 padding 单位阵（即模型只读 16 个 state 值，`joint_position` 不在其中），但它的
  slot 0–2 数值与同样布局的 `test_new_lr` 差很多，怀疑该 run 的 norm stats 由
  `scripts/compute_per_task_norm_stats.py` 用不同索引表/单任务数据算出。已按"截断到 16 维"
  做了精确对齐；这条只影响该 checkpoint 的绝对数值，不影响跨 checkpoint 的结论。
- `pi05_base` 从未做过消融（只当 CKA/PCA 参照），因为 RoboCasa 图像对它是 OOD。

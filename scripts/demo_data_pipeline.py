"""openpi × LeRobot 数据流水线现场演示脚本（分享用）。

逐级打印数据在流水线上的形态：原始一帧 -> 拼接 -> 换视图 -> 归一化 -> 模型变换 -> 拼 batch。

用法（在本仓库根目录）:
    .venv/bin/python scripts/demo_data_pipeline.py
    .venv/bin/python scripts/demo_data_pipeline.py --config pi05_close_drawer_tianpeng_demo --batch 4

注意: 必须用真实脚本文件运行（不能从 stdin 跑），因为 config 里 num_workers>0 时会 spawn 子进程。
"""

from __future__ import annotations

import argparse
import collections
import dataclasses
import json
import pathlib

import numpy as np

import openpi.models.model as _model
import openpi.shared.normalize as _normalize
import openpi.transforms as _transforms
import openpi.training.checkpoints as _checkpoints
from openpi.training import config as _config
from openpi.training import data_loader as _data_loader
from openpi.training.groot_openpi_dataset import GrootOpenpiSingleDataset


def banner(title: str) -> None:
    print(f"\n{'=' * 78}\n{title}\n{'=' * 78}")


def describe(name: str, arr) -> None:
    a = np.asarray(arr)
    print(f"    {name:32s} shape={a.shape!s:22s} dtype={a.dtype}")


def apply_group(group: _transforms.Group, data: dict) -> dict:
    """按 openpi 的顺序依次施加一组变换（对应 transform_dataset 的实现）。"""
    out = data
    for t in group.inputs:
        out = t(out)
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="pi05_close_drawer_tianpeng_demo")
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--dump-frames", default=None, metavar="OUT.png", help="把第 0 帧三视角横向拼图存成图片（贴 PPT 用）")
    args = parser.parse_args()

    cfg = _config.get_config(args.config)
    ds_path = pathlib.Path(cfg.data.data_dirs[0]["path"])
    meta_dir = ds_path / "meta"

    # ---------------------------------------------------------------- 0. 数据集概览
    banner("0. 数据有什么 —— meta/ 四件套")
    info = json.loads((meta_dir / "info.json").read_text())
    print(f"  数据集目录: {ds_path}")
    print(f"  robot_type={info['robot_type']}  episodes={info['total_episodes']}  frames={info['total_frames']}  fps={info['fps']}")
    print(f"  chunks={info['total_chunks']} (每 chunk {info['chunks_size']} 集)  features={list(info['features'])[:6]} ...")
    print(f"  data_path  = {info['data_path']}")
    print(f"  video_path = {info['video_path']}")

    episodes = [json.loads(line) for line in (meta_dir / "episodes.jsonl").read_text().splitlines()]
    lens = np.array([e["length"] for e in episodes])
    print(f"\n  episodes.jsonl: {len(episodes)} 行；单集帧数 min/mean/max = {lens.min()}/{lens.mean():.1f}/{lens.max()}")
    counter = collections.Counter(tuple(e["tasks"]) for e in episodes)
    for tasks, n in counter.most_common():
        print(f"    tasks={list(tasks)} -> {n} 集")
    print(f"\n  tasks.jsonl:")
    for line in (meta_dir / "tasks.jsonl").read_text().splitlines():
        t = json.loads(line)
        print(f"    task_index={t['task_index']}: {t['task']!r}")

    # ---------------------------------------------------------------- 1~2. 取一帧原始样本
    banner("1~2. 数据长什么样 —— 数据集吐出的一个原始样本")
    dataset_meta = {"path": str(ds_path), "filter_key": cfg.data.data_dirs[0].get("filter_key")}
    ds = GrootOpenpiSingleDataset(dataset_meta=dataset_meta, action_horizon=cfg.model.action_horizon)
    raw = ds[0]
    print(f"  __len__ = {len(ds)}  （= 所有 episode 帧数之和）")
    for k, v in raw.items():
        if isinstance(v, str):
            print(f"    {k:32s} = {v!r}")
        else:
            describe(k, v)
    print(f"\n    state 前 6 维 = {np.round(np.asarray(raw['observation/state'])[:6], 4)}")
    print(f"    action[0] 前 6 维 = {np.round(np.asarray(raw['actions'])[0, :6], 4)}")
    print("    注: state 23 维 / action 21 维是「拼接产物」，原生是 53 维 state + 12 维 action")
    if args.dump_frames:
        import cv2

        grid = np.concatenate(
            [
                np.asarray(raw["observation/image"]),
                np.asarray(raw["observation/right_image"]),
                np.asarray(raw["observation/wrist_image"]),
            ],
            axis=1,
        )
        # cv2 按 BGR 存图：三路相机本身是 RGB，这里转一下保证颜色正确
        cv2.imwrite(args.dump_frames, cv2.cvtColor(grid, cv2.COLOR_RGB2BGR))
        print(f"    [已保存三视角拼图] {args.dump_frames}  (left | right | wrist)")
    # ---------------------------------------------------------------- 3. RobocasaInputs
    banner("3. RobocasaInputs —— 排成模型认识的 4 视图接口")
    data_config = cfg.data.create(cfg.assets_dirs, cfg.model)
    item = apply_group(data_config.data_transforms, raw)
    for k, v in item["image"].items():
        describe(f"image/{k}", v)
    for k, v in item["image_mask"].items():
        print(f"    image_mask/{k:22s} = {bool(np.asarray(v).item())}")
    describe("state", item["state"])
    describe("actions", item["actions"])
    print(f"    prompt = {item['prompt']!r}")
    print("    注: right_wrist 没有真实相机 -> 补零 + mask=False（pi05 下模型会忽略）")

    # ---------------------------------------------------------------- 4. 归一化
    banner("4. Normalize —— z-score（用 32 维 norm stats 裁到实际维度）")
    norm_stats = _checkpoints.load_norm_stats(cfg.norm_stats_dir, None)
    for k, st in norm_stats.items():
        describe(f"norm_stats/{k}.mean", st.mean)
    before, after = np.asarray(item["state"]), None
    item = _transforms.Normalize(norm_stats)(item)
    after = np.asarray(item["state"])
    print(f"    state 归一化前 min/max = {before.min():+.3f}/{before.max():+.3f}")
    print(f"    state 归一化后 min/max = {after.min():+.3f}/{after.max():+.3f}  （目标: 大致落在 [-1,1]）")
    print("    注: actions 用同一套机制归一化；推理时 Unnormalize 把它还原回物理量纲")

    # ---------------------------------------------------------------- 5. 模型变换
    banner("5. model_transforms —— resize / 分词 / 补齐")
    # 重新走一遍完整链路（data_transforms + Normalize + model_transforms），与 transform_dataset 完全一致
    item = apply_group(data_config.data_transforms, raw)
    item = _transforms.Normalize(norm_stats)(item)
    item = apply_group(data_config.model_transforms, item)
    for k, v in item["image"].items():
        describe(f"image/{k}", v)
    describe("state (pad 到 32)", item["state"])
    describe("actions (50 步, pad 到 32)", item["actions"])
    describe("tokenized_prompt", item["tokenized_prompt"])
    n_valid = int(np.count_nonzero(item["tokenized_prompt_mask"]))
    print(f"    有效 token 数 = {n_valid} / {cfg.model.max_token_len}（其余为 padding）")
    ids = np.asarray(item["tokenized_prompt"])[: n_valid].tolist()
    print(f"    token 序列前 24 个 id = {ids[:24]}")
    print("    pi05 把 state 离散成 256 个整数桶、拼进文本: 'Task: <指令>, State: <整数串>;\\nAction: '")

    # ---------------------------------------------------------------- 6. batch
    banner(f"6. collate —— 真实训练 batch（B={args.batch}, 走 create_data_loader）")
    run_cfg = dataclasses.replace(cfg, batch_size=args.batch, num_workers=0)
    loader = _data_loader.create_data_loader(
        run_cfg, shuffle=False, num_batches=1, framework="pytorch", norm_stats=norm_stats
    )
    obs, actions, source = next(iter(loader))
    for k, v in obs.images.items():
        mask = bool(np.asarray(obs.image_masks[k]).ravel()[0])
        print(f"    obs.images[{k!r}]{'':4s} {tuple(v.shape)} {v.dtype} mask={mask}")
    print(f"    obs.state                 {tuple(obs.state.shape)} {obs.state.dtype}")
    print(f"    obs.tokenized_prompt      {tuple(obs.tokenized_prompt.shape)} {obs.tokenized_prompt.dtype}")
    print(f"    obs.tokenized_prompt_mask {tuple(obs.tokenized_prompt_mask.shape)} {obs.tokenized_prompt_mask.dtype}")
    print(f"    actions                   {tuple(actions.shape)} {actions.dtype}")
    print(f"    source                    {tuple(np.asarray(source).shape)} {np.asarray(source)}")
    if obs.images["base_0_rgb"].numel():
        img = obs.images["base_0_rgb"]
        print(f"    图像值域检查: min={img.min():.2f} max={img.max():.2f}（uint8 -> [-1,1] float32）")
    print("\n  一句话: batch 里每一项都对应模型的一个输入/监督信号，形状即契约。")


if __name__ == "__main__":
    main()

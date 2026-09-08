import dataclasses
import functools
import gc
import logging
import os
import platform
import subprocess
from typing import Any

import etils.epath as epath
import flax.nnx as nnx
from flax.training import common_utils
import flax.traverse_util as traverse_util
import jax
import jax.experimental
import jax.numpy as jnp
import numpy as np
import optax
import tqdm_loggable.auto as tqdm
import tyro
import wandb

import openpi.models.model as _model
import openpi.shared.array_typing as at
import openpi.shared.nnx_utils as nnx_utils
import openpi.training.checkpoints as _checkpoints
import openpi.training.config as _config
import openpi.training.data_loader as _data_loader
import openpi.training.optimizer as _optimizer
import openpi.training.sharding as sharding
import openpi.training.utils as training_utils
import openpi.training.weight_loaders as _weight_loaders


def init_logging():
    """Custom logging format for better readability."""
    level_mapping = {"DEBUG": "D", "INFO": "I", "WARNING": "W", "ERROR": "E", "CRITICAL": "C"}

    class CustomFormatter(logging.Formatter):
        def format(self, record):
            record.levelname = level_mapping.get(record.levelname, record.levelname)
            return super().format(record)

    formatter = CustomFormatter(
        fmt="%(asctime)s.%(msecs)03d [%(levelname)s] %(message)-80s (%(process)d:%(filename)s:%(lineno)s)",
        datefmt="%H:%M:%S",
    )

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    logger.handlers[0].setFormatter(formatter)


def init_wandb(config: _config.TrainConfig, *, resuming: bool, log_code: bool = False, enabled: bool = True):
    if not enabled:
        wandb.init(mode="disabled")
        return

    ckpt_dir = config.checkpoint_dir
    if not ckpt_dir.exists():
        raise FileNotFoundError(f"Checkpoint directory {ckpt_dir} does not exist.")
    if resuming:
        run_id = (ckpt_dir / "wandb_id.txt").read_text().strip()
        wandb.init(id=run_id, resume="must", project=config.project_name)
    else:
        wandb.init(
            name=config.exp_name,
            config=dataclasses.asdict(config),
            project=config.project_name,
        )
        (ckpt_dir / "wandb_id.txt").write_text(wandb.run.id)

    if log_code:
        wandb.run.log_code(epath.Path(__file__).parent.parent)


def _load_weights_and_validate(loader: _weight_loaders.WeightLoader, params_shape: at.Params) -> at.Params:
    """Loads and validates the weights. Returns a loaded subset of the weights."""
    loaded_params = loader.load(params_shape)
    at.check_pytree_equality(expected=params_shape, got=loaded_params, check_shapes=True, check_dtypes=True)

    # Remove jax.ShapeDtypeStruct from the loaded params. This makes sure that only the loaded params are returned.
    return traverse_util.unflatten_dict(
        {k: v for k, v in traverse_util.flatten_dict(loaded_params).items() if not isinstance(v, jax.ShapeDtypeStruct)}
    )


@at.typecheck
def init_train_state(
    config: _config.TrainConfig, init_rng: at.KeyArrayLike, mesh: jax.sharding.Mesh, *, resume: bool
) -> tuple[training_utils.TrainState, Any]:
    tx = _optimizer.create_optimizer(config.optimizer, config.lr_schedule, weight_decay_mask=None)

    def init(rng: at.KeyArrayLike, partial_params: at.Params | None = None) -> training_utils.TrainState:
        rng, model_rng = jax.random.split(rng)
        # initialize the model (and its parameters).
        model = config.model.create(model_rng)

        # Merge the partial params into the model.
        if partial_params is not None:
            graphdef, state = nnx.split(model)
            # This will produce an error if the partial params are not a subset of the state.
            state.replace_by_pure_dict(partial_params)
            model = nnx.merge(graphdef, state)

        params = nnx.state(model)
        # Convert frozen params to bfloat16.
        params = nnx_utils.state_map(params, config.freeze_filter, lambda p: p.replace(p.value.astype(jnp.bfloat16)))

        return training_utils.TrainState(
            step=0,
            params=params,
            model_def=nnx.graphdef(model),
            tx=tx,
            opt_state=tx.init(params.filter(config.trainable_filter)),
            ema_decay=config.ema_decay,
            ema_params=None if config.ema_decay is None else params,
        )

    train_state_shape = jax.eval_shape(init, init_rng)
    state_sharding = sharding.fsdp_sharding(train_state_shape, mesh, log=True)

    if resume:
        return train_state_shape, state_sharding

    partial_params = _load_weights_and_validate(config.weight_loader, train_state_shape.params.to_pure_dict())
    replicated_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())

    # Initialize the train state and mix in the partial params.
    train_state = jax.jit(
        init,
        donate_argnums=(1,),  # donate the partial params buffer.
        in_shardings=replicated_sharding,
        out_shardings=state_sharding,
    )(init_rng, partial_params)

    return train_state, state_sharding


@at.typecheck
def train_step(
    config: _config.TrainConfig,
    rng: at.KeyArrayLike,
    state: training_utils.TrainState,
    batch: tuple[_model.Observation, _model.Actions],
) -> tuple[training_utils.TrainState, dict[str, at.Array]]:
    model = nnx.merge(state.model_def, state.params)
    model.train()

    @at.typecheck
    def loss_fn(
        model: _model.BaseModel, rng: at.KeyArrayLike, observation: _model.Observation, actions: _model.Actions
    ):
        chunked_loss = model.compute_loss(rng, observation, actions, train=True)
        return jnp.mean(chunked_loss)

    train_rng = jax.random.fold_in(rng, state.step)
    observation, actions = batch

    # Filter out frozen params.
    diff_state = nnx.DiffState(0, config.trainable_filter)
    loss, grads = nnx.value_and_grad(loss_fn, argnums=diff_state)(model, train_rng, observation, actions)

    params = state.params.filter(config.trainable_filter)
    updates, new_opt_state = state.tx.update(grads, state.opt_state, params)
    new_params = optax.apply_updates(params, updates)

    # Update the model in place and return the new full state.
    nnx.update(model, new_params)
    new_params = nnx.state(model)

    new_state = dataclasses.replace(state, step=state.step + 1, params=new_params, opt_state=new_opt_state)
    if state.ema_decay is not None:
        new_state = dataclasses.replace(
            new_state,
            ema_params=jax.tree.map(
                lambda old, new: state.ema_decay * old + (1 - state.ema_decay) * new, state.ema_params, new_params
            ),
        )

    # Filter out params that aren't kernels.
    kernel_params = nnx.state(
        model,
        nnx.All(
            nnx.Param,
            nnx.Not(nnx_utils.PathRegex(".*/(bias|scale|pos_embedding|input_embedding)")),
            lambda _, x: x.value.ndim > 1,
        ),
    )
    info = {
        "loss": loss,
        "grad_norm": optax.global_norm(grads),
        "param_norm": optax.global_norm(kernel_params),
    }
    return new_state, info


def compute_per_source_losses(
    config: _config.TrainConfig,
    replicated_sharding: jax.sharding.NamedSharding,
    state: training_utils.TrainState,
    observation: _model.Observation,
    actions: _model.Actions,
    source_ids: jax.Array,
) -> dict[str, jax.Array]:
    """Mean flow-matching loss per data source, on the exact batch the model just saw.

    Uses the EMA weights (online as fallback) in eval mode (no augmentation) so it is a
    stable, low-noise per-source training-loss probe. ``source_ids`` is the per-sample
    dataset index attached by the Groot datasets; results are returned in data-config order
    (index ``i`` = ``data_dirs[i]``), NaN where a source is absent from this batch.

    Correct under FSDP / data parallelism: the batch arrives data-sharded across devices, so
    we first gather a full copy onto every device (``with_sharding_constraint`` to the
    replicated sharding, an all-gather) and then compute the per-source sums locally. Every
    device therefore produces the same *global* values and no extra cross-device reduction
    is needed. On a single device the gather is a no-op.
    """
    # Gather the whole batch (and its per-sample source ids) onto every device.
    observation = jax.lax.with_sharding_constraint(
        observation, jax.tree.map(lambda _: replicated_sharding, observation)
    )
    actions = jax.lax.with_sharding_constraint(actions, replicated_sharding)
    source_ids = jax.lax.with_sharding_constraint(source_ids, replicated_sharding)

    params = state.ema_params if state.ema_params is not None else state.params
    model = nnx.merge(state.model_def, params)
    model.eval()
    rng = jax.random.key(0)  # deterministic probe (no augmentation since train=False)
    # compute_loss returns a per-(sample, action-timestep) loss; reduce over the horizon.
    chunked = model.compute_loss(rng, observation, actions, train=False)  # (b, ah)
    per_sample = jnp.mean(chunked, axis=-1)  # (b,)
    src = jnp.asarray(source_ids, dtype=jnp.int32)
    data_dirs = getattr(config.data, "data_dirs", None)
    num_sources = len(data_dirs) if data_dirs else 1
    sums = jnp.zeros((num_sources,), dtype=per_sample.dtype).at[src].add(per_sample)
    counts = jnp.zeros((num_sources,), dtype=jnp.int32).at[src].add(1)
    return {
        "per_source_loss": jnp.where(counts > 0, sums / jnp.maximum(counts, 1), jnp.nan),
        "per_source_count": counts,
    }


def _host_mem_gb() -> tuple[float, float]:
    """Return (current RSS GB, peak RSS GB) of this process from /proc/self/status.

    Diagnostic only: used to log how much host memory a checkpoint save actually
    allocates and whether it is released afterwards.
    """
    rss = peak = 0.0
    try:
        with open("/proc/self/status") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    rss = int(line.split()[1]) / 1048576
                elif line.startswith("VmHWM:"):
                    peak = int(line.split()[1]) / 1048576
    except OSError:
        pass
    return rss, peak


def _total_worker_rss_gb() -> float:
    """Total RSS (GB) of this process's children (torch DataLoader workers)."""
    try:
        out = subprocess.run(
            ["ps", "-o", "rss=", "--ppid", str(os.getpid())],
            capture_output=True,
            text=True,
            check=False,
        ).stdout
        total = sum(int(x) for x in out.split() if x.strip().isdigit())
        return total / 1048576
    except Exception:  # noqa: BLE001 - diagnostic only
        return float("nan")


def _cgroup_mem_gb() -> tuple[float, float, float]:
    """Return (anon GB, file GB, active_file GB) from this cgroup's memory.stat.

    Diagnostic only: cgroup file cache (what container dashboards / memory.current
    report as "used") includes reclaimable page cache -- this is what grows toward
    the container limit during training.
    """
    try:
        with open("/proc/self/cgroup") as f:
            cg = f.read().strip().splitlines()[-1].split(":", 2)[-1]
        with open(f"/sys/fs/cgroup{cg}/memory.stat") as f:
            stats = {}
            for line in f:
                parts = line.split()
                if len(parts) == 2:
                    try:
                        stats[parts[0]] = int(parts[1])
                    except ValueError:
                        pass
        # memory.stat values are in BYTES -> divide by 1024**3 to get GB.
        anon = stats.get("anon", 0) / 1073741824
        file_ = stats.get("file", 0) / 1073741824
        active_file = stats.get("active_file", 0) / 1073741824
        return anon, file_, active_file
    except (OSError, ValueError):
        return float("nan"), float("nan"), float("nan")


def main(config: _config.TrainConfig):
    init_logging()
    logging.info(f"Running on: {platform.node()}")

    if config.batch_size % jax.device_count() != 0:
        raise ValueError(
            f"Batch size {config.batch_size} must be divisible by the number of devices {jax.device_count()}."
        )

    jax.config.update("jax_compilation_cache_dir", str(epath.Path("~/.cache/jax").expanduser()))

    rng = jax.random.key(config.seed)
    train_rng, init_rng = jax.random.split(rng)

    mesh = sharding.make_mesh(config.fsdp_devices)
    data_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec(sharding.DATA_AXIS))
    replicated_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())

    checkpoint_manager, resuming = _checkpoints.initialize_checkpoint_dir(
        config.checkpoint_dir,
        keep_period=config.keep_period,
        overwrite=config.overwrite,
        resume=config.resume,
    )
    init_wandb(config, resuming=resuming, enabled=config.wandb_enabled)

    # On resume, prefer the norm stats persisted with the checkpoint so training continues
    # with exactly the same normalization the checkpoint was trained with (mirrors the
    # inference path in `policy_config.create_trained_policy`). If the checkpoint has no
    # persisted norm stats, falls back to the config-derived ones (e.g. from data dirs).
    checkpoint_norm_stats = None
    if resuming:
        asset_id = config.data.assets.asset_id or (
            None if config.data.repo_id is tyro.MISSING else config.data.repo_id
        )
        checkpoint_norm_stats = _checkpoints.load_norm_stats_from_checkpoint(
            config.checkpoint_dir, asset_id
        )
    elif config.norm_stats_dir is not None:
        # Fresh fine-tune reusing a previously-trained model's normalization: pull the
        # norm stats persisted with that checkpoint (new data + new exp_name) instead of
        # recomputing them from the (possibly new) data config. Groot-style configs have
        # no asset_id, so the stats are stored at <step>/assets/norm_stats.json directly.
        logging.info(
            f"Reusing norm stats from checkpoint dir: {config.norm_stats_dir}"
        )
        checkpoint_norm_stats = _checkpoints.load_norm_stats(
            config.norm_stats_dir, None
        )

    data_loader = _data_loader.create_data_loader(
        config,
        sharding=data_sharding,
        shuffle=True,
        norm_stats=checkpoint_norm_stats,
    )
    data_iter = iter(data_loader)
    # The loader now yields (observation, actions, source_ids); source_ids tags each sample
    # with its dataset index (Groot datasets) or is None otherwise. It is used to log
    # per-source training losses.
    observation, actions, source = next(data_iter)
    batch = (observation, actions)
    logging.info(f"Initialized data loader:\n{training_utils.array_tree_to_info(batch)}")

    # Log images from first batch to sanity check.
    images_to_log = [
        wandb.Image(np.concatenate([np.array(img[i]) for img in observation.images.values()], axis=1))
        for i in range(min(5, len(next(iter(observation.images.values())))))
    ]
    wandb.log({"camera_views": images_to_log}, step=0)

    train_state, train_state_sharding = init_train_state(config, init_rng, mesh, resume=resuming)
    jax.block_until_ready(train_state)
    logging.info(f"Initialized train state:\n{training_utils.array_tree_to_info(train_state.params)}")

    if resuming:
        resume_step = _checkpoints.latest_resumable_step(config.checkpoint_dir)
        if resume_step is None:
            raise RuntimeError(
                "No resumable checkpoint found (no `train_state` saved). Resume requires a "
                "checkpoint that saved the full training state, i.e. a step where "
                "step % save_train_state_interval == 0."
            )
        train_state = _checkpoints.restore_state(checkpoint_manager, train_state, data_loader, step=resume_step)

    ptrain_step = jax.jit(
        functools.partial(train_step, config),
        in_shardings=(replicated_sharding, train_state_sharding, data_sharding),
        out_shardings=(train_state_sharding, replicated_sharding),
        donate_argnums=(1,),
    )

    # Create the lr schedule once on the host; used to log the exact lr at each log step.
    lr_schedule = config.lr_schedule.create()

    # Per-source train-loss probe: names of the data sources (in loader order) plus a jitted
    # no-grad loss pass evaluated at log time on the batch the model just trained on. Only
    # active when the loader tags each sample with its dataset index ("source").
    source_names = []
    for ds in (getattr(config.data, "data_dirs", None) or []):
        if isinstance(ds, dict):
            source_names.append(str(ds.get("task") or os.path.basename(str(ds.get("path", "")))))
        else:
            source_names.append(str(ds))
    per_source_step = jax.jit(
        functools.partial(compute_per_source_losses, config, replicated_sharding),
        in_shardings=(train_state_sharding, data_sharding, data_sharding, data_sharding),
        out_shardings={
            "per_source_loss": replicated_sharding,
            "per_source_count": replicated_sharding,
        },
    )

    start_step = int(train_state.step)
    pbar = tqdm.tqdm(
        range(start_step, config.num_train_steps),
        initial=start_step,
        total=config.num_train_steps,
        dynamic_ncols=True,
    )

    infos = []
    for step in pbar:
        with sharding.set_mesh(mesh):
            train_state, info = ptrain_step(train_rng, train_state, batch)
        infos.append(info)
        if step % config.log_interval == 0:
            stacked_infos = common_utils.stack_forest(infos)
            reduced_info = jax.device_get(jax.tree.map(jnp.mean, stacked_infos))
            # Per-source train-loss probe on the batch the model just trained on (EMA weights,
            # no-grad, no augmentation). Sources absent from this batch are skipped.
            if source is not None and source_names:
                with sharding.set_mesh(mesh):
                    probe = per_source_step(train_state, observation, actions, source)
                per_src = jax.device_get(probe["per_source_loss"])
                counts = jax.device_get(probe["per_source_count"])
                for i, name in enumerate(source_names):
                    if i < counts.shape[0] and int(counts[i]) > 0:
                        reduced_info[f"loss/{name}"] = float(per_src[i])
            info_str = ", ".join(f"{k}={v:.4f}" for k, v in reduced_info.items())
            pbar.write(f"Step {step}: {info_str}")
            # Log the exact lr at this log step (host-side), not the interval mean.
            reduced_info["lr"] = float(lr_schedule(step))
            wandb.log(reduced_info, step=step)
            infos = []
        # Periodic container-memory snapshot: attribute the gradual climb to the main
        # process, the DataLoader workers, or the cgroup file cache respectively. cgroup
        # anon = real process memory, file = reclaimable page cache (what container
        # dashboards / memory.current count as "used").
        # if step % 500 == 0:
        #     rss, peak = _host_mem_gb()
        #     workers = _total_worker_rss_gb()
        #     cg_anon, cg_file, cg_active = _cgroup_mem_gb()
        #     logging.info(
        #         "Step %d mem: main rss=%.1fGB (peak %.1fGB), workers=%.1fGB, "
        #         "cgroup anon=%.1fGB file=%.1fGB active_file=%.1fGB",
        #         step,
        #         rss,
        #         peak,
        #         workers,
        #         cg_anon,
        #         cg_file,
        #         cg_active,
        #     )
        observation, actions, source = next(data_iter)
        batch = (observation, actions)

        if (step % config.save_interval == 0 and step > start_step) or step == config.num_train_steps - 1:
            # Weights + norm stats are saved every `save_interval`; the full training state
            # (needed for resume) is additionally saved every `save_train_state_interval`,
            # and always on the final step so training can always be resumed from the end.
            save_train_state_interval = config.save_train_state_interval or config.save_interval
            save_train_state = (step % save_train_state_interval == 0) or step == config.num_train_steps - 1
            # Diagnostic: log host RSS / worker RSS around the save so we can tell whether
            # the checkpoint save permanently retains host memory or just spikes transiently.
            before_rss, before_peak = _host_mem_gb()
            before_workers = _total_worker_rss_gb()
            _checkpoints.save_state(
                checkpoint_manager,
                train_state,
                data_loader,
                step,
                save_train_state=save_train_state,
            )
            # Orbax saves asynchronously: the background commit thread (and the host
            # buffers it holds from copying the train state) stays referenced until it
            # is torn down. Waiting here releases each save's buffers promptly (a few
            # seconds on local disk) instead of leaving them for the next save to clean up.
            # checkpoint_manager.wait_until_finished()
            # gc.collect()
            # after_rss, after_peak = _host_mem_gb()
            # after_workers = _total_worker_rss_gb()
            # logging.info(
            #     "Checkpoint save step=%d host mem: main rss %.1f->%.1f GB (peak %.1f), "
            #     "workers %.1f->%.1f GB, total rss %.1f->%.1f GB",
            #     step,
            #     before_rss,
            #     after_rss,
            #     after_peak,
            #     before_workers,
            #     after_workers,
            #     before_rss + before_workers,
            #     after_rss + after_workers,
            # )

    logging.info("Waiting for checkpoint manager to finish")
    checkpoint_manager.wait_until_finished()


if __name__ == "__main__":
    main(_config.cli())

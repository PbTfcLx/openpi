from __future__ import annotations

import asyncio
import concurrent.futures as futures
import dataclasses
import logging
from typing import Protocol

from etils import epath
import jax
import orbax.checkpoint as ocp
import orbax.checkpoint.future as future

from openpi.shared import array_typing as at
import openpi.shared.normalize as _normalize
import openpi.training.data_loader as _data_loader
import openpi.training.utils as training_utils

# File (in the checkpoint directory) that records checkpoint steps which should never be deleted.
# This is persisted so that checkpoints we resume from are protected across multiple resume sessions,
# not just the current one.
_PROTECTED_STEPS_FILE = "protected_steps.txt"


def _load_protected_steps(checkpoint_dir: epath.Path) -> set[int]:
    """Load the set of checkpoint steps that should never be deleted."""
    path = checkpoint_dir / _PROTECTED_STEPS_FILE
    if not path.exists():
        return set()
    return {int(line) for line in path.read_text().splitlines() if line.strip().isdigit()}


def _save_protected_steps(checkpoint_dir: epath.Path, steps: set[int]) -> None:
    """Persist the set of checkpoint steps that should never be deleted."""
    path = checkpoint_dir / _PROTECTED_STEPS_FILE
    path.write_text("\n".join(str(s) for s in sorted(steps)) + "\n")


# Bound how many GB of checkpoint data Orbax copies/writes concurrently (param name in the
# installed orbax is `save_concurrent_gb`). With the default (None = unbounded), Orbax copies
# the entire ~40GB tree to host memory at once and runs the multi-threaded ocdbt writer flat
# out -- observed as an ~80GB RAM spike + ~1000% CPU at save time (dominated by fp32 optimizer
# m,v and EMA params). Capping concurrent GB bounds both peak host memory and the number of
# busy writer threads. This only affects write buffering/concurrency, NOT the on-disk format
# (resume-compatible).
_SAVE_CONCURRENT_GB = 24


def initialize_checkpoint_dir(
    checkpoint_dir: epath.Path | str, *, keep_period: int | None, overwrite: bool, resume: bool
) -> tuple[ocp.CheckpointManager, bool]:
    checkpoint_dir = epath.Path(checkpoint_dir).resolve()
    resuming = False
    if checkpoint_dir.exists():
        if overwrite:
            checkpoint_dir.rmtree()
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
            logging.info(f"Wiped checkpoint directory {checkpoint_dir}")
        elif resume:
            resuming = True
        else:
            raise FileExistsError(
                f"Checkpoint directory {checkpoint_dir} already exists. Use --overwrite or --resume "
                "to indicate how to handle it."
            )

    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # If we're resuming, protect the checkpoint we're resuming from so the CheckpointManager
    # doesn't delete it during cleanup (e.g. right after the next checkpoint is saved). This
    # matters when the resumed checkpoint is not a multiple of `keep_period` (e.g. the final
    # checkpoint of the previous run), since `max_to_keep=1` would otherwise drop it. The
    # protection is persisted to disk so it survives across multiple resume sessions.
    should_keep_fn = None
    if resuming:
        existing_steps = ocp.utils.checkpoint_steps(checkpoint_dir)
        if existing_steps:
            resume_step = max(existing_steps)
            if resume_step > 0:
                # Persist the resume step so it is protected now and in future resume sessions.
                protected_steps = _load_protected_steps(checkpoint_dir)
                protected_steps.add(resume_step)
                _save_protected_steps(checkpoint_dir, protected_steps)
                protected = frozenset(protected_steps)

                def should_keep_fn(step: int) -> bool:
                    # Keep all previously resumed-from checkpoints, and preserve keep_period behavior.
                    return step in protected or (keep_period is not None and step % keep_period == 0)

                logging.info(
                    f"Resuming from checkpoint step {resume_step}; protecting it from deletion "
                    f"(protected steps: {sorted(protected_steps)})."
                )

    mngr = ocp.CheckpointManager(
        checkpoint_dir,
        item_handlers={
            "assets": CallbackHandler(),
            "train_state": ocp.PyTreeCheckpointHandler(save_concurrent_gb=_SAVE_CONCURRENT_GB),
            "params": ocp.PyTreeCheckpointHandler(save_concurrent_gb=_SAVE_CONCURRENT_GB),
        },
        options=ocp.CheckpointManagerOptions(
            max_to_keep=1,
            keep_period=keep_period,
            should_keep_fn=should_keep_fn,
            create=False,
            async_options=ocp.AsyncOptions(timeout_secs=7200),
        ),
    )

    # Special case: the checkpoint directory exists and the user requests to resume training, but the training run did
    # not get to the first checkpoint saved. In this case, we don't actually want the train script to try and restore a
    # checkpoint, since it will fail.
    if resuming and tuple(mngr.all_steps()) in [(), (0,)]:
        logging.info("Checkpoint directory exists, but does not contain any checkpoints. Aborting resume.")
        resuming = False

    return mngr, resuming


def save_state(
    checkpoint_manager: ocp.CheckpointManager,
    state: training_utils.TrainState,
    data_loader: _data_loader.DataLoader,
    step: int,
):
    def save_assets(directory: epath.Path):
        # Persist the normalization stats with the checkpoint so inference can use exactly
        # the same stats the model was trained with, independent of code/config changes.
        # Groot-style configs have asset_id=None, so save at the assets root instead of
        # under a per-asset subdirectory.
        data_config = data_loader.data_config()
        norm_stats = data_config.norm_stats
        if norm_stats is None:
            return
        if data_config.asset_id is not None:
            _normalize.save(directory / data_config.asset_id, norm_stats)
        else:
            _normalize.save(directory, norm_stats)

    # Split params that can be used for inference into a separate item.
    with at.disable_typechecking():
        train_state, params = _split_params(state)
    items = {
        "assets": save_assets,
        "train_state": train_state,
        "params": {"params": params},
    }
    checkpoint_manager.save(step, items)


def restore_state(
    checkpoint_manager: ocp.CheckpointManager,
    state: training_utils.TrainState,
    data_loader: _data_loader.DataLoader,
    step: int | None = None,
) -> training_utils.TrainState:
    del data_loader

    with at.disable_typechecking():
        # Split params that can be used for inference into a separate item.
        train_state, params = _split_params(state)
        restored = checkpoint_manager.restore(
            step,
            items={
                "train_state": train_state,
                "params": {"params": params},
            },
        )
    return _merge_params(restored["train_state"], restored["params"])


def load_norm_stats(assets_dir: epath.Path | str, asset_id: str) -> dict[str, _normalize.NormStats] | None:
    norm_stats_dir = epath.Path(assets_dir) / asset_id
    norm_stats = _normalize.load(norm_stats_dir)
    logging.info(f"Loaded norm stats from {norm_stats_dir}")
    return norm_stats


class Callback(Protocol):
    def __call__(self, directory: epath.Path) -> None: ...


class CallbackHandler(ocp.AsyncCheckpointHandler):
    """A CheckpointHandler for calling an arbitrary function asynchronously. Only for saving, not for restoring."""

    def save(self, directory: epath.Path, args: CallbackSave):
        if jax.process_index() == 0:
            args.callback(directory)

    async def async_save(self, directory: epath.Path, args: CallbackSave) -> list[futures.Future]:
        return [future.CommitFutureAwaitingContractedSignals(asyncio.to_thread(self.save, directory, args))]

    def restore(self, *args, **kwargs):
        raise NotImplementedError("CallbackHandler does not support restore")


@ocp.args.register_with_handler(CallbackHandler, for_save=True)
@dataclasses.dataclass
class CallbackSave(ocp.args.CheckpointArgs):
    callback: Callback


@ocp.args.register_with_handler(CallbackHandler, for_restore=True)
class CallbackRestore(ocp.args.CheckpointArgs): ...


def _split_params(state: training_utils.TrainState) -> tuple[training_utils.TrainState, at.Params]:
    if state.ema_params is not None:
        params = state.ema_params
        train_state = dataclasses.replace(state, ema_params=None)
    else:
        params = state.params
        train_state = dataclasses.replace(state, params={})
    return train_state, params


def _merge_params(train_state: training_utils.TrainState, params: dict[str, at.Params]) -> training_utils.TrainState:
    # Revert the logic inside `_split_params`. Assumes that existence of `params` means that EMA params were used during the split.
    if train_state.params:
        return dataclasses.replace(train_state, ema_params=params["params"])
    return dataclasses.replace(train_state, params=params["params"])

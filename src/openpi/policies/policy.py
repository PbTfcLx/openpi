from collections.abc import Sequence
import copy
import logging
import pathlib
import time
from typing import Any, TypeAlias

import flax
import flax.traverse_util
import jax
import jax.numpy as jnp
import numpy as np
from openpi_client import base_policy as _base_policy
import torch
from typing_extensions import override

from openpi import transforms as _transforms
from openpi.models import model as _model
from openpi.shared import array_typing as at
from openpi.shared import nnx_utils

BasePolicy: TypeAlias = _base_policy.BasePolicy


class Policy(BasePolicy):
    def __init__(
        self,
        model: _model.BaseModel,
        *,
        rng: at.KeyArrayLike | None = None,
        transforms: Sequence[_transforms.DataTransformFn] = (),
        output_transforms: Sequence[_transforms.DataTransformFn] = (),
        sample_kwargs: dict[str, Any] | None = None,
        metadata: dict[str, Any] | None = None,
        pytorch_device: str = "cpu",
        is_pytorch: bool = False,
    ):
        """Initialize the Policy.

        Args:
            model: The model to use for action sampling.
            rng: Random number generator key for JAX models. Ignored for PyTorch models.
            transforms: Input data transformations to apply before inference.
            output_transforms: Output data transformations to apply after inference.
            sample_kwargs: Additional keyword arguments to pass to model.sample_actions.
            metadata: Additional metadata to store with the policy.
            pytorch_device: Device to use for PyTorch models (e.g., "cpu", "cuda:0").
                          Only relevant when is_pytorch=True.
            is_pytorch: Whether the model is a PyTorch model. If False, assumes JAX model.
        """
        self._model = model
        self._input_transform = _transforms.compose(transforms)
        self._output_transform = _transforms.compose(output_transforms)
        self._sample_kwargs = sample_kwargs or {}
        self._metadata = metadata or {}
        self._is_pytorch_model = is_pytorch
        self._pytorch_device = pytorch_device

        if self._is_pytorch_model:
            self._model = self._model.to(pytorch_device)
            self._model.eval()
            self._sample_actions = model.sample_actions
        else:
            # JAX model setup
            self._sample_actions = nnx_utils.module_jit(model.sample_actions)
            self._rng = rng or jax.random.key(0)

    @override
    def infer(self, obs: dict, *, noise: np.ndarray | None = None) -> dict:  # type: ignore[misc]
        # Optional eval determinism keys. When present, derive the action-sampling
        # noise deterministically from (episode_seed, replan_idx) instead of the
        # advancing global RNG, making inference order-independent so runs are
        # reproducible and different checkpoints can be compared on same scenes.
        rng_seed = None
        if "eval/episode_seed" in obs and "eval/replan_idx" in obs:
            rng_seed = (int(obs["eval/episode_seed"]), int(obs["eval/replan_idx"]))

        # Classifier-free guidance (optional): if ``cfg_scale`` is set in sample_kwargs, the
        # observation is additionally run through the input transform with an *empty* instruction
        # (state kept) to obtain an unconditional observation. Only supported for JAX pi0/pi05.
        cfg_scale = self._sample_kwargs.get("cfg_scale")
        if cfg_scale is not None:
            if self._is_pytorch_model:
                raise NotImplementedError("CFG inference is only supported for JAX models.")
            if not hasattr(self._model, "pi05"):
                raise NotImplementedError("CFG inference is only supported for pi0/pi05 models.")

        # Make a copy since transformations may modify the inputs in place.
        def run_input_transform(raw: dict) -> dict:
            inputs = jax.tree.map(lambda x: x, raw)
            # These control keys are consumed here; strip them before transforms.
            inputs.pop("eval/episode_seed", None)
            inputs.pop("eval/replan_idx", None)
            return self._input_transform(inputs)

        inputs = run_input_transform(obs)
        uncond_inputs = None
        if cfg_scale is not None:
            raw_uncond = copy.deepcopy(obs)
            raw_uncond["prompt"] = ""
            uncond_inputs = run_input_transform(raw_uncond)

        if not self._is_pytorch_model:
            # Make a batch and convert to jax.Array.
            inputs = jax.tree.map(lambda x: jnp.asarray(x)[np.newaxis, ...], inputs)
            if uncond_inputs is not None:
                uncond_inputs = jax.tree.map(lambda x: jnp.asarray(x)[np.newaxis, ...], uncond_inputs)
            self._rng, sample_rng_or_pytorch_device = jax.random.split(self._rng)
        else:
            # Convert inputs to PyTorch tensors and move to correct device
            inputs = jax.tree.map(lambda x: torch.from_numpy(np.array(x)).to(self._pytorch_device)[None, ...], inputs)
            sample_rng_or_pytorch_device = self._pytorch_device

        # Prepare kwargs for sample_actions
        sample_kwargs = dict(self._sample_kwargs)
        sample_kwargs.pop("cfg_scale", None)  # reserved: handled via the unconditional branch below
        if noise is not None:
            noise = torch.from_numpy(noise).to(self._pytorch_device) if self._is_pytorch_model else jnp.asarray(noise)

            if noise.ndim == 2:  # If noise is (action_horizon, action_dim), add batch dimension
                noise = noise[None, ...]  # Make it (1, action_horizon, action_dim)
            sample_kwargs["noise"] = noise
        elif not self._is_pytorch_model and rng_seed is not None:
            # Deterministic flow-matching noise: a pure function of the episode
            # seed and replan index, independent of the shared RNG / call order.
            noise_key = jax.random.fold_in(jax.random.key(rng_seed[0]), rng_seed[1])
            sample_kwargs["noise"] = jax.random.normal(
                noise_key, (1, self._model.action_horizon, self._model.action_dim)
            )

        observation = _model.Observation.from_dict(inputs)
        if uncond_inputs is not None:
            sample_kwargs["uncond_observation"] = _model.Observation.from_dict(uncond_inputs)
            sample_kwargs["cfg_scale"] = float(cfg_scale)
        start_time = time.monotonic()
        outputs = {
            "state": inputs["state"],
            "actions": self._sample_actions(sample_rng_or_pytorch_device, observation, **sample_kwargs),
        }
        model_time = time.monotonic() - start_time
        if self._is_pytorch_model:
            outputs = jax.tree.map(lambda x: np.asarray(x[0, ...].detach().cpu()), outputs)
        else:
            outputs = jax.tree.map(lambda x: np.asarray(x[0, ...]), outputs)

        outputs = self._output_transform(outputs)
        outputs["policy_timing"] = {
            "infer_ms": model_time * 1000,
        }
        return outputs

    @property
    def metadata(self) -> dict[str, Any]:
        return self._metadata


class PolicyRecorder(_base_policy.BasePolicy):
    """Records the policy's behavior to disk."""

    def __init__(self, policy: _base_policy.BasePolicy, record_dir: str):
        self._policy = policy

        logging.info(f"Dumping policy records to: {record_dir}")
        self._record_dir = pathlib.Path(record_dir)
        self._record_dir.mkdir(parents=True, exist_ok=True)
        self._record_step = 0

    @override
    def infer(self, obs: dict) -> dict:  # type: ignore[misc]
        results = self._policy.infer(obs)

        data = {"inputs": obs, "outputs": results}
        data = flax.traverse_util.flatten_dict(data, sep="/")

        output_path = self._record_dir / f"step_{self._record_step}"
        self._record_step += 1

        np.save(output_path, np.asarray(data))
        return results

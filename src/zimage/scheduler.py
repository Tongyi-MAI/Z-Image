"""FlowMatchEulerDiscreteScheduler implementation."""

# Modified from https://github.com/huggingface/diffusers/blob/main/src/diffusers/schedulers/scheduling_flow_match_euler_discrete.py
from dataclasses import dataclass
import math
from typing import List, Optional, Tuple, Union

import numpy as np
import torch


@dataclass
class SchedulerOutput:
    prev_sample: torch.FloatTensor


class SchedulerConfig:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

    def get(self, key, default=None):
        return self.__dict__.get(key, default)

    def __getattr__(self, name):
        return self.__dict__.get(name)


class FlowMatchEulerDiscreteScheduler:
    """Euler scheduler for flow matching."""

    def __init__(
        self,
        num_train_timesteps: int = 1000,
        shift: float = 1.0,
        use_dynamic_shifting: bool = False,
        **kwargs,
    ):
        self.num_train_timesteps = num_train_timesteps
        self.shift = shift
        self.use_dynamic_shifting = use_dynamic_shifting
        self.config = SchedulerConfig(
            num_train_timesteps=num_train_timesteps,
            shift=shift,
            use_dynamic_shifting=use_dynamic_shifting,
        )

        timesteps = np.linspace(
            1, num_train_timesteps, num_train_timesteps, dtype=np.float32
        )[::-1].copy()
        timesteps = torch.from_numpy(timesteps).to(dtype=torch.float32)
        sigmas = timesteps / num_train_timesteps

        if not use_dynamic_shifting:
            sigmas = shift * sigmas / (1 + (shift - 1) * sigmas)

        self.timesteps = sigmas * num_train_timesteps
        self.sigmas = sigmas.to("cpu")
        self.sigma_min = self.sigmas[-1].item()
        self.sigma_max = self.sigmas[0].item()

        self._step_index = None
        self._begin_index = None

    def set_timesteps(
        self,
        num_inference_steps: Optional[int] = None,
        device: Union[str, torch.device] = None,
        sigmas: Optional[List[float]] = None,
        mu: Optional[float] = None,
        timesteps: Optional[List[float]] = None,
    ):
        passed_timesteps = timesteps
        if num_inference_steps is None:
            num_inference_steps = len(sigmas) if sigmas is not None else len(timesteps)

        self.num_inference_steps = num_inference_steps

        if sigmas is None:
            if timesteps is None:
                timesteps = np.linspace(
                    self._sigma_to_t(self.sigma_max),
                    self._sigma_to_t(self.sigma_min),
                    num_inference_steps + 1,
                )[:-1]
            sigmas = timesteps / self.num_train_timesteps
        else:
            sigmas = np.array(sigmas).astype(np.float32)

        if self.use_dynamic_shifting:
            sigmas = self.time_shift(mu, 1.0, sigmas)
        else:
            sigmas = self.shift * sigmas / (1 + (self.shift - 1) * sigmas)

        sigmas = torch.from_numpy(sigmas).to(dtype=torch.float32, device=device)

        if passed_timesteps is None:
            timesteps = sigmas * self.num_train_timesteps
        else:
            timesteps = torch.from_numpy(passed_timesteps).to(
                dtype=torch.float32, device=device
            )

        sigmas = torch.cat([sigmas, torch.zeros(1, device=sigmas.device)])

        self.timesteps = timesteps
        self.sigmas = sigmas
        self._step_index = None
        self._begin_index = None

    def index_for_timestep(self, timestep, schedule_timesteps=None):
        if schedule_timesteps is None:
            schedule_timesteps = self.timesteps

        indices = (schedule_timesteps == timestep).nonzero()
        pos = 1 if len(indices) > 1 else 0
        return indices[pos].item()

    def _init_step_index(self, timestep):
        if self._begin_index is None:
            if isinstance(timestep, torch.Tensor):
                timestep = timestep.to(self.timesteps.device)
            self._step_index = self.index_for_timestep(timestep)
        else:
            self._step_index = self._begin_index

    def step(
        self,
        model_output: torch.FloatTensor,
        timestep: Union[float, torch.FloatTensor],
        sample: torch.FloatTensor,
        return_dict: bool = True,
        **kwargs,
    ) -> Union[SchedulerOutput, Tuple]:
        """Predict the sample at the previous timestep."""
        if self._step_index is None:
            self._init_step_index(timestep)

        sample = sample.to(torch.float32)
        sigma_idx = self._step_index
        sigma = self.sigmas[sigma_idx]
        sigma_next = self.sigmas[sigma_idx + 1]

        dt = sigma_next - sigma
        prev_sample = sample + dt * model_output
        self._step_index += 1
        prev_sample = prev_sample.to(model_output.dtype)

        if not return_dict:
            return (prev_sample,)
        return SchedulerOutput(prev_sample=prev_sample)

    def _sigma_to_t(self, sigma):
        return sigma * self.num_train_timesteps

    def time_shift(self, mu: float, sigma: float, t: torch.Tensor):
        return math.exp(mu) / (math.exp(mu) + (1 / t - 1) ** sigma)

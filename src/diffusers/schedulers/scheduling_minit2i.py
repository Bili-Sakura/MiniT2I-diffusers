from __future__ import annotations

import torch

from .._hf import load_hf_diffusers_submodules

_hf = load_hf_diffusers_submodules(
    "configuration_utils",
    "schedulers.scheduling_utils",
)
ConfigMixin = _hf["configuration_utils"].ConfigMixin
register_to_config = _hf["configuration_utils"].register_to_config
SchedulerMixin = _hf["schedulers.scheduling_utils"].SchedulerMixin


class MiniT2IFlowMatchScheduler(SchedulerMixin, ConfigMixin):
    """Flow-matching timestep scheduler for MiniT2I."""

    config_name = "scheduler_config.json"

    @register_to_config
    def __init__(
        self,
        train_t_schedule: str = "lognorm",
        t_lognorm_mu: float = -0.8,
        t_lognorm_sigma: float = 0.8,
        num_inference_steps: int = 100,
    ):
        if train_t_schedule not in {"uniform", "lognorm"}:
            raise ValueError(f"Unsupported train_t_schedule: {train_t_schedule}")

    def sample_train_timesteps(self, batch_size, device, dtype=torch.float32, generator=None):
        if self.config.train_t_schedule == "uniform":
            return torch.rand(batch_size, device=device, dtype=dtype, generator=generator)
        normal = torch.randn(batch_size, device=device, dtype=torch.float32, generator=generator)
        normal = normal * self.config.t_lognorm_sigma + self.config.t_lognorm_mu
        return torch.sigmoid(normal).to(dtype=dtype)

    def get_inference_timesteps(self, num_inference_steps=None, device=None, dtype=torch.float32):
        steps = int(num_inference_steps or self.config.num_inference_steps)
        return torch.linspace(0.0, 1.0, steps + 1, device=device, dtype=dtype)

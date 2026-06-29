from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class MiniT2ITrainingTimestepConfig:
    train_t_schedule: str = "lognorm"
    t_lognorm_mu: float = -0.8
    t_lognorm_sigma: float = 0.8


def sample_train_timesteps(
    batch_size: int,
    device: torch.device,
    config: MiniT2ITrainingTimestepConfig,
    dtype: torch.dtype = torch.float32,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    if config.train_t_schedule == "uniform":
        return torch.rand(batch_size, device=device, dtype=dtype, generator=generator)
    if config.train_t_schedule != "lognorm":
        raise ValueError(f"Unsupported train_t_schedule: {config.train_t_schedule}")
    normal = torch.randn(batch_size, device=device, dtype=torch.float32, generator=generator)
    normal = normal * config.t_lognorm_sigma + config.t_lognorm_mu
    return torch.sigmoid(normal).to(dtype=dtype)


def load_training_timestep_config(
    scheduler_dir: str | None = None,
    *,
    train_t_schedule: str = "lognorm",
    t_lognorm_mu: float = -0.8,
    t_lognorm_sigma: float = 0.8,
) -> MiniT2ITrainingTimestepConfig:
    if scheduler_dir is None:
        return MiniT2ITrainingTimestepConfig(
            train_t_schedule=train_t_schedule,
            t_lognorm_mu=t_lognorm_mu,
            t_lognorm_sigma=t_lognorm_sigma,
        )

    import json
    from pathlib import Path

    config_path = Path(scheduler_dir) / "scheduler_config.json"
    if not config_path.exists():
        raise FileNotFoundError(f"Missing scheduler config: {config_path}")

    raw = json.loads(config_path.read_text(encoding="utf-8"))
    return MiniT2ITrainingTimestepConfig(
        train_t_schedule=raw.get("train_t_schedule", train_t_schedule),
        t_lognorm_mu=float(raw.get("t_lognorm_mu", t_lognorm_mu)),
        t_lognorm_sigma=float(raw.get("t_lognorm_sigma", t_lognorm_sigma)),
    )

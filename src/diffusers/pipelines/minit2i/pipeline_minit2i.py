# Copyright 2025 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from __future__ import annotations

import inspect
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

os.environ.setdefault("USE_FLAX", "0")
os.environ.setdefault("TRANSFORMERS_NO_FLAX", "1")

import torch
from huggingface_hub import snapshot_download
from PIL import Image
from transformers import AutoTokenizer, T5EncoderModel
from transformers import logging as transformers_logging

from ..._hf import load_hf_diffusers_submodules
from ...models.transformers.transformer_minit2i import MiniT2IMMJiTModel

_hf = load_hf_diffusers_submodules(
    "pipelines.pipeline_utils",
    "schedulers",
    "schedulers.scheduling_flow_match_euler_discrete",
    "schedulers.scheduling_utils",
    "utils.torch_utils",
)
DiffusionPipeline = _hf["pipelines.pipeline_utils"].DiffusionPipeline
ImagePipelineOutput = _hf["pipelines.pipeline_utils"].ImagePipelineOutput
FlowMatchEulerDiscreteScheduler = _hf["schedulers.scheduling_flow_match_euler_discrete"].FlowMatchEulerDiscreteScheduler
KarrasDiffusionSchedulers = _hf["schedulers.scheduling_utils"].KarrasDiffusionSchedulers
randn_tensor = _hf["utils.torch_utils"].randn_tensor

transformers_logging.set_verbosity_error()

DEFAULT_NUM_INFERENCE_STEPS = 100
NOISE_INIT_SCALE = 2.0

EXAMPLE_DOC_STRING = """
    Examples:
        ```py
        >>> from pathlib import Path
        >>> import torch
        >>> from diffusers import DiffusionPipeline, FlowMatchEulerDiscreteScheduler

        >>> model_dir = Path("./minit2i-diffusers").resolve()
        >>> pipe = DiffusionPipeline.from_pretrained(
        ...     str(model_dir),
        ...     local_files_only=True,
        ...     custom_pipeline=str(model_dir / "pipeline.py"),
        ...     trust_remote_code=True,
        ...     torch_dtype=torch.bfloat16,
        ... )
        >>> pipe.to("cuda")
        >>> pipe.scheduler = FlowMatchEulerDiscreteScheduler.from_config(pipe.scheduler.config)

        >>> generator = torch.Generator(device="cuda").manual_seed(42)
        >>> image = pipe(
        ...     "a cinematic portrait of a robot musician",
        ...     num_inference_steps=100,
        ...     guidance_scale=6.0,
        ...     generator=generator,
        ... ).images[0]
        >>> image.save("demo.png")
        ```
"""

MODEL_ALIASES: Dict[str, str] = {
    "b": "minit2i-b-16",
    "b16": "minit2i-b-16",
    "b-16": "minit2i-b-16",
    "base": "minit2i-b-16",
    "minit2i-b16": "minit2i-b-16",
    "minit2i-b-16": "minit2i-b-16",
    "minit2i-b/16": "minit2i-b-16",
    "l": "minit2i-l-16",
    "l16": "minit2i-l-16",
    "l-16": "minit2i-l-16",
    "large": "minit2i-l-16",
    "minit2i-l16": "minit2i-l-16",
    "minit2i-l-16": "minit2i-l-16",
    "minit2i-l/16": "minit2i-l-16",
}


def resolve_model_type(model_type: str) -> str:
    key = model_type.lower().replace("_", "-")
    if key not in MODEL_ALIASES:
        choices = ", ".join(sorted(set(MODEL_ALIASES)))
        raise ValueError(f"Unknown model_type={model_type!r}. Expected one of: {choices}")
    return MODEL_ALIASES[key]


class MiniT2ITextToImagePipeline(DiffusionPipeline):
    r"""
    Text-to-image pipeline for MiniT2I pixel-space flow matching.

    Parameters:
        transformer ([`MiniT2IMMJiTModel`]):
            MiniT2I MM-JiT transformer that predicts flow-matching velocity in pixel space.
        scheduler ([`FlowMatchEulerDiscreteScheduler`]):
            Flow-matching Euler scheduler. Other [`KarrasDiffusionSchedulers`] can be swapped at inference time.
        tokenizer ([`AutoTokenizer`], *optional*):
            Tokenizer for the text encoder.
        text_encoder ([`T5EncoderModel`], *optional*):
            Text encoder used to embed prompts.
    """

    model_cpu_offload_seq = "text_encoder->transformer"
    _optional_components = ["tokenizer", "text_encoder"]

    def __init__(
        self,
        transformer: MiniT2IMMJiTModel,
        scheduler: KarrasDiffusionSchedulers,
        tokenizer=None,
        text_encoder=None,
        text_encoder_name: str = "google/flan-t5-large",
        model_type: str = "b16",
        repo_id_or_path: Optional[str] = None,
        default_num_inference_steps: int = DEFAULT_NUM_INFERENCE_STEPS,
    ):
        super().__init__()
        if scheduler is None:
            scheduler = self._default_inference_scheduler()
        self.register_modules(
            transformer=transformer,
            scheduler=scheduler,
            tokenizer=tokenizer,
            text_encoder=text_encoder,
        )
        self.register_to_config(
            text_encoder_name=text_encoder_name,
            model_type=model_type,
            repo_id_or_path=repo_id_or_path,
            default_num_inference_steps=int(default_num_inference_steps),
        )
        self._variant_transformers: Dict[str, MiniT2IMMJiTModel] = {}
        self._active_model_type = resolve_model_type(model_type)

    @staticmethod
    def _default_inference_scheduler() -> FlowMatchEulerDiscreteScheduler:
        return FlowMatchEulerDiscreteScheduler(
            num_train_timesteps=1000,
            shift=1.0,
            stochastic_sampling=False,
        )

    @classmethod
    def _load_scheduler_from_dir(
        cls,
        scheduler_dir: Path,
        model_kwargs: Dict[str, Any],
    ) -> Tuple[KarrasDiffusionSchedulers, int]:
        config_path = scheduler_dir / "scheduler_config.json"
        if not config_path.exists():
            return cls._default_inference_scheduler(), DEFAULT_NUM_INFERENCE_STEPS

        config = json.loads(config_path.read_text(encoding="utf-8"))
        class_name = config.get("_class_name", "")
        default_steps = int(config.get("num_inference_steps", DEFAULT_NUM_INFERENCE_STEPS))

        if class_name == "MiniT2IFlowMatchScheduler":
            return cls._default_inference_scheduler(), default_steps

        schedulers_pkg = _hf["schedulers"]
        if hasattr(schedulers_pkg, class_name):
            scheduler_cls = getattr(schedulers_pkg, class_name)
            return scheduler_cls.from_pretrained(str(scheduler_dir), **model_kwargs), default_steps

        return cls._default_inference_scheduler(), default_steps

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: Union[str, os.PathLike],
        torch_dtype: Optional[torch.dtype] = None,
        text_encoder_dtype: torch.dtype = torch.float32,
        local_files_only: bool = False,
        revision: Optional[str] = None,
        cache_dir: Optional[Union[str, os.PathLike]] = None,
        model_type: str = "b16",
        variant: Optional[str] = None,
        **kwargs,
    ):
        model_kwargs = dict(kwargs)
        model_kwargs.pop("custom_pipeline", None)
        model_kwargs.pop("trust_remote_code", None)

        root = Path(pretrained_model_name_or_path)
        if not root.exists():
            root = Path(
                snapshot_download(
                    repo_id=str(pretrained_model_name_or_path),
                    revision=revision,
                    cache_dir=cache_dir,
                    local_files_only=local_files_only,
                )
            )

        variant_dir = resolve_model_type(model_type)
        transformer_path = cls._resolve_transformer_path(root, variant_dir)
        transformer = MiniT2IMMJiTModel.from_pretrained(
            transformer_path,
            torch_dtype=torch_dtype,
            variant=variant,
            **model_kwargs,
        )

        scheduler_dir = root / "scheduler"
        if scheduler_dir.exists():
            scheduler, default_num_inference_steps = cls._load_scheduler_from_dir(scheduler_dir, model_kwargs)
        else:
            scheduler = cls._default_inference_scheduler()
            default_num_inference_steps = DEFAULT_NUM_INFERENCE_STEPS

        text_encoder_name = transformer.mmjit_config.llm
        tokenizer = AutoTokenizer.from_pretrained(text_encoder_name, local_files_only=local_files_only)
        text_encoder = T5EncoderModel.from_pretrained(
            text_encoder_name,
            torch_dtype=text_encoder_dtype,
            local_files_only=local_files_only,
        )
        return cls(
            transformer=transformer,
            scheduler=scheduler,
            tokenizer=tokenizer,
            text_encoder=text_encoder,
            text_encoder_name=text_encoder_name,
            model_type=model_type,
            repo_id_or_path=str(pretrained_model_name_or_path),
            default_num_inference_steps=default_num_inference_steps,
        )

    @staticmethod
    def _resolve_transformer_path(root: Path, variant_dir: str) -> Path:
        variant_transformer = root / variant_dir / "transformer"
        if variant_transformer.exists():
            return variant_transformer
        root_transformer = root / "transformer"
        if root_transformer.exists():
            return root_transformer
        raise FileNotFoundError(
            f"Could not find transformer weights under {root}. "
            f"Tried {variant_transformer} and {root_transformer}."
        )

    def _get_transformer(
        self,
        model_type: Optional[str],
        repo_id_or_path: Optional[str],
        torch_dtype: Optional[torch.dtype] = None,
        variant: Optional[str] = None,
    ) -> MiniT2IMMJiTModel:
        active_type = resolve_model_type(model_type or self.config.model_type)
        if active_type == self._active_model_type and self.transformer is not None:
            return self.transformer
        if active_type in self._variant_transformers:
            return self._variant_transformers[active_type]

        repo = repo_id_or_path or self.config.repo_id_or_path
        if repo is None:
            raise ValueError("model_type switching requires repo_id_or_path to be set on the pipeline.")

        root = Path(repo)
        if not root.exists():
            root = Path(snapshot_download(repo_id=str(repo)))
        transformer = MiniT2IMMJiTModel.from_pretrained(
            self._resolve_transformer_path(root, active_type),
            torch_dtype=torch_dtype,
            variant=variant,
        )
        self._variant_transformers[active_type] = transformer
        if active_type == resolve_model_type(self.config.model_type):
            self.transformer = transformer
            self._active_model_type = active_type
        return transformer

    def save_pretrained(self, save_directory: Union[str, os.PathLike], **kwargs):
        save_directory = Path(save_directory)
        save_directory.mkdir(parents=True, exist_ok=True)
        self.transformer.save_pretrained(save_directory / "transformer", **kwargs)
        self.scheduler.save_pretrained(save_directory / "scheduler")
        super().save_pretrained(save_directory, **kwargs)

    @staticmethod
    def prepare_extra_step_kwargs(
        scheduler: KarrasDiffusionSchedulers,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]],
    ) -> Dict[str, Any]:
        kwargs: Dict[str, Any] = {}
        step_params = set(inspect.signature(scheduler.step).parameters.keys())
        if "generator" in step_params:
            kwargs["generator"] = generator
        return kwargs

    def check_inputs(
        self,
        prompt: Union[str, List[str]],
        guidance_scale: float,
        num_inference_steps: int,
        output_type: str,
    ) -> None:
        if not isinstance(prompt, str) and not (isinstance(prompt, list) and all(isinstance(p, str) for p in prompt)):
            raise TypeError(f"`prompt` must be a string or list of strings, got {type(prompt)}.")
        if guidance_scale < 0:
            raise ValueError(f"`guidance_scale` must be non-negative, got {guidance_scale}.")
        if num_inference_steps <= 0:
            raise ValueError(f"`num_inference_steps` must be positive, got {num_inference_steps}.")
        if output_type not in {"pil", "np", "pt", "latent"}:
            raise ValueError(f"Unsupported `output_type`: {output_type}")

    def prepare_latents(
        self,
        batch_size: int,
        image_size: int,
        in_channels: int,
        device: torch.device,
        dtype: torch.dtype,
        generator: Optional[torch.Generator] = None,
        latents: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        shape = (batch_size, in_channels, image_size, image_size)
        if latents is None:
            latents = randn_tensor(shape, generator=generator, device=device, dtype=dtype)
            latents = latents * NOISE_INIT_SCALE
        else:
            latents = latents.to(device=device, dtype=dtype)
            if tuple(latents.shape) != shape:
                raise ValueError(f"Invalid `latents` shape: {tuple(latents.shape)}. Expected {shape}.")
        return latents

    def _encode_prompt(
        self,
        prompt: Union[str, List[str]],
        device: torch.device,
        transformer: Optional[MiniT2IMMJiTModel] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if isinstance(prompt, str):
            prompt = [prompt]
        transformer = transformer or self.transformer
        if self.tokenizer is None:
            self.tokenizer = AutoTokenizer.from_pretrained(self.config.text_encoder_name)
        if self.text_encoder is None:
            self.text_encoder = T5EncoderModel.from_pretrained(self.config.text_encoder_name)
        if next(self.text_encoder.parameters()).device != device:
            self.text_encoder.to(device)
        cfg = transformer.mmjit_config
        tokens = self.tokenizer(
            prompt,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=cfg.prompt_length,
        )
        input_ids = tokens.input_ids.to(device)
        attn = tokens.attention_mask.to(device)
        text = self.text_encoder(input_ids=input_ids, attention_mask=attn).last_hidden_state
        return text, attn

    @staticmethod
    def _cfg_velocity(
        transformer: MiniT2IMMJiTModel,
        x: torch.Tensor,
        t: torch.Tensor,
        text: torch.Tensor,
        mask: torch.Tensor,
        cfg_scale: float,
    ) -> torch.Tensor:
        batch_size = x.shape[0]
        doubled_x = torch.cat([x, x], dim=0)
        doubled_t = torch.cat([t, t], dim=0)
        doubled_text = torch.cat([text, text], dim=0)
        null_mask = torch.zeros_like(mask)
        doubled_mask = torch.cat([mask, null_mask], dim=0)
        velocity = transformer.pred_velocity(doubled_x, doubled_t, doubled_text, doubled_mask)
        cond, uncond = velocity[:batch_size], velocity[batch_size:]
        cfg_interval = transformer.mmjit_config.cfg_interval
        use_cfg = ((t >= cfg_interval[0]) & (t <= cfg_interval[1])).to(velocity.dtype)
        scale = torch.where(
            use_cfg[:, None, None, None] > 0,
            torch.tensor(cfg_scale, device=x.device, dtype=velocity.dtype),
            torch.tensor(1.0, device=x.device, dtype=velocity.dtype),
        )
        return uncond + (cond - uncond) * scale

    @torch.no_grad()
    def __call__(
        self,
        prompt: Union[str, List[str]],
        num_images_per_prompt: int = 1,
        guidance_scale: float = 6.0,
        num_inference_steps: Optional[int] = None,
        generator: Optional[torch.Generator] = None,
        latents: Optional[torch.Tensor] = None,
        output_type: str = "pil",
        return_dict: bool = True,
        progress: bool = True,
        model_type: Optional[str] = None,
        repo_id_or_path: Optional[str] = None,
        variant: Optional[str] = None,
        torch_dtype: Optional[torch.dtype] = None,
    ) -> Union[ImagePipelineOutput, Tuple]:
        r"""
        Generate images from text prompts with MiniT2I.

        Args:
            prompt (`str` or `list[str]`):
                Text prompt or batch of prompts.
            num_images_per_prompt (`int`, defaults to `1`):
                Number of images to generate per prompt.
            guidance_scale (`float`, defaults to `6.0`):
                Classifier-free guidance scale. CFG is active when `guidance_scale != 1.0`.
            num_inference_steps (`int`, *optional*):
                Number of denoising steps. Defaults to the pipeline config value.
            generator (`torch.Generator`, *optional*):
                RNG for reproducibility.
            latents (`torch.Tensor`, *optional*):
                Pre-generated pixel latents with shape `(batch, channels, height, width)`.
            output_type (`str`, defaults to `"pil"`):
                `"pil"`, `"np"`, `"pt"`, or `"latent"`.
            return_dict (`bool`, defaults to `True`):
                Return [`ImagePipelineOutput`] if True.
            progress (`bool`, defaults to `True`):
                Whether to show a progress bar during denoising.
            model_type (`str`, *optional*):
                MiniT2I variant alias such as `"b16"` or `"l16"`.
            repo_id_or_path (`str`, *optional*):
                Hub id or local path used when switching `model_type`.
            variant (`str`, *optional*):
                Weight variant passed to `from_pretrained`.
            torch_dtype (`torch.dtype`, *optional*):
                Optional dtype override when loading a different transformer variant.
        """
        num_inference_steps = int(num_inference_steps or self.config.default_num_inference_steps)
        self.check_inputs(prompt, guidance_scale, num_inference_steps, output_type)

        transformer = self._get_transformer(model_type, repo_id_or_path, torch_dtype=torch_dtype, variant=variant)
        device = self._execution_device
        transformer = transformer.to(device)

        if isinstance(prompt, str):
            prompt_batch = [prompt] * num_images_per_prompt
        else:
            prompt_batch = []
            for entry in prompt:
                prompt_batch.extend([entry] * num_images_per_prompt)

        batch_size = len(prompt_batch)
        mmjit_cfg = transformer.mmjit_config
        model_dtype = next(transformer.parameters()).dtype

        text, attn = self._encode_prompt(prompt_batch, device, transformer=transformer)
        text = text.to(dtype=model_dtype)
        attn = attn.to(dtype=model_dtype)

        if getattr(self.scheduler.config, "stochastic_sampling", False):
            raise ValueError(
                "MiniT2I expects deterministic FlowMatchEulerDiscreteScheduler stepping "
                "(scheduler.config.stochastic_sampling=False)."
            )

        extra_step_kwargs = self.prepare_extra_step_kwargs(self.scheduler, generator=generator)
        self.scheduler.set_timesteps(num_inference_steps, device=device)
        num_train_timesteps = self.scheduler.config.num_train_timesteps

        latents = self.prepare_latents(
            batch_size=batch_size,
            image_size=mmjit_cfg.image_size,
            in_channels=mmjit_cfg.in_channels,
            device=device,
            dtype=model_dtype,
            generator=generator,
            latents=latents,
        )

        timesteps = self.scheduler.timesteps
        if progress:
            timesteps = self.progress_bar(timesteps)

        using_cfg = guidance_scale != 1.0
        for timestep in timesteps:
            flow_time = 1.0 - float(timestep) / num_train_timesteps
            t = torch.full((batch_size,), flow_time, device=device, dtype=model_dtype)
            if using_cfg:
                velocity = self._cfg_velocity(transformer, latents, t, text, attn, guidance_scale)
            else:
                velocity = transformer.pred_velocity(latents, t, text, attn)

            # MiniT2I integrates velocity from noise (t=0) to data (t=1); flip sign for
            # FlowMatchEulerDiscreteScheduler sigma decreasing from 1 to 0.
            latents = self.scheduler.step(-velocity, timestep, latents, **extra_step_kwargs).prev_sample

        if output_type == "latent":
            images = latents
        else:
            images = (latents.clamp(-1, 1) * 127.5 + 128.0).clamp(0, 255).to(torch.uint8)
            if output_type == "pt":
                images = images.float() / 255.0
            else:
                images = images.permute(0, 2, 3, 1).cpu().numpy()
                if output_type == "pil":
                    images = [Image.fromarray(image) for image in images]

        self.maybe_free_model_hooks()
        if not return_dict:
            return (images,)
        return ImagePipelineOutput(images=images)

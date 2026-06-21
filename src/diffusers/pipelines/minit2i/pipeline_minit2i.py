from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, List, Optional, Union

os.environ.setdefault("USE_FLAX", "0")
os.environ.setdefault("TRANSFORMERS_NO_FLAX", "1")

import torch
from huggingface_hub import snapshot_download
from PIL import Image
from transformers import AutoTokenizer, T5EncoderModel
from transformers import logging as transformers_logging

from ..._hf import load_hf_diffusers_submodule
from ...models.transformers.transformer_minit2i import MiniT2IMMJiTModel
from ...schedulers.scheduling_minit2i import MiniT2IFlowMatchScheduler

_pipeline_utils = load_hf_diffusers_submodule("pipelines.pipeline_utils")
DiffusionPipeline = _pipeline_utils.DiffusionPipeline
ImagePipelineOutput = _pipeline_utils.ImagePipelineOutput

transformers_logging.set_verbosity_error()

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
    """Text-to-image pipeline for MiniT2I pixel-space flow matching."""

    model_cpu_offload_seq = "text_encoder->transformer"
    _optional_components = ["tokenizer", "text_encoder"]

    def __init__(
        self,
        transformer: MiniT2IMMJiTModel,
        scheduler: Optional[MiniT2IFlowMatchScheduler] = None,
        tokenizer=None,
        text_encoder=None,
        text_encoder_name: str = "google/flan-t5-large",
        train_t_schedule: str = "lognorm",
        t_lognorm_mu: float = -0.8,
        t_lognorm_sigma: float = 0.8,
        num_inference_steps: int = 100,
        model_type: str = "b16",
        repo_id_or_path: Optional[str] = None,
    ):
        super().__init__()
        if not isinstance(scheduler, MiniT2IFlowMatchScheduler):
            scheduler = MiniT2IFlowMatchScheduler(
                train_t_schedule=train_t_schedule,
                t_lognorm_mu=t_lognorm_mu,
                t_lognorm_sigma=t_lognorm_sigma,
                num_inference_steps=num_inference_steps,
            )
        self.register_modules(
            transformer=transformer,
            scheduler=scheduler,
            tokenizer=tokenizer,
            text_encoder=text_encoder,
        )
        self.register_to_config(
            text_encoder_name=text_encoder_name,
            train_t_schedule=scheduler.config.train_t_schedule,
            t_lognorm_mu=scheduler.config.t_lognorm_mu,
            t_lognorm_sigma=scheduler.config.t_lognorm_sigma,
            num_inference_steps=scheduler.config.num_inference_steps,
            model_type=model_type,
            repo_id_or_path=repo_id_or_path,
        )
        self._variant_transformers: Dict[str, MiniT2IMMJiTModel] = {}
        self._active_model_type = resolve_model_type(model_type)

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
            scheduler = MiniT2IFlowMatchScheduler.from_pretrained(scheduler_dir)
        else:
            scheduler = MiniT2IFlowMatchScheduler()

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

    def _encode_prompt(self, prompt: Union[str, List[str]], device, transformer: Optional[MiniT2IMMJiTModel] = None):
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

    @torch.no_grad()
    def __call__(
        self,
        prompt: Union[str, List[str]],
        num_images_per_prompt: int = 1,
        guidance_scale: float = 6.0,
        num_inference_steps: Optional[int] = None,
        generator: Optional[torch.Generator] = None,
        output_type: str = "pil",
        return_dict: bool = True,
        progress: bool = True,
        model_type: Optional[str] = None,
        repo_id_or_path: Optional[str] = None,
        variant: Optional[str] = None,
        torch_dtype: Optional[torch.dtype] = None,
    ):
        transformer = self._get_transformer(model_type, repo_id_or_path, torch_dtype=torch_dtype, variant=variant)
        device = self._execution_device
        transformer = transformer.to(device)
        if isinstance(prompt, str):
            prompt_batch = [prompt] * num_images_per_prompt
        else:
            prompt_batch = []
            for p in prompt:
                prompt_batch.extend([p] * num_images_per_prompt)

        old_steps = transformer.mmjit_config.n_T
        transformer.model.cfg.n_T = int(num_inference_steps or self.scheduler.config.num_inference_steps)
        try:
            text, attn = self._encode_prompt(prompt_batch, device, transformer=transformer)
            model_dtype = next(transformer.parameters()).dtype
            images = transformer.sample(
                text.to(dtype=model_dtype),
                attn.to(dtype=model_dtype),
                cfg_scale=guidance_scale,
                generator=generator,
                progress=progress,
            )
        finally:
            transformer.model.cfg.n_T = old_steps

        images = (images.clamp(-1, 1) * 127.5 + 128.0).clamp(0, 255).to(torch.uint8)
        images = images.permute(0, 2, 3, 1).cpu().numpy()
        if output_type == "pil":
            images = [Image.fromarray(image) for image in images]
        if not return_dict:
            return (images,)
        return ImagePipelineOutput(images=images)

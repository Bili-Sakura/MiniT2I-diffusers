#!/usr/bin/env python3
"""Convert a legacy MiniT2I .pt checkpoint to a diffusers-style Hub folder."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

import torch

LIB_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = LIB_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from diffusers.models.transformers.transformer_minit2i import MMJiTConfig, MiniT2IMMJiTModel
from diffusers.schedulers.scheduling_minit2i import MiniT2IFlowMatchScheduler


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Convert MiniT2I checkpoint to diffusers-style directory.")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to legacy checkpoint.pt")
    parser.add_argument("--output", type=str, required=True, help="Output directory")
    parser.add_argument("--variant-name", type=str, default="minit2i-b-16", help="Variant subdirectory name")
    parser.add_argument(
        "--train-t-schedule",
        type=str,
        default="lognorm",
        choices=["uniform", "lognorm"],
        help="Training timestep schedule for exported scheduler",
    )
    parser.add_argument("--t-lognorm-mu", type=float, default=-0.8)
    parser.add_argument("--t-lognorm-sigma", type=float, default=0.8)
    parser.add_argument("--num-inference-steps", type=int, default=100)
    return parser


def build_transformer_from_checkpoint(ckpt_path: Path) -> MiniT2IMMJiTModel:
    payload = torch.load(ckpt_path, map_location="cpu")
    cfg = MMJiTConfig(**payload["config"])
    transformer = MiniT2IMMJiTModel(**asdict(cfg))
    state_dict = {}
    for key, value in payload["state_dict"].items():
        state_dict[f"model.{key}"] = value
    transformer.load_state_dict(state_dict, strict=True)
    return transformer


def main() -> None:
    args = build_parser().parse_args()
    out_dir = Path(args.output)
    variant_dir = out_dir / args.variant_name
    transformer_dir = variant_dir / "transformer"
    scheduler_dir = out_dir / "scheduler"

    transformer = build_transformer_from_checkpoint(Path(args.checkpoint))
    transformer_dir.mkdir(parents=True, exist_ok=True)
    transformer.save_pretrained(transformer_dir)

    scheduler = MiniT2IFlowMatchScheduler(
        train_t_schedule=args.train_t_schedule,
        t_lognorm_mu=args.t_lognorm_mu,
        t_lognorm_sigma=args.t_lognorm_sigma,
        num_inference_steps=args.num_inference_steps,
    )
    scheduler_dir.mkdir(parents=True, exist_ok=True)
    scheduler.save_pretrained(scheduler_dir)

    model_index = {
        "_class_name": "MiniT2ITextToImagePipeline",
        "_diffusers_version": "0.32.0",
        "scheduler": ["diffusers", "MiniT2IFlowMatchScheduler"],
        "text_encoder": ["transformers", "T5EncoderModel"],
        "tokenizer": ["transformers", "AutoTokenizer"],
        "transformer": ["diffusers", "MiniT2IMMJiTModel"],
    }
    (out_dir / "model_index.json").write_text(json.dumps(model_index, indent=2) + "\n", encoding="utf-8")

    metadata = {
        "task": "text-to-image",
        "variant": args.variant_name,
        "text_encoder": transformer.mmjit_config.llm,
        "source_checkpoint": str(Path(args.checkpoint).resolve()),
    }
    (out_dir / "conversion_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(f"Saved diffusers bundle to {out_dir}")


if __name__ == "__main__":
    main()

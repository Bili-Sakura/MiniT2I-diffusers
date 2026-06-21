#!/usr/bin/env python3
"""Sample images from a MiniT2I diffusers pipeline."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

LIB_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = LIB_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from diffusers.pipelines.minit2i.pipeline_minit2i import MiniT2ITextToImagePipeline


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Sample images from MiniT2I.")
    parser.add_argument("--model", type=str, required=True, help="Hub id or local diffusers folder")
    parser.add_argument("--output", type=str, required=True, help="Output PNG path")
    parser.add_argument("--prompt", type=str, required=True, help="Text prompt")
    parser.add_argument("--model-type", type=str, default="b16", help="Model variant alias")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--cfg", type=float, default=6.0, help="Classifier-free guidance scale")
    parser.add_argument("--dtype", choices=["float32", "bfloat16"], default="bfloat16")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float32
    pipe = MiniT2ITextToImagePipeline.from_pretrained(
        args.model,
        model_type=args.model_type,
        torch_dtype=dtype,
    )
    pipe.to(args.device)
    generator = torch.Generator(device=args.device).manual_seed(args.seed)
    image = pipe(
        args.prompt,
        model_type=args.model_type,
        repo_id_or_path=args.model,
        guidance_scale=args.cfg,
        num_inference_steps=args.steps,
        generator=generator,
    ).images[0]
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path)
    print(f"Saved {output_path}")


if __name__ == "__main__":
    main()

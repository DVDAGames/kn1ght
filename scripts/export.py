#!/usr/bin/env python3
"""
Export a kn1ght checkpoint to HuggingFace-ready artifacts.

Produces:
  dist/kn1ght-bullet/
    config.json              ← model architecture (gpt2-compatible)
    generation_config.json   ← default generation parameters
    tokenizer.json           ← BPE tokenizer (transformers.js-compatible)
    tokenizer_config.json    ← HF tokenizer metadata
    onnx/
      model.onnx             ← full-precision ONNX for transformers.js
      model_quantized.onnx   ← int8 quantized (smaller, ~4× faster on CPU)

Usage:
  uv run python scripts/export.py
  uv run python scripts/export.py --checkpoint .data/models/kn1ght-dpo/ckpt_000300.pt
  uv run python scripts/export.py --no-quantize
"""

import argparse
import json
import shutil
import sys
from pathlib import Path

import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
from train import ChessGPT, ModelConfig, TOKENIZER_PATH

CHECKPOINT  = ROOT / ".data/models/kn1ght-dpo/ckpt_000300.pt"
OUTPUT_DIR  = ROOT / "dist/kn1ght-bullet"
ONNX_DIR    = OUTPUT_DIR / "onnx"
OPSET       = 17


# ── Thin wrapper that returns only logits (ONNX can't handle optional outputs) ─

class _LogitsOnly(nn.Module):
    def __init__(self, model: ChessGPT):
        super().__init__()
        self.model = model

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        logits, _ = self.model(input_ids)
        return logits


def load_model(checkpoint: Path, device: torch.device) -> tuple[ChessGPT, ModelConfig]:
    ckpt = torch.load(checkpoint, map_location=device, weights_only=False)
    cfg  = ModelConfig(**ckpt["model_cfg"])
    model = ChessGPT(cfg)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model, cfg


def export_onnx(model: ChessGPT, cfg: ModelConfig, path: Path) -> None:
    wrapper = _LogitsOnly(model)
    dummy = torch.zeros(1, 8, dtype=torch.long)  # (batch=1, seq=8)

    torch.onnx.export(
        wrapper,
        (dummy,),
        str(path),
        input_names=["input_ids"],
        output_names=["logits"],
        dynamic_axes={
            "input_ids": {0: "batch_size", 1: "sequence_length"},
            "logits":    {0: "batch_size", 1: "sequence_length"},
        },
        opset_version=OPSET,
        do_constant_folding=True,
        dynamo=False,
    )
    print(f"  Exported ONNX → {path.name}  ({path.stat().st_size / 1e6:.1f} MB)")


def quantize_onnx(fp32_path: Path, q_path: Path) -> None:
    try:
        from onnxruntime.quantization import quantize_dynamic, QuantType
    except ImportError:
        print("  Skipping quantization — onnxruntime not installed")
        return

    quantize_dynamic(
        str(fp32_path),
        str(q_path),
        weight_type=QuantType.QInt8,
    )
    print(f"  Quantized  ONNX → {q_path.name}  ({q_path.stat().st_size / 1e6:.1f} MB)")


def write_config(cfg: ModelConfig, output_dir: Path) -> None:
    # Use gpt2 model_type so transformers.js recognises the generation loop.
    # Field names follow the HF GPT-2 config schema.
    config = {
        "model_type":                    "gpt2",
        "architectures":                 ["GPT2LMHeadModel"],
        "_name_or_path":                 "InterwebAlchemy/kn1ght-bullet",
        "vocab_size":                    cfg.vocab_size,
        "n_embd":                        cfg.n_embd,
        "n_head":                        cfg.n_head,
        "n_layer":                       cfg.n_layer,
        "n_positions":                   cfg.block_size,
        "n_inner":                       4 * cfg.n_embd,
        "activation_function":           "gelu_new",
        "resid_pdrop":                   0.0,
        "embd_pdrop":                    0.0,
        "attn_pdrop":                    0.0,
        "layer_norm_epsilon":            1e-5,
        "initializer_range":             0.02,
        "scale_attn_weights":            True,
        "reorder_and_upcast_attn":       False,
        "scale_attn_by_inverse_layer_idx": False,
        "use_cache":                     True,
        "bos_token_id":                  0,
        "eos_token_id":                  1,
        "pad_token_id":                  3,
    }
    path = output_dir / "config.json"
    path.write_text(json.dumps(config, indent=2))
    print(f"  Wrote config.json")


def write_generation_config(output_dir: Path) -> None:
    gen_cfg = {
        "_from_model_config": True,
        "bos_token_id":       0,
        "eos_token_id":       1,
        "pad_token_id":       3,
        "max_new_tokens":     256,
        "do_sample":          True,
        "temperature":        0.8,
        "top_k":              40,
    }
    path = output_dir / "generation_config.json"
    path.write_text(json.dumps(gen_cfg, indent=2))
    print(f"  Wrote generation_config.json")


def write_tokenizer(output_dir: Path) -> None:
    # tokenizer.json is already in transformers.js-compatible HF format
    shutil.copy(TOKENIZER_PATH, output_dir / "tokenizer.json")
    print(f"  Copied  tokenizer.json")

    tokenizer_config = {
        "tokenizer_class":  "PreTrainedTokenizerFast",
        "bos_token":        "[g_start]",
        "eos_token":        "[g_end]",
        "unk_token":        "[unknown]",
        "pad_token":        "[pad]",
        "model_max_length": 256,
        "clean_up_tokenization_spaces": False,
    }
    path = output_dir / "tokenizer_config.json"
    path.write_text(json.dumps(tokenizer_config, indent=2))
    print(f"  Wrote  tokenizer_config.json")


def main():
    parser = argparse.ArgumentParser(description="Export kn1ght checkpoint for HuggingFace")
    parser.add_argument("--checkpoint", type=Path, default=CHECKPOINT)
    parser.add_argument("--output",     type=Path, default=OUTPUT_DIR)
    parser.add_argument("--no-quantize", action="store_true")
    args = parser.parse_args()

    args.output.mkdir(parents=True, exist_ok=True)
    ONNX_DIR_ = args.output / "onnx"
    ONNX_DIR_.mkdir(exist_ok=True)

    print(f"Loading checkpoint: {args.checkpoint}")
    model, cfg = load_model(args.checkpoint, torch.device("cpu"))
    print(f"  {model.num_params / 1e6:.1f}M params  |  "
          f"{cfg.n_layer}L {cfg.n_head}H {cfg.n_embd}d  block_size={cfg.block_size}")

    print("\nExporting ONNX...")
    fp32_path = ONNX_DIR_ / "model.onnx"
    export_onnx(model, cfg, fp32_path)

    if not args.no_quantize:
        print("\nQuantizing...")
        quantize_onnx(fp32_path, ONNX_DIR_ / "model_quantized.onnx")

    print("\nWriting config files...")
    write_config(cfg, args.output)
    write_generation_config(args.output)
    write_tokenizer(args.output)

    print(f"\nDone. Artifacts at: {args.output}")
    print("\nTo publish:")
    print("  huggingface-cli login")
    print(f"  huggingface-cli upload InterwebAlchemy/kn1ght-bullet {args.output} .")


if __name__ == "__main__":
    main()

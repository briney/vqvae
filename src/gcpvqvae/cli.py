"""Command line interface entry points for the GCP-VQVAE toolkit."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import yaml

from gcpvqvae.data.protein_io import write_protein_file
from gcpvqvae.models.gcpvqvae_model import GCPVQVAE
from gcpvqvae.system.eval import evaluate_from_config
from gcpvqvae.system.train import train_from_config
from gcpvqvae.utils.seed import seed_everything


def _load_model_from_checkpoint(ckpt_path: str, device: str) -> tuple[GCPVQVAE, dict]:
    """Loads a model and config from a checkpoint file for inference."""
    ckpt = torch.load(ckpt_path, map_location=device)
    cfg = ckpt['config']
    model = GCPVQVAE(cfg['model'])
    model.load_state_dict(ckpt['model_state_dict'])
    model.to(device)
    model.eval()
    return model, cfg


def encode_cmd() -> None:
    """Encode protein backbones to VQ tokens."""
    parser = argparse.ArgumentParser(description="Encode a protein chain to discrete tokens.")
    parser.add_argument("input_path", type=Path, help="Path to the input PDB or mmCIF file.")
    parser.add_argument("--chain", required=True, help="The chain ID to process.")
    parser.add_argument("--out", type=Path, required=True, help="Path to the output .npz file.")
    parser.add_argument("--checkpoint", type=Path, required=True, help="Path to the model checkpoint.")
    parser.add_argument("--device", default="cpu", help="Device to run the model on (e.g., 'cuda:0').")
    args = parser.parse_args()

    model, cfg = _load_model_from_checkpoint(str(args.checkpoint), args.device)

    print(f"Encoding {args.input_path} chain {args.chain}...")
    max_len = cfg.get('data', {}).get('length_cap', 2048)
    result = model.encode(str(args.input_path), args.chain, max_length=max_len)

    if result is None:
        print(f"Error: Could not process {args.input_path} chain {args.chain}.", file=sys.stderr)
        sys.exit(1)

    np.savez(
        args.out,
        tokens=result['tokens'],
        mask=result['mask'],
        pose_header_R=result['pose_header'][0],
        pose_header_t=result['pose_header'][1],
        chain_id=result['chain_id'],
        aatype=result['aatype'],
        input_format=result['input_format'],
    )
    print(f"Successfully saved tokens to {args.out}")


def decode_cmd() -> None:
    """Decode VQ tokens to backbone coordinates."""
    parser = argparse.ArgumentParser(description="Decode discrete tokens to a protein structure.")
    parser.add_argument("input_npz", type=Path, help="Path to the input .npz file.")
    parser.add_argument(
        "--out",
        type=Path,
        required=False,
        help=(
            "Path to the output file. If not specified, it will be automatically inferred "
            "from the input file name and format (e.g., input.pdb -> input_decoded.pdb)."
        ),
    )
    parser.add_argument("--checkpoint", type=Path, required=True, help="Path to the model checkpoint.")
    parser.add_argument("--device", default="cpu", help="Device to run the model on (e.g., 'cuda:0').")
    args = parser.parse_args()

    model, _ = _load_model_from_checkpoint(str(args.checkpoint), args.device)

    print(f"Decoding {args.input_npz}...")
    data = np.load(args.input_npz, allow_pickle=True)
    tokens = data['tokens']
    pose_header = (data['pose_header_R'], data['pose_header_t'])

    # Determine output path
    out_path = args.out
    if out_path is None:
        input_format = str(data.get('input_format', 'cif')) # Default to cif for old files
        out_path = args.input_npz.with_name(f"{args.input_npz.stem}_decoded.{input_format}")

    result = model.decode(tokens, pose_header=pose_header)

    write_protein_file(
        coords=result['coords'],
        mask=data['mask'],
        aatype=str(data['aatype']),
        chain_id=str(data['chain_id']),
        path=str(out_path),
    )
    print(f"Successfully saved decoded structure to {out_path}")


def train_cmd() -> None:
    """Train the model from a configuration file."""
    parser = argparse.ArgumentParser(description="Train the GCP-VQVAE model.")
    parser.add_argument("--config", type=Path, required=True, help="Path to the training config YAML file.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility.")
    args = parser.parse_args()

    seed_everything(args.seed)
    train_from_config(str(args.config))


def eval_cmd() -> None:
    """Evaluate a trained model checkpoint."""
    parser = argparse.ArgumentParser(description="Evaluate a trained GCP-VQVAE model.")
    parser.add_argument("--config", type=Path, required=True, help="Path to the evaluation config YAML file.")
    parser.add_argument("--checkpoint", type=Path, required=True, help="Path to the model checkpoint to evaluate.")
    args = parser.parse_args()

    evaluate_from_config(str(args.config), str(args.checkpoint))

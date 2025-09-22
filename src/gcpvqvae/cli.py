"""Command line interface entry points for the GCP-VQVAE toolkit."""

from __future__ import annotations

import sys
from pathlib import Path

import click
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


@click.group(name="gpcvq")
def main() -> None:
    """
    A command-line toolkit for training and using GCP-VQVAE models
    for protein backbone tokenization.
    """
    pass


@main.command()
@click.argument("input_path", type=click.Path(exists=True, path_type=Path))
@click.option("--chain", required=True, help="The chain ID to process.")
@click.option("--out", type=click.Path(path_type=Path), required=True, help="Path to the output .npz file.")
@click.option("--checkpoint", type=click.Path(exists=True, path_type=Path), required=True, help="Path to the model checkpoint.")
@click.option("--device", default="cpu", help="Device to run the model on (e.g., 'cuda:0').")
def encode(input_path, chain, out, checkpoint, device) -> None:
    """Encode a protein backbone to VQ tokens."""
    model, cfg = _load_model_from_checkpoint(str(checkpoint), device)

    click.echo(f"Encoding {input_path} chain {chain}...")
    max_len = cfg.get('data', {}).get('length_cap', 2048)
    result = model.encode(str(input_path), chain, max_length=max_len)

    if result is None:
        click.echo(f"Error: Could not process {input_path} chain {chain}.", err=True)
        sys.exit(1)

    np.savez(
        out,
        tokens=result['tokens'],
        mask=result['mask'],
        pose_header_R=result['pose_header'][0],
        pose_header_t=result['pose_header'][1],
        chain_id=result['chain_id'],
        aatype=result['aatype'],
        input_format=result['input_format'],
    )
    click.echo(f"Successfully saved tokens to {out}")


@main.command()
@click.argument("input_npz", type=click.Path(exists=True, path_type=Path))
@click.option(
    "--out",
    type=click.Path(path_type=Path),
    required=False,
    help=(
        "Path to the output file. If not specified, it will be automatically inferred "
        "from the input file name and format (e.g., input.pdb -> input_decoded.pdb)."
    ),
)
@click.option("--checkpoint", type=click.Path(exists=True, path_type=Path), required=True, help="Path to the model checkpoint.")
@click.option("--device", default="cpu", help="Device to run the model on (e.g., 'cuda:0').")
def decode(input_npz, out, checkpoint, device) -> None:
    """Decode VQ tokens to backbone coordinates."""
    model, _ = _load_model_from_checkpoint(str(checkpoint), device)

    click.echo(f"Decoding {input_npz}...")
    data = np.load(input_npz, allow_pickle=True)
    tokens = data['tokens']
    pose_header = (data['pose_header_R'], data['pose_header_t'])

    # Determine output path
    out_path = out
    if out_path is None:
        input_format = str(data.get('input_format', 'cif')) # Default to cif for old files
        out_path = input_npz.with_name(f"{input_npz.stem}_decoded.{input_format}")

    result = model.decode(tokens, pose_header=pose_header)

    write_protein_file(
        coords=result['coords'],
        mask=data['mask'],
        aatype=str(data['aatype']),
        chain_id=str(data['chain_id']),
        path=str(out_path),
    )
    click.echo(f"Successfully saved decoded structure to {out_path}")


@main.command()
@click.option("--config", type=click.Path(exists=True, path_type=Path), required=True, help="Path to the training config YAML file.")
@click.option("--seed", type=int, default=42, help="Random seed for reproducibility.")
def train(config, seed) -> None:
    """Train the model from a configuration file."""
    seed_everything(seed)
    train_from_config(str(config))


@main.command()
@click.option("--config", type=click.Path(exists=True, path_type=Path), required=True, help="Path to the evaluation config YAML file.")
@click.option("--checkpoint", type=click.Path(exists=True, path_type=Path), required=True, help="Path to the model checkpoint to evaluate.")
def eval(config, checkpoint) -> None:
    """Evaluate a trained model checkpoint."""
    evaluate_from_config(str(config), str(checkpoint))

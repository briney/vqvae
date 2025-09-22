"""Evaluation utilities for trained checkpoints."""

from __future__ import annotations

import torch
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm

from gcpvqvae.data.dataset import BackboneDataset, collate_backbones
from gcpvqvae.geometry.metrics import codebook_perplexity, rmsd
from gcpvqvae.models.gcpvqvae_model import GCPVQVAE


class Evaluator:
    """Orchestrates the evaluation process."""
    def __init__(self, model, config: dict, device: str):
        self.model = model
        self.config = config
        self.device = device

        # Dataloader
        dataset = BackboneDataset(**config['data'])
        self.dataloader = DataLoader(
            dataset,
            batch_size=config['train']['batch_size'],
            shuffle=False,
            num_workers=config['data']['num_workers'],
            collate_fn=collate_backbones,
        )

    @torch.no_grad()
    def evaluate(self):
        """Main evaluation loop."""
        self.model.eval()

        total_rmsd = 0.0
        all_indices = []
        num_batches = 0

        for batch in tqdm(self.dataloader, desc="Evaluating"):
            if batch is None: continue
            num_batches += 1

            batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}

            # Forward pass to get predictions
            z_enc = self.model.gcp_encoder(batch)
            h_lat = self.model.enc_transformer(z_enc, mask=batch['mask'])
            vq_out = self.model.vq(h_lat)
            z_q = vq_out['z_q']
            h_dec = self.model.dec_transformer(z_q, mask=batch['mask'])
            pred_coords = self.model.decoder_head(h_dec)

            # Compute metrics
            batch_rmsd = rmsd(pred_coords, batch['coords'], batch['mask'])
            total_rmsd += batch_rmsd.item()
            all_indices.append(vq_out['indices'])

        avg_rmsd = total_rmsd / num_batches

        all_indices = torch.cat(all_indices, dim=0)
        perplexity = codebook_perplexity(all_indices)

        print("\n--- Evaluation Results ---")
        print(f"Average RMSD: {avg_rmsd:.4f} Å")
        print(f"Codebook Perplexity: {perplexity.item():.4f}")
        print("--------------------------")

        return {"rmsd": avg_rmsd, "perplexity": perplexity.item()}


def evaluate_from_config(config_path: str, checkpoint_path: str):
    """Load config and checkpoint, then run evaluation."""
    device = "cuda" if torch.cuda.is_available() else "cpu"

    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # Load model from checkpoint
    ckpt = torch.load(checkpoint_path, map_location=device)
    model = GCPVQVAE(ckpt['config'])
    model.load_state_dict(ckpt['model_state_dict'])
    model.to(device)

    evaluator = Evaluator(model, config, device)
    evaluator.evaluate()

"""Evaluation utilities for trained checkpoints."""

from __future__ import annotations

import torch
import yaml
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt

from gcpvqvae.data.dataset import BackboneDataset, collate_backbones
from gcpvqvae.geometry.metrics import codebook_perplexity, rmsd, tm_score, gdt_ts
from gcpvqvae.models.gcpvqvae_model import GCPVQVAE


class Evaluator:
    """Orchestrates the evaluation process."""
    def __init__(self, model, config: dict, device: str):
        self.model = model
        self.config = config
        self.device = device
        self.output_dir = self.config.get("output_dir", ".")

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

        all_rmsds, all_tms, all_gdts, all_lengths = [], [], [], []
        all_indices = []

        for batch in tqdm(self.dataloader, desc="Evaluating"):
            if batch is None: continue

            batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}

            # Forward pass to get predictions
            z_enc = self.model.gcp_encoder(batch)
            h_lat = self.model.enc_transformer(z_enc, mask=batch['mask'])
            vq_out = self.model.vq(h_lat)
            z_q = vq_out['z_q']
            h_dec = self.model.dec_transformer(z_q, mask=batch['mask'])
            pred_coords = self.model.decoder_head(h_dec)

            all_indices.append(vq_out['indices'].cpu())

            # Loop over the batch to compute per-protein metrics
            for i in range(pred_coords.shape[0]):
                L = int(batch['mask'][i].sum().item())
                if L == 0: continue

                # Unpad the coordinates and mask
                p_coords = pred_coords[i, :L]
                t_coords = batch['coords'][i, :L]
                mask = batch['mask'][i, :L]

                # Pass as a batch of 1 to the looped rmsd implementation
                batch_rmsd = rmsd(p_coords.unsqueeze(0), t_coords.unsqueeze(0), mask.unsqueeze(0))
                all_rmsds.append(batch_rmsd.item())

                # TM-score and GDT-TS expect unbatched inputs
                tm = tm_score(p_coords, t_coords, mask)
                all_tms.append(tm.item())

                gdt = gdt_ts(p_coords, t_coords, mask)
                all_gdts.append(gdt.item())

                all_lengths.append(L)

        # --- Aggregate and Report Metrics ---

        # Codebook metrics
        all_indices = torch.cat(all_indices, dim=0)
        perplexity = codebook_perplexity(all_indices)

        num_codes = self.model.vq.K
        active_codes = len(torch.unique(all_indices))
        codebook_utilization = active_codes / num_codes if num_codes > 0 else 0

        # Structure metrics
        metrics = {
            "RMSD": np.array(all_rmsds),
            "TM-score": np.array(all_tms),
            "GDT-TS": np.array(all_gdts)
        }

        print("\n--- Evaluation Results ---")
        for name, values in metrics.items():
            if len(values) > 0:
                print(f"{name}:")
                print(f"  Mean: {np.mean(values):.4f}")
                print(f"  Std:  {np.std(values):.4f}")
                print(f"  Median: {np.median(values):.4f}")
            else:
                print(f"{name}: No data")

        print("\nCodebook:")
        print(f"  Perplexity: {perplexity.item():.4f}")
        print(f"  Utilization: {codebook_utilization:.4f} ({active_codes}/{num_codes})")
        print("--------------------------\n")

        # RMSD vs. Length plot
        if all_lengths and all_rmsds:
            plt.figure(figsize=(8, 6))
            plt.scatter(all_lengths, all_rmsds, alpha=0.5)
            plt.xlabel("Protein Length (residues)")
            plt.ylabel("RMSD (Å)")
            plt.title("RMSD vs. Protein Length")

            # Fit and plot trendline
            if len(all_lengths) > 1:
                try:
                    m, b = np.polyfit(all_lengths, all_rmsds, 1)
                    plt.plot(np.array(all_lengths), m * np.array(all_lengths) + b, color='red')
                    print(f"RMSD vs. Length trend: slope={m:.4e} Å/residue, intercept={b:.4f} Å")
                except (np.linalg.LinAlgError, TypeError):
                    print("Could not fit trendline for RMSD vs. Length.")

            plot_path = f"{self.output_dir}/rmsd_vs_length.png"
            plt.savefig(plot_path)
            print(f"Saved RMSD vs. Length plot to {plot_path}")
        else:
            print("Not enough data to generate RMSD vs. Length plot.")

        return {
            "rmsd": np.mean(all_rmsds) if all_rmsds else 0,
            "tm_score": np.mean(all_tms) if all_tms else 0,
            "gdt_ts": np.mean(all_gdts) if all_gdts else 0,
            "perplexity": perplexity.item(),
            "codebook_utilization": codebook_utilization,
        }


def evaluate_from_config(config_path: str, checkpoint_path: str):
    """Load config and checkpoint, then run evaluation."""
    device = "cuda" if torch.cuda.is_available() else "cpu"

    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # Load model from checkpoint
    ckpt = torch.load(checkpoint_path, map_location=device)
    # The config inside the checkpoint is what we should use for the model
    model_cfg = ckpt['config']['model']
    model = GCPVQVAE(model_cfg)
    model.load_state_dict(ckpt['model_state_dict'])
    model.to(device)

    evaluator = Evaluator(model, config, device)
    evaluator.evaluate()
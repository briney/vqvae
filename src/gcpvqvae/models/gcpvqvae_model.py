"""
The main GCP-VQVAE model, which assembles the encoder, quantizer, and decoder.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
from torch import nn

from gcpvqvae.data.featurize import featurize_backbone
from gcpvqvae.data.mmcif import load_mmcif, write_mmcif, ParsedMmcif
from gcpvqvae.models.decoder import Rigid6DHead
from gcpvqvae.models.gcpnet import GCPNetEncoder
from gcpvqvae.models.losses import ReconstructionLoss
from gcpvqvae.models.transformer import Transformer
from gcpvqvae.models.vq import VectorQuantizer


def unpad_and_rebatch(tensor: torch.Tensor, batch_idx: torch.Tensor):
    """Un-batches a flattened tensor back into a padded batch."""
    # Get a list of tensors, one for each sample in the batch
    batch_size = batch_idx.max().item() + 1
    tensors = [tensor[batch_idx == i] for i in range(batch_size)]
    # Pad the list of tensors to create a batched tensor
    return torch.nn.utils.rnn.pad_sequence(tensors, batch_first=True)


class GCPVQVAE(nn.Module):
    """
    The end-to-end GCP-VQVAE model.
    """
    def __init__(self, cfg: dict[str, Any]):
        super().__init__()
        self.cfg = cfg

        # Encoder components
        self.gcp_encoder = GCPNetEncoder(**cfg['gcp'])
        self.enc_transformer = Transformer(**cfg['enc'])

        # Vector quantizer
        self.vq = VectorQuantizer(**cfg['vq'])

        # Decoder components
        self.dec_transformer = Transformer(**cfg['dec'], project_out=False)
        self.decoder_head = Rigid6DHead(**cfg['decoder_head'])

        # Loss function
        self.rec_loss = ReconstructionLoss(**cfg['loss'])

    def forward(self, batch: dict[str, torch.Tensor]):
        """Full forward pass for training."""
        # Encoder pass
        z_enc_flat = self.gcp_encoder(batch)
        z_enc = unpad_and_rebatch(z_enc_flat, batch['batch_idx'])
        h_lat = self.enc_transformer(z_enc, mask=batch['mask'])

        # Vector quantization
        vq_out = self.vq(h_lat)
        z_q = vq_out['z_q']

        # Decoder pass
        h_dec = self.dec_transformer(z_q, mask=batch['mask'])
        pred_coords = self.decoder_head(h_dec)

        # Loss calculation
        rec_losses = self.rec_loss(pred_coords, batch['coords'], batch['mask'])

        total_loss = (
            rec_losses['loss_rec'] +
            vq_out['loss_code'] +
            vq_out['loss_commit'] +
            vq_out['loss_orth']
        )

        return {
            "loss": total_loss,
            **rec_losses,
            "loss_code": vq_out['loss_code'],
            "loss_commit": vq_out['loss_commit'],
            "loss_orth": vq_out['loss_orth'],
            "indices": vq_out['indices'],
        }

    @torch.no_grad()
    def encode(self, mmcif_path: str, chain_id: str, max_length: int = 2048) -> dict[str, Any] | None:
        """Encode an mmCIF file to discrete tokens."""
        self.eval()

        # 1. Load and featurize data
        parsed_mmcif = load_mmcif(mmcif_path, chain_id, max_length=max_length)
        if parsed_mmcif is None:
            return None

        features = featurize_backbone(parsed_mmcif)

        # The GNN encoder expects a batch dictionary, but for a single item,
        # we can pass the features directly.
        features_dict = features.__dict__
        mask = parsed_mmcif.mask.unsqueeze(0) # Transformer needs batch dim

        # 2. Run encoder and VQ
        z_enc = self.gcp_encoder(features_dict)
        h_lat = self.enc_transformer(z_enc.unsqueeze(0), mask=mask)
        vq_out = self.vq(h_lat)

        return {
            "tokens": vq_out['indices'].squeeze(0).cpu().numpy(),
            "length": parsed_mmcif.coords.shape[0],
            "mask": parsed_mmcif.mask.cpu().numpy(),
            "pose_header": (
                parsed_mmcif.pose_header[0].cpu().numpy(),
                parsed_mmcif.pose_header[1].cpu().numpy(),
            ),
            "chain_id": parsed_mmcif.chain_id,
            "aatype": parsed_mmcif.aatype,
        }

    @torch.no_grad()
    def decode(self, tokens: np.ndarray, pose_header=None) -> dict[str, Any]:
        """Decode a sequence of tokens back to coordinates."""
        self.eval()

        tokens_tensor = torch.from_numpy(tokens).long().to(self.vq.codebook.weight.device)
        tokens_tensor = tokens_tensor.unsqueeze(0) # Add batch dimension

        # 1. De-quantize tokens
        z_q = self.vq.codebook(tokens_tensor)

        # 2. Run decoder
        h_dec = self.dec_transformer(z_q)
        pred_coords = self.decoder_head(h_dec).squeeze(0) # Remove batch dim

        # 3. (Optional) Decentralize to restore original pose
        if pose_header is not None:
            R, t = pose_header
            R = torch.from_numpy(R).to(pred_coords.device)
            t = torch.from_numpy(t).to(pred_coords.device)
            pred_coords = torch.einsum('ij,kjl->kil', R, pred_coords) + t.unsqueeze(1)

        # 4. (Optional) Write to mmCIF file string
        # For now, just return the coords. The CLI will handle writing.
        return {
            "coords": pred_coords.cpu().numpy(),
        }

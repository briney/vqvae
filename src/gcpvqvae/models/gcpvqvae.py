"""High-level end-to-end module for the GCP-VQVAE architecture."""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, Mapping, Optional, Tuple

import numpy as np
import torch
from torch import Tensor, nn

from gcpvqvae.data.batch import EdgeStorage, ProteinBatch, protein_batch_from_graph_dict
from gcpvqvae.data.featurize import featurize_backbone
from gcpvqvae.data.mmcif import PAD_INDEX, BackboneRecord, load_mmcif
from gcpvqvae.models.decoder import RotationDecoder
from gcpvqvae.models.gcpnet import GCPNetConfig, GCPNetEncoder
from gcpvqvae.models.losses import reconstruction_loss
from gcpvqvae.models.transformer import GCPTokensTransformer, TransformerConfig
from gcpvqvae.models.vq import VectorQuantizer
from gcpvqvae.utils.checkpoint import load_checkpoint


@dataclass
class DataPipelineConfig:
    """Configuration governing single-chain preprocessing."""

    length_cap: int = 2048
    knn: int = 16


@dataclass
class VectorQuantizerConfig:
    """Configuration for the latent codebook."""

    num_codes: int = 4096
    dim: int = 256
    beta: float = 0.25
    decay: float = 0.99
    epsilon: float = 1e-5
    kmeans_iters: int = 10
    rotation_trick: bool = True
    orthogonal_reg_weight: float = 0.0
    orthogonal_reg_max_codes: int = 512


@dataclass
class LatentAdapterConfig:
    """Optional linear adapter bridging the GCP encoder and Transformer."""

    enabled: bool = False
    output_dim: Optional[int] = None
    bias: bool = False


@dataclass
class RotationHeadConfig:
    """Parameters for the rigid 6D rotation decoder head."""

    input_dim: Optional[int] = None
    translation_scale: float = 1.0
    template: Optional[Tensor] = None


@dataclass
class GCPVQVAEConfig:
    """Top-level configuration for the full GCP-VQVAE model."""

    gcp: GCPNetConfig = field(default_factory=GCPNetConfig)
    encoder: TransformerConfig = field(
        default_factory=lambda: TransformerConfig(input_dim=256, output_dim=256)
    )
    decoder: TransformerConfig = field(
        default_factory=lambda: TransformerConfig(
            input_dim=256, num_layers=16, num_heads=16, num_kv_heads=1
        )
    )
    vq: VectorQuantizerConfig = field(default_factory=VectorQuantizerConfig)
    adapter: LatentAdapterConfig = field(default_factory=LatentAdapterConfig)
    rotation: RotationHeadConfig = field(default_factory=RotationHeadConfig)
    data: DataPipelineConfig = field(default_factory=DataPipelineConfig)

    def __post_init__(self) -> None:
        # Ensure dimensional consistency between the sub-modules.
        if self.adapter.enabled:
            target_dim = self.adapter.output_dim or self.vq.dim
            if target_dim is None:
                raise ValueError(
                    "Latent adapter requires either 'output_dim' or 'vq.dim' to be set"
                )
            self.adapter.output_dim = target_dim
            self.encoder.input_dim = target_dim
        else:
            self.encoder.input_dim = self.gcp.latent_dim
        self.encoder.output_dim = self.vq.dim
        self.decoder.input_dim = self.vq.dim
        if self.decoder.output_dim is None:
            # The rotation head consumes the decoder's model dimension when the
            # output projection is omitted.
            decoder_out = self.decoder.model_dim
        else:
            decoder_out = self.decoder.output_dim
        if self.rotation.input_dim is None:
            self.rotation.input_dim = decoder_out


class GCPVQVAE(nn.Module):
    """High-level module exposing encode/decode utilities and training forward."""

    def __init__(self, config: Optional[GCPVQVAEConfig] = None) -> None:
        super().__init__()
        self.config = config or GCPVQVAEConfig()

        self.encoder_gcp = GCPNetEncoder(self.config.gcp)
        self._initialize_gcp_encoder()
        self.latent_adapter: Optional[nn.Module]
        if self.config.adapter.enabled:
            self.latent_adapter = nn.Linear(
                self.config.gcp.latent_dim,
                self.config.adapter.output_dim or self.config.vq.dim,
                bias=self.config.adapter.bias,
            )
        else:
            self.latent_adapter = None
        self.encoder_transformer = GCPTokensTransformer(self.config.encoder)
        self.vq = VectorQuantizer(
            self.config.vq.num_codes,
            self.config.vq.dim,
            beta=self.config.vq.beta,
            decay=self.config.vq.decay,
            epsilon=self.config.vq.epsilon,
            kmeans_iters=self.config.vq.kmeans_iters,
            rotation_trick=self.config.vq.rotation_trick,
            orthogonal_reg_weight=self.config.vq.orthogonal_reg_weight,
            orthogonal_reg_max_codes=self.config.vq.orthogonal_reg_max_codes,
        )
        self.decoder_transformer = GCPTokensTransformer(self.config.decoder)
        self.rotation_decoder = RotationDecoder(
            self.config.rotation.input_dim,
            translation_scale=self.config.rotation.translation_scale,
            template=self.config.rotation.template,
        )

    # ------------------------------------------------------------------ utils
    def _initialize_gcp_encoder(self) -> None:
        init_mode = getattr(self.config.gcp, "init", "random") or "random"
        mode = init_mode.lower()
        if mode == "random":
            return
        if mode != "pretrained":
            raise ValueError(f"Unsupported GCPNet initialisation mode '{init_mode}'")

        checkpoint = getattr(self.config.gcp, "init_checkpoint", None)
        if not checkpoint:
            raise ValueError(
                "GCPNet pretrained initialisation requires 'checkpoint' to be set"
            )

        state = load_checkpoint(checkpoint, map_location="cpu")
        state_dict = self._extract_gcp_state_dict(state)
        if state_dict is None:
            raise ValueError(
                "Checkpoint does not contain GCPNet weights – expected a mapping under "
                "'gcp_state', 'model_state', 'state_dict', or a raw state dict"
            )

        missing, unexpected = self.encoder_gcp.load_state_dict(
            state_dict, strict=getattr(self.config.gcp, "strict_init", True)
        )
        if missing or unexpected:
            warnings.warn(
                "Loaded GCPNet weights with missing keys %s and unexpected keys %s"
                % (missing, unexpected),
                UserWarning,
            )

    @staticmethod
    def _extract_gcp_state_dict(state: Mapping[str, Any]) -> Optional[Dict[str, Tensor]]:
        if not isinstance(state, Mapping):
            return None

        candidate_keys = (
            "gcp_state",
            "gcpnet_state",
            "encoder_state",
            "encoder_state_dict",
            "model_state",
            "state_dict",
        )
        for key in candidate_keys:
            if key in state:
                candidate = state[key]
                if isinstance(candidate, Mapping):
                    return {k: v for k, v in candidate.items() if isinstance(k, str)}

        if all(isinstance(k, str) and isinstance(v, torch.Tensor) for k, v in state.items()):
            return {k: v for k, v in state.items() if isinstance(v, torch.Tensor)}

        return None

    def _device(self) -> torch.device:
        return next(self.parameters()).device

    def _dtype(self) -> torch.dtype:
        return next(self.parameters()).dtype

    def _flatten_batch(self, tensor: Tensor) -> Tensor:
        if tensor.ndim < 2:
            raise ValueError("Expected tensor with batch and length dimensions")
        batch, length = tensor.shape[:2]
        return tensor.reshape(batch * length, *tensor.shape[2:])

    def _reshape_batch(self, tensor: Tensor, batch: int, length: int) -> Tensor:
        return tensor.reshape(batch, length, *tensor.shape[1:])

    def _project_embeddings(self, embeddings: Tensor) -> Tensor:
        if self.latent_adapter is None:
            return embeddings
        return self.latent_adapter(embeddings)

    # ----------------------------------------------------------------- forward
    def forward(self, batch: Dict[str, Tensor]) -> Dict[str, Tensor]:
        device = self._device()
        dtype = self._dtype()

        mask = batch["mask"].to(torch.bool).to(device)
        if "nan_mask" in batch:
            mask = mask & ~batch["nan_mask"].to(torch.bool).to(device)

        coords = batch["coords"].to(device=device, dtype=dtype)
        proto = protein_batch_from_graph_dict(batch)
        proto = proto.to(device=device, dtype=dtype)

        batch_size, max_len, _ = batch["node_scalars"].shape

        gcp_out = self.encoder_gcp(proto)
        flat_embeddings = gcp_out["embeddings"]
        latent_dim = flat_embeddings.shape[-1]
        padded = flat_embeddings.new_zeros((batch_size * max_len, latent_dim))
        padded.index_copy_(0, proto.valid_indices, flat_embeddings)
        embeddings = padded.reshape(batch_size, max_len, latent_dim)
        projected = self._project_embeddings(embeddings)

        enc_hidden = self.encoder_transformer(projected, mask=mask)
        quantized, indices, vq_losses = self.vq(enc_hidden, mask=mask)

        dec_hidden = self.decoder_transformer(quantized, mask=mask)
        recon_coords, final_pose = self.rotation_decoder(dec_hidden, mask=mask)

        rec_loss, rec_components = reconstruction_loss(
            recon_coords,
            coords,
            mask=mask,
            return_components=True,
        )

        total_loss = rec_loss
        total_loss = total_loss + vq_losses["commitment"]
        total_loss = total_loss + vq_losses["codebook"]
        total_loss = total_loss + vq_losses["orthogonality"]

        return {
            "gcp_embeddings": embeddings,
            "encoder_hidden": enc_hidden,
            "quantized": quantized,
            "indices": indices,
            "decoded": recon_coords,
            "mask": mask,
            "pose": final_pose,
            "vq_losses": vq_losses,
            "reconstruction": rec_loss,
            "reconstruction_components": rec_components,
            "total_loss": total_loss,
        }

    def commit_updates(self) -> None:
        """Apply any deferred updates inside the module."""

        self.vq.commit_pending_codebook()

    # ----------------------------------------------------------- encode helper
    def _build_metadata(self, record: BackboneRecord) -> Dict[str, Any]:
        return {
            "path": record.path,
            "chain_id": record.chain_id,
            "sequence": record.seq_string,
            "seq_indices": record.seq.clone().cpu(),
            "residue_ids": list(record.residue_ids),
            "residue_names": list(record.residue_names),
        }

    def _run_encoder(
        self,
        node_scalars: Tensor,
        node_vectors: Tensor,
        ca_positions: Tensor,
        edge_index: Tensor,
        edge_scalars: Tensor,
        edge_vectors: Tensor,
        edge_frames: Tensor,
        mask: Tensor,
    ) -> Tuple[Tensor, Tensor, Tensor, Dict[str, Tensor]]:
        device = self._device()
        dtype = self._dtype()

        node_scalars = node_scalars.to(device=device, dtype=dtype)
        node_vectors = node_vectors.to(device=device, dtype=dtype)
        edge_index = edge_index.to(device=device)
        edge_scalars = edge_scalars.to(device=device, dtype=dtype)
        edge_vectors = edge_vectors.to(device=device, dtype=dtype)
        edge_frames = edge_frames.to(device=device, dtype=dtype)
        mask = mask.to(device=device, dtype=torch.bool)
        ca_positions = ca_positions.to(device=device, dtype=dtype)

        proto = ProteinBatch(
            h=node_scalars,
            chi=node_vectors,
            e={
                "knn_k": EdgeStorage(
                    edge_index=edge_index,
                    scalars=edge_scalars,
                    vectors=edge_vectors,
                    frames=edge_frames,
                    batch=None,
                    name="knn_k",
                )
            },
            xi=ca_positions,
            batch=torch.zeros(node_scalars.shape[0], dtype=torch.long, device=node_scalars.device),
            ptr=torch.tensor([0, node_scalars.shape[0]], device=node_scalars.device),
            mask=mask,
        )
        proto.valid_indices = torch.arange(node_scalars.shape[0], device=node_scalars.device)
        proto.full_mask = mask.unsqueeze(0)
        proto.batch_size = 1
        proto.max_length = node_scalars.shape[0]
        proto = proto.to(device=device, dtype=dtype)
        gcp_out = self.encoder_gcp(proto)
        embeddings = gcp_out["embeddings"].unsqueeze(0)
        projected = self._project_embeddings(embeddings)
        enc_hidden = self.encoder_transformer(projected, mask=mask.unsqueeze(0))
        quantized, indices, vq_losses = self.vq(enc_hidden, mask=mask.unsqueeze(0))
        return embeddings, enc_hidden, quantized, {"indices": indices, "vq": vq_losses}

    # ------------------------------------------------------------------- encode
    @torch.no_grad()
    def encode(
        self,
        mmcif_path: str,
        *,
        chain_id: Optional[str] = None,
        length_cap: Optional[int] = None,
        k: Optional[int] = None,
    ) -> Dict[str, Any]:
        was_training = self.training
        self.eval()
        try:
            records = load_mmcif(
                mmcif_path,
                chain_id=chain_id,
                length_cap=length_cap or self.config.data.length_cap,
            )
            if not records:
                raise ValueError(f"No suitable chains found in {mmcif_path}")
            record = records[0]
            features = featurize_backbone(record, k=k or self.config.data.knn)

            mask = record.mask.clone()
            if record.nan_mask.numel():
                mask = mask & ~record.nan_mask

            embeddings, enc_hidden, quantized, extras = self._run_encoder(
                features["node_scalars"],
                features["node_vectors"],
                record.coords[:, 1, :],
                features["edge_index"],
                features["edge_scalars"],
                features["edge_vectors"],
                features["edge_frames"],
                mask,
            )

            indices = extras["indices"].squeeze(0).cpu()
            vq_losses = extras["vq"]

            result = {
                "tokens": indices.clone(),
                "length": int(record.length),
                "mask": mask.clone().cpu(),
                "pose_header": (
                    record.rotation.clone().cpu(),
                    record.translation.clone().cpu(),
                ),
                "gcp_embeddings": embeddings.squeeze(0).cpu(),
                "encoder_embeddings": enc_hidden.squeeze(0).cpu(),
                "quantized": quantized.squeeze(0).cpu(),
                "vq_losses": {k: v.detach().cpu() for k, v in vq_losses.items()},
                "metadata": self._build_metadata(record),
            }
            return result
        finally:
            self.train(was_training)

    # ------------------------------------------------------------------- decode
    @torch.no_grad()
    def decode(
        self,
        tokens: Iterable[int] | np.ndarray | Tensor,
        *,
        pose_header: Optional[Tuple[Tensor | np.ndarray, Tensor | np.ndarray]] = None,
        mask: Optional[Iterable[bool] | Tensor | np.ndarray] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        was_training = self.training
        self.eval()
        try:
            device = self._device()
            dtype = self._dtype()

            tokens_tensor = torch.as_tensor(tokens, dtype=torch.long, device=device)
            if tokens_tensor.ndim == 1:
                tokens_tensor = tokens_tensor.unsqueeze(0)
            batch, length = tokens_tensor.shape

            if mask is None:
                mask_tensor = tokens_tensor >= 0
            else:
                mask_tensor = torch.as_tensor(mask, dtype=torch.bool, device=device)
                if mask_tensor.ndim == 1:
                    mask_tensor = mask_tensor.unsqueeze(0)

            latent_dim = self.config.vq.dim
            dequant = torch.zeros((batch, length, latent_dim), device=device, dtype=dtype)
            valid = mask_tensor & (tokens_tensor >= 0)
            if valid.any():
                flat_indices = tokens_tensor[valid]
                dequant[valid] = self.vq.embedding.index_select(0, flat_indices).to(dtype)

            dec_hidden = self.decoder_transformer(dequant, mask=mask_tensor)
            coords_central, final_pose = self.rotation_decoder(
                dec_hidden, mask=mask_tensor
            )

            if pose_header is not None:
                rot, trans = pose_header
                rot_t = torch.as_tensor(rot, dtype=dtype, device=device)
                trans_t = torch.as_tensor(trans, dtype=dtype, device=device)
                if rot_t.ndim == 2:
                    rot_t = rot_t.unsqueeze(0)
                if trans_t.ndim == 1:
                    trans_t = trans_t.unsqueeze(0)
            else:
                rot_t = torch.eye(3, device=device, dtype=dtype).expand(batch, 3, 3)
                trans_t = torch.zeros((batch, 3), device=device, dtype=dtype)

            coords_global = coords_central @ rot_t.transpose(-1, -2)
            coords_global = coords_global + trans_t.unsqueeze(1).unsqueeze(2)

            mask_cpu = mask_tensor.clone().cpu()

            records_out: Optional[Any]
            if metadata is not None:
                seq_indices = metadata.get("seq_indices")
                if isinstance(seq_indices, Tensor):
                    seq_tensor = seq_indices.clone()
                elif seq_indices is not None:
                    seq_tensor = torch.tensor(seq_indices, dtype=torch.long)
                else:
                    seq_tensor = torch.full((length,), PAD_INDEX, dtype=torch.long)
                if seq_tensor.ndim == 1:
                    seq_tensor = seq_tensor.unsqueeze(0)
                if seq_tensor.shape[0] != batch:
                    seq_tensor = seq_tensor.expand(batch, -1)

                records = []
                for b in range(batch):
                    b_mask = mask_cpu[b]
                    valid_len = int(b_mask.sum().item())
                    seq_slice = seq_tensor[b, :valid_len]
                    seq_chars = metadata.get("sequence", "X" * valid_len)
                    if isinstance(seq_chars, str):
                        seq_chars_b = seq_chars[:valid_len]
                    else:
                        seq_chars_b = "".join(seq_chars)[:valid_len]

                    residue_ids = metadata.get("residue_ids", [(i + 1, "") for i in range(valid_len)])
                    residue_names = metadata.get("residue_names", ["UNK"] * valid_len)

                    record = BackboneRecord(
                        path=str(metadata.get("path", "")),
                        chain_id=str(metadata.get("chain_id", "A")),
                        coords=coords_central[b, :valid_len].cpu(),
                        mask=b_mask[:valid_len].cpu(),
                        atom_mask=torch.ones((valid_len, 3), dtype=torch.bool),
                        seq=seq_slice.cpu(),
                        seq_string=seq_chars_b,
                        residue_names=list(residue_names)[:valid_len],
                        residue_ids=list(residue_ids)[:valid_len],
                        rotation=rot_t[b].cpu(),
                        translation=trans_t[b].cpu(),
                        nan_mask=~b_mask[:valid_len].cpu(),
                    )
                    records.append(record)
                if batch == 1:
                    records_out = records[0]
                else:
                    records_out = records
            else:
                records_out = None

            result = {
                "coords": coords_global.squeeze(0).cpu() if batch == 1 else coords_global.cpu(),
                "coords_central": coords_central.squeeze(0).cpu()
                if batch == 1
                else coords_central.cpu(),
                "mask": mask_cpu.squeeze(0) if batch == 1 else mask_cpu,
                "pose": (final_pose[0].cpu(), final_pose[1].cpu()),
                "pose_header": (rot_t.cpu(), trans_t.cpu()),
                "records": records_out,
            }
            return result
        finally:
            self.train(was_training)


__all__ = [
    "GCPVQVAE",
    "GCPVQVAEConfig",
    "VectorQuantizerConfig",
    "LatentAdapterConfig",
    "RotationHeadConfig",
    "DataPipelineConfig",
]

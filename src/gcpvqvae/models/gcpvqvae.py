"""High-level end-to-end module for the GCP-VQVAE architecture."""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from importlib.resources import as_file, files
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional, Tuple

import numpy as np
import torch
from torch import Tensor, nn

from gcpvqvae.data.batch import EdgeStorage, ProteinBatch, protein_batch_from_graph_dict
from gcpvqvae.data.featurize import featurize_backbone
from gcpvqvae.data.mmcif import PAD_INDEX, BackboneRecord, load_mmcif
from gcpvqvae.models.decoder import Dim6RotStructureHead
from gcpvqvae.models.decoders import GeometricTransformerDecoder
from gcpvqvae.models.gcpnet import GCPNetConfig, GCPNetEncoder
from gcpvqvae.models.losses import reconstruction_loss
from gcpvqvae.models.transformer import GCPTokensTransformer, TransformerConfig
from gcpvqvae.models.vq import VectorQuantizer, VectorQuantizerOptions
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
    dim: int = 128
    beta: float = 0.25
    decay: float = 0.99
    epsilon: float = 1e-5
    kmeans_init: bool = True
    kmeans_iters: int = 10
    stochastic_sample_codes: bool = True
    sample_codebook_temp: float = 1.0
    rotation_trick: bool = True
    orthogonal_reg_weight: float = 0.0
    orthogonal_reg_max_codes: int = 512
    orthogonal_reg_active_codes_only: bool = True
    return_zeros_for_masked_padding: bool = True


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
    decoder_output_scaling_factor: float = 1.0
    template: Optional[Tensor] = None


@dataclass
class GCPVQVAEConfig:
    """Top-level configuration for the full GCP-VQVAE model."""

    gcp: GCPNetConfig = field(default_factory=GCPNetConfig)
    encoder: TransformerConfig = field(
        default_factory=lambda: TransformerConfig(input_dim=128, output_dim=128)
    )
    decoder: TransformerConfig = field(
        default_factory=lambda: TransformerConfig(
            input_dim=128, num_layers=16, num_heads=16, num_kv_heads=1
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
        if self.decoder.use_ndlinear and self.decoder.max_length is None:
            raise ValueError(
                "Decoder NdLinear projection requires 'max_length' to be set"
            )


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
        vq_options = VectorQuantizerOptions(
            kmeans_init=self.config.vq.kmeans_init,
            kmeans_iters=self.config.vq.kmeans_iters,
            stochastic_sample_codes=self.config.vq.stochastic_sample_codes,
            sample_codebook_temp=self.config.vq.sample_codebook_temp,
            orthogonal_reg_active_codes_only=self.config.vq.orthogonal_reg_active_codes_only,
            return_zeros_for_masked_padding=self.config.vq.return_zeros_for_masked_padding,
        )
        self.vq = VectorQuantizer(
            self.config.vq.num_codes,
            self.config.vq.dim,
            beta=self.config.vq.beta,
            decay=self.config.vq.decay,
            epsilon=self.config.vq.epsilon,
            rotation_trick=self.config.vq.rotation_trick,
            orthogonal_reg_weight=self.config.vq.orthogonal_reg_weight,
            orthogonal_reg_max_codes=self.config.vq.orthogonal_reg_max_codes,
            options=vq_options,
        )
        self.decoder_transformer = GeometricTransformerDecoder(self.config.decoder)
        self.rotation_decoder = Dim6RotStructureHead(
            self.config.rotation.input_dim,
            template=self.config.rotation.template,
            decoder_output_scaling_factor=self.config.rotation.decoder_output_scaling_factor,
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
            default_checkpoint = self._default_gcp_checkpoint_path()
            if default_checkpoint is None:
                raise ValueError(
                    "GCPNet pretrained initialisation requires 'checkpoint' to be set "
                    "when the packaged default checkpoint cannot be resolved"
                )
            if not default_checkpoint.is_file():
                raise ValueError(
                    "GCPNet pretrained initialisation requires 'checkpoint' to be set "
                    f"when the packaged default checkpoint is unavailable at "
                    f"{default_checkpoint}"
                )
            checkpoint = str(default_checkpoint)
            self.config.gcp.init_checkpoint = checkpoint

        state = load_checkpoint(checkpoint, map_location="cpu")
        state_dict = self._extract_gcp_state_dict(state)
        if state_dict is None:
            raise ValueError(
                "Checkpoint does not contain GCPNet weights – expected a mapping under "
                "'gcp_state', 'model_state', 'state_dict', or a raw state dict"
            )

        target_state = self.encoder_gcp.state_dict()
        coerced = self._coerce_gcp_state_dict(state_dict, target_state)
        if not coerced:
            raise ValueError(
                "Checkpoint does not contain compatible GCPNet parameters for the current configuration"
            )

        strict = getattr(self.config.gcp, "strict_init", True)
        missing, unexpected = self.encoder_gcp.load_state_dict(coerced, strict=strict)
        if (missing or unexpected) and not strict:
            warnings.warn(
                "Loaded GCPNet weights with missing keys %s and unexpected keys %s"
                % (missing, unexpected),
                UserWarning,
            )

    @staticmethod
    def _default_gcp_checkpoint_path() -> Optional[Path]:
        resource = (
            files("gcpvqvae")
            / "models"
            / "checkpoints"
            / "gcpnet"
            / "structure_denoising"
            / "ca_bb"
            / "last.ckpt"
        )

        candidates = []
        try:
            candidates.append(Path(resource))
        except TypeError:
            pass

        base_dir = Path(__file__).resolve()
        candidates.append(
            base_dir.parent
            / "checkpoints"
            / "gcpnet"
            / "structure_denoising"
            / "ca_bb"
            / "last.ckpt"
        )

        try:
            project_root = base_dir.parents[3]
        except IndexError:
            project_root = None
        if project_root is not None:
            candidates.append(
                project_root
                / "models"
                / "checkpoints"
                / "gcpnet"
                / "structure_denoising"
                / "ca_bb"
                / "last.ckpt"
            )

        for candidate in candidates:
            if candidate is not None and candidate.is_file():
                return candidate

        searched = ", ".join(str(path) for path in candidates if path is not None)
        raise FileNotFoundError(
            "GCPNet checkpoint file not found in any known location. "
            f"Searched: {searched}"
        )

    @staticmethod
    def _extract_gcp_state_dict(
        state: Mapping[str, Any],
    ) -> Optional[Dict[str, Tensor]]:
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

        if all(
            isinstance(k, str) and isinstance(v, torch.Tensor) for k, v in state.items()
        ):
            return {k: v for k, v in state.items() if isinstance(v, torch.Tensor)}

        return None

    @staticmethod
    def _coerce_gcp_state_dict(
        source: Mapping[str, Tensor], target: Mapping[str, Tensor]
    ) -> Dict[str, Tensor]:
        coerced: Dict[str, Tensor] = {}
        for key, tensor in source.items():
            if key in target and isinstance(tensor, torch.Tensor):
                if tensor.shape == target[key].shape:
                    coerced[key] = tensor

        rename_map = {
            "embedding.node_scalar_proj.weight": "encoder.gcp_embedding.node_embedding.scalar_out.weight",
            "embedding.node_vector_proj": "encoder.gcp_embedding.node_embedding.vector_down.weight",
            "embedding.edge_scalar_proj.weight": "encoder.gcp_embedding.edge_embedding.scalar_out.weight",
            "embedding.edge_vector_proj": "encoder.gcp_embedding.edge_embedding.vector_down.weight",
        }

        for new_key, old_key in rename_map.items():
            if new_key in coerced:
                continue
            tensor = source.get(old_key)
            target_tensor = target.get(new_key)
            if tensor is None or target_tensor is None:
                continue
            if not isinstance(tensor, torch.Tensor):
                continue
            if tensor.shape != target_tensor.shape:
                continue
            coerced[new_key] = tensor

        return coerced

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
    def forward(
        self,
        batch: Dict[str, Tensor],
        *,
        decoder_only: bool = False,
        return_vq_layer: bool = False,
        mask: Optional[Tensor] = None,
        nan_mask: Optional[Tensor] = None,
    ) -> Dict[str, Tensor]:
        device = self._device()
        dtype = self._dtype()

        mask_tensor = mask if mask is not None else batch["mask"]
        mask_tensor = mask_tensor.to(device=device, dtype=torch.bool)

        if nan_mask is not None:
            nan_mask_tensor = nan_mask.to(device=device, dtype=torch.bool)
        elif "nan_mask" in batch:
            nan_mask_tensor = ~batch["nan_mask"].to(torch.bool).to(device)
        else:
            nan_mask_tensor = None

        valid = (
            mask_tensor if nan_mask_tensor is None else mask_tensor & nan_mask_tensor
        )

        batch_size, max_len = mask_tensor.shape
        latent_dim = self.config.vq.dim

        if decoder_only:
            indices = batch["indices"].to(device=device, dtype=torch.long)
            if indices.ndim == 1:
                indices = indices.unsqueeze(0)

            quantized = torch.zeros(
                (batch_size, max_len, latent_dim), device=device, dtype=dtype
            )
            if valid.any():
                decoded = self.vq.get_output_from_indices(indices[valid])
                quantized[valid] = decoded.to(dtype)

            dec_hidden = self.decoder_transformer(quantized, mask=valid)
            decoded_flat, rigid = self.rotation_decoder(dec_hidden, mask=valid)
            recon_coords = rigid["coordinates"]

            return {
                "quantized": quantized,
                "indices": indices,
                "decoded": recon_coords,
                "decoded_flat": decoded_flat,
                "mask": mask_tensor,
                "valid_mask": valid,
                "pose": rigid,
                "vq_loss": torch.zeros((), device=device, dtype=dtype),
                "vq_metrics": {},
            }

        coords = batch["coords"].to(device=device, dtype=dtype)
        proto = protein_batch_from_graph_dict(batch)
        proto = proto.to(device=device, dtype=dtype)
        proto.full_mask = valid

        gcp_out = self.encoder_gcp(proto)
        flat_embeddings = gcp_out["node_embedding"]
        latent_dim = flat_embeddings.shape[-1]
        padded = flat_embeddings.new_zeros((batch_size * max_len, latent_dim))
        padded.index_copy_(0, proto.valid_indices, flat_embeddings)
        embeddings = padded.reshape(batch_size, max_len, latent_dim)
        projected = self._project_embeddings(embeddings)

        enc_hidden = self.encoder_transformer(projected, mask=valid)
        vq_out = self.vq(enc_hidden, mask=valid, return_metrics=True)
        quantized, indices, vq_loss, vq_metrics = vq_out

        if return_vq_layer:
            return {
                "gcp_embeddings": embeddings,
                "encoder_hidden": enc_hidden,
                "quantized": quantized,
                "indices": indices,
                "mask": mask_tensor,
                "valid_mask": valid,
                "vq_loss": vq_loss,
                "vq_metrics": vq_metrics,
            }

        dec_hidden = self.decoder_transformer(quantized, mask=valid)
        decoded_flat, rigid = self.rotation_decoder(dec_hidden, mask=valid)
        recon_coords = rigid["coordinates"]

        rec_loss, rec_components = reconstruction_loss(
            recon_coords,
            coords,
            mask=valid,
            return_components=True,
        )

        total_loss = rec_loss + vq_loss

        return {
            "gcp_embeddings": embeddings,
            "encoder_hidden": enc_hidden,
            "quantized": quantized,
            "indices": indices,
            "decoded": recon_coords,
            "decoded_flat": decoded_flat,
            "mask": mask_tensor,
            "valid_mask": valid,
            "pose": rigid,
            "vq_loss": vq_loss,
            "vq_metrics": vq_metrics,
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
            batch=torch.zeros(
                node_scalars.shape[0], dtype=torch.long, device=node_scalars.device
            ),
            ptr=torch.tensor([0, node_scalars.shape[0]], device=node_scalars.device),
            mask=mask,
        )
        proto.valid_indices = torch.arange(
            node_scalars.shape[0], device=node_scalars.device
        )
        proto.full_mask = mask.unsqueeze(0)
        proto.batch_size = 1
        proto.max_length = node_scalars.shape[0]
        proto = proto.to(device=device, dtype=dtype)
        gcp_out = self.encoder_gcp(proto)
        embeddings = gcp_out["node_embedding"].unsqueeze(0)
        projected = self._project_embeddings(embeddings)
        valid = mask.unsqueeze(0)
        enc_hidden = self.encoder_transformer(projected, mask=valid)
        vq_out = self.vq(enc_hidden, mask=valid, return_metrics=True)
        quantized, indices, vq_loss, vq_metrics = vq_out
        return (
            embeddings,
            enc_hidden,
            quantized,
            {
                "indices": indices,
                "vq_loss": vq_loss,
                "vq_metrics": vq_metrics,
            },
        )

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

            valid = mask

            embeddings, enc_hidden, quantized, extras = self._run_encoder(
                features["node_scalars"],
                features["node_vectors"],
                record.coords[:, 1, :],
                features["edge_index"],
                features["edge_scalars"],
                features["edge_vectors"],
                features["edge_frames"],
                valid,
            )

            indices = extras["indices"].squeeze(0).cpu()
            vq_loss = extras["vq_loss"]
            vq_metrics = extras["vq_metrics"]

            result = {
                "tokens": indices.clone(),
                "length": int(record.length),
                "mask": mask.clone().cpu(),
                "valid_mask": valid.clone().cpu(),
                "pose_header": (
                    record.rotation.clone().cpu(),
                    record.translation.clone().cpu(),
                ),
                "gcp_embeddings": embeddings.squeeze(0).cpu(),
                "encoder_embeddings": enc_hidden.squeeze(0).cpu(),
                "quantized": quantized.squeeze(0).cpu(),
                "vq_loss": vq_loss.squeeze(0).detach().cpu(),
                "vq_metrics": {k: v.detach().cpu() for k, v in vq_metrics.items()},
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
            dequant = torch.zeros(
                (batch, length, latent_dim), device=device, dtype=dtype
            )
            valid = mask_tensor & (tokens_tensor >= 0)
            if valid.any():
                flat_indices = tokens_tensor[valid]
                decoded = self.vq.get_output_from_indices(flat_indices)
                dequant[valid] = decoded.to(dtype)

            dec_hidden = self.decoder_transformer(dequant, mask=valid)
            decoded_flat, rigid = self.rotation_decoder(dec_hidden, mask=valid)
            coords_central = rigid["coordinates"]

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

            mask_cpu = valid.clone().cpu()

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

                    residue_ids = metadata.get(
                        "residue_ids", [(i + 1, "") for i in range(valid_len)]
                    )
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

            pose_out = {
                "rotations": rigid["rotations"].cpu(),
                "translations": rigid["translations"].cpu(),
            }
            result = {
                "coords": coords_global.squeeze(0).cpu()
                if batch == 1
                else coords_global.cpu(),
                "coords_central": coords_central.squeeze(0).cpu()
                if batch == 1
                else coords_central.cpu(),
                "coords_flat": decoded_flat.squeeze(0).cpu()
                if batch == 1
                else decoded_flat.cpu(),
                "mask": mask_cpu.squeeze(0) if batch == 1 else mask_cpu,
                "valid_mask": mask_cpu.squeeze(0) if batch == 1 else mask_cpu,
                "pose": pose_out,
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

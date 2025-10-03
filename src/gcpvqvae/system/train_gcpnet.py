"""Training harness for pretraining the standalone GCPNet encoder."""

from __future__ import annotations

import contextlib
import dataclasses
import math
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

import torch
import yaml
from torch import Tensor, nn
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader

from gcpvqvae.data.dataset import BackboneDataset, collate_backbones
from gcpvqvae.geometry.metrics import rmsd
from gcpvqvae.models.decoder import RotationDecoder
from gcpvqvae.models.gcpnet import GCPNetConfig, GCPNetEncoder
from gcpvqvae.models.gcpvqvae import RotationHeadConfig
from gcpvqvae.models.losses import reconstruction_loss
from gcpvqvae.system.configuration import update_dataclass
from gcpvqvae.system.train import WarmupCosineScheduler
from gcpvqvae.utils.checkpoint import save_checkpoint
from gcpvqvae.utils.logging import get_logger
from gcpvqvae.utils.seed import seed_everything


Tensor = torch.Tensor


@dataclass
class PretrainDataConfig:
    root: str
    chain_ids: Optional[Tuple[str, ...]] = None
    length_cap: int = 512
    k: int = 16
    num_dataloader_workers: int = 0
    cache: bool = True
    show_progress: bool = True


@dataclass
class PretrainModelConfig:
    gcp: GCPNetConfig = field(default_factory=GCPNetConfig)
    rotation: RotationHeadConfig = field(default_factory=RotationHeadConfig)

    def __post_init__(self) -> None:
        self.gcp.__post_init__()
        if self.rotation.input_dim is None:
            self.rotation.input_dim = self.gcp.latent_dim


@dataclass
class PretrainTrainConfig:
    seed: int = 42
    device: Optional[str] = None
    amp: bool = True
    grad_clip: float = 1.0
    random_rotation: bool = True
    epochs: int = 1
    total_steps: Optional[int] = None
    batch_size: int = 8
    accumulation_steps: int = 1
    learning_rate: float = 1e-3
    min_lr: float = 1e-5
    warmup_steps: int = 0
    weight_decay: float = 1e-4
    log_interval: int = 50
    checkpoint_interval: Optional[int] = None
    output_dir: str = "runs/gcpnet_pretrain"


class MetricTracker:
    """Utility tracking weighted averages for streaming metrics."""

    def __init__(self) -> None:
        self.total = 0.0
        self.weight = 0.0

    def update(self, value: float, weight: float = 1.0) -> None:
        self.total += value * weight
        self.weight += weight

    def reset(self) -> None:
        self.total = 0.0
        self.weight = 0.0

    @property
    def average(self) -> float:
        if self.weight == 0:
            return 0.0
        return self.total / self.weight


def _load_config(path: str | Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError("Configuration file must define a dictionary")
    return config


def _coerce_config(config: Mapping[str, Any] | str | Path) -> Dict[str, Any]:
    if isinstance(config, Mapping):
        return dict(config)
    return _load_config(config)


def _prepare_data_config(raw: Mapping[str, Any]) -> PretrainDataConfig:
    if "root" not in raw or raw["root"] is None:
        raise ValueError("data.root must be provided for GCPNet pretraining")

    chain_ids_raw = raw.get("chain_ids")
    chain_ids: Optional[Tuple[str, ...]]
    if chain_ids_raw is None:
        chain_ids = None
    elif isinstance(chain_ids_raw, Sequence) and not isinstance(chain_ids_raw, (str, bytes)):
        chain_ids = tuple(str(item) for item in chain_ids_raw)
    else:
        raise TypeError("data.chain_ids must be a sequence of strings or null")

    return PretrainDataConfig(
        root=str(raw["root"]),
        chain_ids=chain_ids,
        length_cap=int(raw.get("length_cap", 512)),
        k=int(raw.get("k", 16)),
        num_dataloader_workers=int(raw.get("num_dataloader_workers", 0)),
        cache=bool(raw.get("cache", True)),
        show_progress=bool(raw.get("show_progress", True)),
    )


def _prepare_model_config(raw: Mapping[str, Any]) -> PretrainModelConfig:
    config = PretrainModelConfig()
    for key, value in raw.items():
        if not hasattr(config, key):
            continue
        current = getattr(config, key)
        if dataclasses.is_dataclass(current):
            setattr(config, key, update_dataclass(current, value))
        else:
            setattr(config, key, value)
    config.__post_init__()
    return config


def _prepare_train_config(raw: Mapping[str, Any]) -> PretrainTrainConfig:
    data = dict(raw)
    cfg = PretrainTrainConfig(
        seed=int(data.get("seed", 42)),
        device=data.get("device"),
        amp=bool(data.get("amp", True)),
        grad_clip=float(data.get("grad_clip", 1.0)),
        random_rotation=bool(data.get("random_rotation", True)),
        epochs=int(data.get("epochs", 1)),
        batch_size=int(data.get("batch_size", 8)),
        accumulation_steps=int(data.get("accumulation_steps", 1)),
        learning_rate=float(data.get("learning_rate", 1e-3)),
        min_lr=float(data.get("min_lr", 1e-5)),
        warmup_steps=int(data.get("warmup_steps", 0)),
        weight_decay=float(data.get("weight_decay", 1e-4)),
        log_interval=int(data.get("log_interval", 50)),
        output_dir=str(data.get("output_dir", "runs/gcpnet_pretrain")),
    )

    total_steps = data.get("total_steps")
    cfg.total_steps = int(total_steps) if total_steps is not None else None
    interval = data.get("checkpoint_interval")
    cfg.checkpoint_interval = int(interval) if interval is not None else None
    return cfg


def _random_rotation(device: torch.device, dtype: torch.dtype) -> Tensor:
    mat = torch.randn((3, 3), device=device, dtype=dtype)
    q, r = torch.linalg.qr(mat)
    diag = torch.diagonal(r)
    signs = torch.sign(diag + (diag == 0).to(dtype))
    q = q * signs
    if torch.linalg.det(q) < 0:
        q[:, 0] = -q[:, 0]
    return q


def _apply_random_rotation(batch: Dict[str, Tensor]) -> None:
    coords = batch.get("coords")
    node_vectors = batch.get("node_vectors")
    edge_vectors = batch.get("edge_vectors")
    edge_frames = batch.get("edge_frames")
    edge_batch = batch.get("edge_batch")
    backbone_vectors = batch.get("backbone_vectors")

    if coords is None or node_vectors is None:
        return

    batch_size = coords.shape[0]
    device = coords.device
    dtype = coords.dtype

    for b in range(batch_size):
        rot = _random_rotation(device, dtype)
        coords[b] = coords[b] @ rot.T
        node_vectors[b] = node_vectors[b] @ rot.T
        if backbone_vectors is not None:
            backbone_vectors[b] = backbone_vectors[b] @ rot.T
        if edge_batch is not None and edge_vectors is not None and edge_vectors.numel():
            mask = edge_batch == b
            if mask.any():
                edge_vectors[mask] = edge_vectors[mask] @ rot.T
        if edge_batch is not None and edge_frames is not None and edge_frames.numel():
            mask = edge_batch == b
            if mask.any():
                edge_frames[mask] = edge_frames[mask] @ rot.T


def _move_batch_to_device(batch: Dict[str, Any], device: torch.device) -> Dict[str, Any]:
    for key, value in batch.items():
        if isinstance(value, torch.Tensor):
            batch[key] = value.to(device)
    return batch


class GCPNetPretrainModule(nn.Module):
    def __init__(self, config: PretrainModelConfig) -> None:
        super().__init__()
        self.encoder = GCPNetEncoder(config.gcp)
        self.rotation = RotationDecoder(
            config.rotation.input_dim,
            translation_scale=config.rotation.translation_scale,
            template=config.rotation.template,
        )

    def forward(self, batch: Dict[str, Tensor]) -> Dict[str, Tensor]:
        mask = batch["mask"].to(torch.bool)
        if "nan_mask" in batch:
            mask = mask & ~batch["nan_mask"].to(torch.bool)

        coords = batch["coords"]
        node_scalars = batch["node_scalars"]
        node_vectors = batch["node_vectors"]
        edge_index = batch["edge_index"]
        edge_scalars = batch["edge_scalars"]
        edge_vectors = batch["edge_vectors"]
        edge_frames = batch["edge_frames"]

        batch_size, max_len, _ = node_scalars.shape
        flat_scalars = node_scalars.reshape(batch_size * max_len, -1)
        flat_vectors = node_vectors.reshape(batch_size * max_len, node_vectors.shape[2], node_vectors.shape[3])
        flat_mask = mask.reshape(-1)

        gcp_out = self.encoder(
            flat_scalars,
            flat_vectors,
            edge_index,
            edge_scalars,
            edge_vectors,
            edge_frames,
            mask=flat_mask,
        )
        embeddings = gcp_out["embeddings"].reshape(batch_size, max_len, -1)

        recon_coords, pose = self.rotation(embeddings, mask=mask)
        rec_loss, rec_components = reconstruction_loss(
            recon_coords, coords, mask=mask, return_components=True
        )

        return {
            "total_loss": rec_loss,
            "reconstruction": rec_loss,
            "reconstruction_components": rec_components,
            "coords": recon_coords,
            "mask": mask,
        }


class GCPNetPretrainer:
    def __init__(self, config: Mapping[str, Any] | str | Path) -> None:
        raw = _coerce_config(config)
        self._raw_config = raw
        self.data_cfg = _prepare_data_config(raw.get("data", {}))
        self.model_cfg = _prepare_model_config(raw.get("model", {}))
        self.train_cfg = _prepare_train_config(raw.get("train", {}))
        self.logger = get_logger()

        seed_everything(self.train_cfg.seed)

        self.device = torch.device(
            self.train_cfg.device
            if self.train_cfg.device is not None
            else ("cuda" if torch.cuda.is_available() else "cpu")
        )

        self.module = GCPNetPretrainModule(self.model_cfg).to(self.device)
        self.optimizer = torch.optim.AdamW(
            self.module.parameters(),
            lr=self.train_cfg.learning_rate,
            betas=(0.9, 0.98),
            eps=1e-7,
            weight_decay=self.train_cfg.weight_decay,
        )

        self.amp_enabled = self.train_cfg.amp and self.device.type == "cuda"
        self.scaler = torch.cuda.amp.GradScaler(enabled=self.amp_enabled)

        self.output_dir = Path(self.train_cfg.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_dir = self.output_dir / "checkpoints"
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    def _build_dataloader(self) -> DataLoader:
        dataset = BackboneDataset(
            self.data_cfg.root,
            chain_ids=self.data_cfg.chain_ids,
            length_cap=self.data_cfg.length_cap,
            k=self.data_cfg.k,
            cache=self.data_cfg.cache,
            progress=self.data_cfg.show_progress,
        )
        if len(dataset) == 0:
            raise ValueError("Training dataset is empty")
        return DataLoader(
            dataset,
            batch_size=self.train_cfg.batch_size,
            shuffle=True,
            num_workers=self.data_cfg.num_dataloader_workers,
            pin_memory=self.device.type == "cuda",
            collate_fn=collate_backbones,
            drop_last=False,
        )

    def _save_checkpoint(self, name: str, global_step: int, epoch: int) -> None:
        path = self.checkpoint_dir / name
        state = {
            "epoch": epoch,
            "global_step": global_step,
            "gcp_state": self.module.encoder.state_dict(),
            "rotation_state": self.module.rotation.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "scaler": self.scaler.state_dict(),
            "config": self._raw_config,
        }
        save_checkpoint(state, path)
        self.logger.info("Saved checkpoint to %s", path)

    def run(self) -> None:
        dataloader = self._build_dataloader()
        batches_per_epoch = len(dataloader)
        if batches_per_epoch == 0:
            raise ValueError("Dataset produced no batches")

        steps_per_epoch = math.ceil(batches_per_epoch / max(self.train_cfg.accumulation_steps, 1))
        if self.train_cfg.total_steps is not None:
            total_steps = max(int(self.train_cfg.total_steps), 1)
        else:
            total_steps = max(self.train_cfg.epochs, 1) * max(steps_per_epoch, 1)

        scheduler = WarmupCosineScheduler(
            self.optimizer,
            warmup_steps=self.train_cfg.warmup_steps,
            total_steps=total_steps,
            base_lr=self.train_cfg.learning_rate,
            min_lr=self.train_cfg.min_lr,
        )

        trackers = {
            "loss": MetricTracker(),
            "recon": MetricTracker(),
            "recon_total": MetricTracker(),
            "recon_aligned": MetricTracker(),
            "recon_distance": MetricTracker(),
            "recon_direction": MetricTracker(),
            "rmsd": MetricTracker(),
        }

        global_step = 0
        accum_counter = 0
        samples_since_log = 0
        residues_since_log = 0
        last_log_time = time.perf_counter()

        autocast_context = (
            torch.cuda.amp.autocast if self.amp_enabled else contextlib.nullcontext
        )
        autocast_kwargs = {"dtype": torch.bfloat16} if self.amp_enabled else {}

        self.module.train()
        self.optimizer.zero_grad(set_to_none=True)

        for epoch in range(max(self.train_cfg.epochs, 1)):
            if global_step >= total_steps:
                break
            for batch in dataloader:
                if global_step >= total_steps:
                    break

                batch = _move_batch_to_device(batch, self.device)
                if self.train_cfg.random_rotation:
                    _apply_random_rotation(batch)  # type: ignore[arg-type]

                with autocast_context(**autocast_kwargs):
                    outputs = self.module(batch)  # type: ignore[arg-type]
                    loss = outputs["total_loss"] / max(self.train_cfg.accumulation_steps, 1)

                self.scaler.scale(loss).backward()
                accum_counter += 1

                mask = outputs["mask"]
                batch_size = int(mask.shape[0])
                residue_count = int(mask.sum().item())

                trackers["loss"].update(float(outputs["total_loss"].detach().item()), batch_size)
                trackers["recon"].update(float(outputs["reconstruction"].detach().item()), batch_size)

                recon_components = outputs.get("reconstruction_components")
                if isinstance(recon_components, Mapping):
                    total_component = recon_components.get("total")
                    if total_component is not None:
                        trackers["recon_total"].update(float(total_component.detach().item()), batch_size)
                    aligned_component = recon_components.get("aligned_mse")
                    if aligned_component is not None:
                        trackers["recon_aligned"].update(float(aligned_component.detach().item()), batch_size)
                    distance_component = recon_components.get("distance")
                    if distance_component is not None:
                        trackers["recon_distance"].update(float(distance_component.detach().item()), batch_size)
                    direction_component = recon_components.get("direction")
                    if direction_component is not None:
                        trackers["recon_direction"].update(float(direction_component.detach().item()), batch_size)

                with torch.no_grad():
                    rmsd_value = rmsd(
                        outputs["coords"].detach(),
                        batch["coords"],
                        mask=outputs["mask"],
                    ).item()
                trackers["rmsd"].update(rmsd_value, batch_size)

                samples_since_log += batch_size
                residues_since_log += residue_count

                if accum_counter >= max(self.train_cfg.accumulation_steps, 1):
                    self.scaler.unscale_(self.optimizer)
                    if self.train_cfg.grad_clip > 0:
                        clip_grad_norm_(self.module.parameters(), self.train_cfg.grad_clip)
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                    self.optimizer.zero_grad(set_to_none=True)
                    accum_counter = 0

                    global_step += 1
                    scheduler.step()

                    if (
                        self.train_cfg.log_interval > 0
                        and global_step % self.train_cfg.log_interval == 0
                    ):
                        elapsed = time.perf_counter() - last_log_time
                        denom = max(elapsed, 1e-6)
                        lr = self.optimizer.param_groups[0]["lr"]
                        self.logger.info(
                            "step %d/%d | loss %.4f | recon %.4f | rmsd %.3f Å | lr %.2e | %.2f seq/s %.2f res/s",
                            global_step,
                            total_steps,
                            trackers["loss"].average,
                            trackers["recon"].average,
                            trackers["rmsd"].average,
                            lr,
                            samples_since_log / denom,
                            residues_since_log / denom,
                        )
                        for tracker in trackers.values():
                            tracker.reset()
                        samples_since_log = 0
                        residues_since_log = 0
                        last_log_time = time.perf_counter()

                    interval = self.train_cfg.checkpoint_interval
                    if interval is not None and interval > 0 and global_step % interval == 0:
                        self._save_checkpoint(f"step{global_step:06d}.pt", global_step, epoch + 1)

        if accum_counter and global_step < total_steps:
            self.scaler.unscale_(self.optimizer)
            if self.train_cfg.grad_clip > 0:
                clip_grad_norm_(self.module.parameters(), self.train_cfg.grad_clip)
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.optimizer.zero_grad(set_to_none=True)
            global_step += 1
            scheduler.step()

        self._save_checkpoint("final.pt", global_step, max(self.train_cfg.epochs, 1))


def train(config: Mapping[str, Any] | str | Path) -> None:
    """Entry point mirroring :func:`gcpvqvae.system.train.train`."""

    trainer = GCPNetPretrainer(config)
    trainer.run()


__all__ = ["train", "GCPNetPretrainer"]


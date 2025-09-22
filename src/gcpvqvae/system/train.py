"""High level training harness for the GCP-VQVAE model."""

from __future__ import annotations

import contextlib
import dataclasses
import math
import random
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch import nn
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader

import yaml

from gcpvqvae.data.dataset import BackboneDataset, collate_backbones
from gcpvqvae.data.mmcif import BackboneRecord, write_mmcif
from gcpvqvae.geometry.metrics import rmsd
from gcpvqvae.models.gcpvqvae import GCPVQVAE, GCPVQVAEConfig
from gcpvqvae.utils.checkpoint import save_checkpoint
from gcpvqvae.utils.logging import get_logger
from gcpvqvae.utils.seed import seed_everything


Tensor = torch.Tensor


@dataclass
class DataConfig:
    root: str
    chain_ids: Optional[Sequence[str]] = None
    k: int = 16
    num_workers: int = 0
    cache: bool = True


@dataclass
class StageConfig:
    name: str
    length_cap: int
    batch_size: int
    base_lr: float
    min_lr: float
    warmup_steps: int
    total_steps: Optional[int] = None
    epochs: Optional[int] = None
    accumulation_steps: int = 1
    nan_mask_prob: float = 0.0
    nan_mask_span: Tuple[int, int] = (1, 1)

    def effective_total_steps(self, batches_per_epoch: int) -> int:
        if self.total_steps is not None:
            return self.total_steps
        if self.epochs is None:
            raise ValueError(f"Stage {self.name} requires either total_steps or epochs")
        if batches_per_epoch == 0:
            raise ValueError("Cannot compute steps with an empty dataset")
        steps_per_epoch = math.ceil(batches_per_epoch / max(self.accumulation_steps, 1))
        return self.epochs * max(steps_per_epoch, 1)


@dataclass
class ExportConfig:
    enabled: bool = True
    directory: Optional[str] = None
    every_n_steps: Optional[int] = None
    on_stage_end: bool = True
    num_samples: int = 1


@dataclass
class TrainConfig:
    seed: int = 42
    device: Optional[str] = None
    amp: bool = True
    clip_grad: float = 1.0
    random_rotation: bool = True
    log_interval: int = 50
    checkpoint_interval: Optional[int] = None
    output_dir: str = "runs"
    export: ExportConfig = field(default_factory=ExportConfig)
    stages: List[StageConfig] = field(default_factory=list)


class MetricTracker:
    """Utility tracking running averages for logging."""

    def __init__(self) -> None:
        self.total = 0.0
        self.count = 0.0

    def update(self, value: float, weight: float = 1.0) -> None:
        self.total += value * weight
        self.count += weight

    def reset(self) -> None:
        self.total = 0.0
        self.count = 0.0

    @property
    def average(self) -> float:
        if self.count == 0:
            return 0.0
        return self.total / self.count


class WarmupCosineScheduler:
    """Cosine learning-rate schedule with linear warmup."""

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        *,
        warmup_steps: int,
        total_steps: int,
        base_lr: float,
        min_lr: float,
    ) -> None:
        self.optimizer = optimizer
        self.warmup_steps = max(warmup_steps, 0)
        self.total_steps = max(total_steps, 1)
        self.base_lr = base_lr
        self.min_lr = min_lr
        self._step = 0
        self.update(0)

    def _lr_at(self, step: int) -> float:
        step = min(max(step, 0), self.total_steps)
        if self.total_steps <= 0:
            return self.min_lr
        if self.warmup_steps > 0 and step <= self.warmup_steps:
            progress = step / max(self.warmup_steps, 1)
            return self.min_lr + (self.base_lr - self.min_lr) * progress
        if self.total_steps == self.warmup_steps:
            return self.base_lr
        progress = (step - self.warmup_steps) / max(self.total_steps - self.warmup_steps, 1)
        progress = min(max(progress, 0.0), 1.0)
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return self.min_lr + (self.base_lr - self.min_lr) * cosine

    def update(self, step: int) -> None:
        lr = self._lr_at(step)
        for group in self.optimizer.param_groups:
            group["lr"] = lr
        self._step = step

    def step(self) -> None:
        self.update(self._step + 1)


def _update_dataclass(instance: Any, updates: Dict[str, Any]) -> Any:
    if not dataclasses.is_dataclass(instance) or not isinstance(updates, dict):
        return instance
    for key, value in updates.items():
        if not hasattr(instance, key):
            continue
        current = getattr(instance, key)
        if dataclasses.is_dataclass(current):
            setattr(instance, key, _update_dataclass(current, value))
        else:
            setattr(instance, key, value)
    return instance


def _build_model_config(raw: Optional[Dict[str, Any]]) -> GCPVQVAEConfig:
    config = GCPVQVAEConfig()
    if raw:
        for key, value in raw.items():
            if not hasattr(config, key):
                continue
            current = getattr(config, key)
            if dataclasses.is_dataclass(current):
                setattr(config, key, _update_dataclass(current, value))
            else:
                setattr(config, key, value)
        config.rotation.input_dim = None
        config.__post_init__()
    return config


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


def _apply_nan_mask(batch: Dict[str, Tensor], prob: float, span: Tuple[int, int]) -> None:
    if prob <= 0.0:
        return
    nan_mask = batch.get("nan_mask")
    lengths = batch.get("lengths")
    if nan_mask is None or lengths is None:
        return
    min_span, max_span = span
    for i in range(nan_mask.shape[0]):
        if random.random() >= prob:
            continue
        length = int(lengths[i].item())
        if length <= 0:
            continue
        span_length = random.randint(min_span, min(max_span, length))
        if span_length <= 0:
            continue
        start = random.randint(0, max(length - span_length, 0))
        end = min(start + span_length, length)
        nan_mask[i, start:end] = True


def _prepare_stage_config(data: Dict[str, Any]) -> StageConfig:
    span = data.get("nan_mask_span", (1, 1))
    if isinstance(span, Sequence):
        span_tuple = (int(span[0]), int(span[1]))
    else:
        span_tuple = (1, 1)
    total_steps = data.get("total_steps")
    epochs = data.get("epochs")
    return StageConfig(
        name=str(data.get("name", "stage")),
        length_cap=int(data.get("length_cap", 512)),
        batch_size=int(data.get("batch_size", 1)),
        base_lr=float(data.get("base_lr", 1e-4)),
        min_lr=float(data.get("min_lr", 1e-6)),
        warmup_steps=int(data.get("warmup_steps", 0)),
        total_steps=int(total_steps) if total_steps is not None else None,
        epochs=int(epochs) if epochs is not None else None,
        accumulation_steps=int(data.get("accumulation_steps", 1)),
        nan_mask_prob=float(data.get("nan_mask_prob", 0.0)),
        nan_mask_span=span_tuple,
    )


def _prepare_train_config(raw: Dict[str, Any]) -> TrainConfig:
    export_cfg = raw.get("export", {})
    every_n = export_cfg.get("every_n_steps")
    export = ExportConfig(
        enabled=bool(export_cfg.get("enabled", True)),
        directory=export_cfg.get("directory"),
        every_n_steps=int(every_n) if every_n is not None else None,
        on_stage_end=bool(export_cfg.get("on_stage_end", True)),
        num_samples=int(export_cfg.get("num_samples", 1)),
    )
    stages = [_prepare_stage_config(stage) for stage in raw.get("stages", [])]
    if not stages:
        raise ValueError("Training configuration must specify at least one stage")
    checkpoint = raw.get("checkpoint_interval")
    return TrainConfig(
        seed=int(raw.get("seed", 42)),
        device=raw.get("device"),
        amp=bool(raw.get("amp", True)),
        clip_grad=float(raw.get("clip_grad", 1.0)),
        random_rotation=bool(raw.get("random_rotation", True)),
        log_interval=int(raw.get("log_interval", 50)),
        checkpoint_interval=int(checkpoint) if checkpoint is not None else None,
        output_dir=str(raw.get("output_dir", "runs")),
        export=export,
        stages=stages,
    )


def _prepare_data_config(raw: Dict[str, Any]) -> DataConfig:
    root = raw.get("root")
    if root is None:
        raise ValueError("Data configuration requires a 'root' path")
    return DataConfig(
        root=str(root),
        chain_ids=raw.get("chain_ids"),
        k=int(raw.get("k", 16)),
        num_workers=int(raw.get("num_workers", 0)),
        cache=bool(raw.get("cache", True)),
    )


def _load_config(path: str | Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError("Configuration file must define a dictionary")
    return config


def _codebook_statistics(vq: nn.Module) -> Tuple[float, float]:
    if not hasattr(vq, "usage"):
        return 0.0, 0.0
    usage = getattr(vq, "usage").detach().float()
    if usage.numel() == 0:
        return 0.0, 0.0
    active = float((usage > 0).sum().item()) / float(usage.numel())
    total = usage.sum()
    if total <= 0:
        return active, 0.0
    probs = usage / total
    mask = probs > 0
    if mask.any():
        uniform = math.log(1.0 / probs.numel())
        kl = float((probs[mask] * (torch.log(probs[mask]) - uniform)).sum().item())
    else:
        kl = 0.0
    return active, kl


def _export_samples(
    model: GCPVQVAE,
    dataset: BackboneDataset,
    export_cfg: ExportConfig,
    stage_name: str,
    global_step: int,
    output_dir: Path,
    logger,
) -> None:
    if not export_cfg.enabled:
        return
    if not hasattr(dataset, "_keys") or not dataset._keys:  # type: ignore[attr-defined]
        return
    base_dir = Path(export_cfg.directory) if export_cfg.directory else output_dir / "exports"
    target_dir = base_dir / f"{stage_name}_step{global_step:06d}"
    target_dir.mkdir(parents=True, exist_ok=True)

    was_training = model.training
    model.eval()
    try:
        keys: List[Tuple[str, str]] = dataset._keys  # type: ignore[attr-defined]
        for key in keys[: export_cfg.num_samples]:
            path, chain_id = key
            try:
                encoded = model.encode(path, chain_id=chain_id)
            except Exception as exc:  # pragma: no cover - encoding errors are logged
                logger.warning("Failed to encode %s chain %s: %s", path, chain_id, exc)
                continue

            tokens = encoded["tokens"].cpu().numpy().astype(np.int32)
            mask = encoded["mask"].cpu().numpy().astype(np.bool_)
            rotation, translation = encoded["pose_header"]
            np.savez(
                target_dir / f"{Path(path).stem}_{chain_id}.npz",
                tokens=tokens,
                mask=mask,
                length=np.int32(encoded["length"]),
                rotation=rotation.cpu().numpy(),
                translation=translation.cpu().numpy(),
            )

            try:
                decoded = model.decode(
                    tokens,
                    pose_header=(rotation, translation),
                    mask=mask,
                    metadata=encoded.get("metadata"),
                )
            except Exception as exc:  # pragma: no cover - decoding errors are logged
                logger.warning("Failed to decode tokens for %s chain %s: %s", path, chain_id, exc)
                continue

            records = decoded.get("records")
            record: Optional[BackboneRecord]
            if isinstance(records, BackboneRecord):
                record = records
            elif isinstance(records, list) and records:
                record = records[0]
            else:
                record = None

            if record is None:
                continue

            cif_path = target_dir / f"{Path(path).stem}_{chain_id}_recon.cif"
            try:
                write_mmcif(record, str(cif_path))
            except Exception as exc:  # pragma: no cover - gemmi may be unavailable
                logger.warning("Failed to write mmCIF for %s chain %s: %s", path, chain_id, exc)
    finally:
        model.train(was_training)


class Trainer:
    def __init__(self, config_path: str | Path) -> None:
        raw = _load_config(config_path)
        self.data_cfg = _prepare_data_config(raw.get("data", {}))
        self.train_cfg = _prepare_train_config(raw.get("train", {}))
        self.model_cfg = _build_model_config(raw.get("model"))
        self.logger = get_logger()

        seed_everything(self.train_cfg.seed)

        self.device = torch.device(
            self.train_cfg.device
            if self.train_cfg.device is not None
            else ("cuda" if torch.cuda.is_available() else "cpu")
        )

        self.model = GCPVQVAE(self.model_cfg).to(self.device)
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.train_cfg.stages[0].base_lr,
            betas=(0.9, 0.98),
            eps=1e-7,
            weight_decay=1e-3,
        )

        self.global_step = 0
        self.output_dir = Path(self.train_cfg.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.amp_enabled = self.train_cfg.amp and self.device.type == "cuda"

    def _log_stage_progress(
        self,
        stage: StageConfig,
        stage_step: int,
        total_steps: int,
        trackers: Dict[str, MetricTracker],
        samples: int,
        residues: int,
        elapsed: float,
    ) -> None:
        loss_avg = trackers["loss"].average
        rec_avg = trackers["recon"].average
        rmsd_avg = trackers["rmsd"].average
        perplexity_avg = trackers["perplexity"].average
        trackers["loss"].reset()
        trackers["recon"].reset()
        trackers["rmsd"].reset()
        trackers["perplexity"].reset()

        util, kl = _codebook_statistics(self.model.vq)
        lr = self.optimizer.param_groups[0]["lr"]
        denom = max(elapsed, 1e-6)
        ex_speed = samples / denom
        res_speed = residues / denom

        self.logger.info(
            "[%s] step %d/%d | loss %.4f | rec %.4f | rmsd %.3f Å | perp %.2f | util %.2f%% | KL %.4f | lr %.2e | %.2f seq/s %.2f res/s",
            stage.name,
            stage_step,
            total_steps,
            loss_avg,
            rec_avg,
            rmsd_avg,
            perplexity_avg,
            util * 100.0,
            kl,
            lr,
            ex_speed,
            res_speed,
        )

    def _save_checkpoint(self, stage: StageConfig) -> None:
        ckpt_dir = self.output_dir / "checkpoints"
        ckpt_path = ckpt_dir / f"{stage.name}_step{self.global_step:06d}.pt"
        save_checkpoint(
            {
                "model": self.model.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "stage": stage.name,
                "global_step": self.global_step,
                "config": dataclasses.asdict(self.model_cfg),
            },
            ckpt_path,
        )

    def _build_dataloader(self, stage: StageConfig) -> DataLoader:
        dataset = BackboneDataset(
            self.data_cfg.root,
            chain_ids=self.data_cfg.chain_ids,
            length_cap=stage.length_cap,
            k=self.data_cfg.k,
            cache=self.data_cfg.cache,
        )
        loader = DataLoader(
            dataset,
            batch_size=stage.batch_size,
            shuffle=True,
            num_workers=self.data_cfg.num_workers,
            pin_memory=self.device.type == "cuda",
            collate_fn=collate_backbones,
            drop_last=False,
        )
        return loader

    def run(self) -> None:
        for stage_idx, stage in enumerate(self.train_cfg.stages):
            self.logger.info("Starting stage %d: %s", stage_idx + 1, stage.name)

            dataloader = self._build_dataloader(stage)
            batches_per_epoch = len(dataloader)
            total_updates = stage.effective_total_steps(batches_per_epoch)
            scheduler = WarmupCosineScheduler(
                self.optimizer,
                warmup_steps=stage.warmup_steps,
                total_steps=total_updates,
                base_lr=stage.base_lr,
                min_lr=stage.min_lr,
            )

            trackers = {
                "loss": MetricTracker(),
                "recon": MetricTracker(),
                "rmsd": MetricTracker(),
                "perplexity": MetricTracker(),
            }

            samples_since_log = 0
            residues_since_log = 0
            last_log_time = time.perf_counter()

            stage_step = 0
            accum_counter = 0

            autocast_context = (
                torch.cuda.amp.autocast if self.amp_enabled else contextlib.nullcontext
            )
            autocast_kwargs = {"dtype": torch.bfloat16} if self.amp_enabled else {}

            self.optimizer.zero_grad(set_to_none=True)

            while stage_step < total_updates:
                for batch in dataloader:
                    if self.train_cfg.random_rotation:
                        _apply_random_rotation(batch)
                    if stage.nan_mask_prob > 0.0:
                        _apply_nan_mask(batch, stage.nan_mask_prob, stage.nan_mask_span)

                    with autocast_context(**autocast_kwargs):
                        outputs = self.model(batch)
                        loss = outputs["total_loss"] / max(stage.accumulation_steps, 1)

                    loss.backward()
                    accum_counter += 1

                    batch_mask = batch["mask"]
                    batch_size = int(batch_mask.shape[0])
                    residue_count = int(batch_mask.sum().item())

                    trackers["loss"].update(float(outputs["total_loss"].detach().item()), batch_size)
                    trackers["recon"].update(float(outputs["reconstruction"].detach().item()), batch_size)
                    with torch.no_grad():
                        target_coords = batch["coords"].to(
                            device=self.device, dtype=outputs["decoded"].dtype
                        )
                        rmsd_val = rmsd(outputs["decoded"].detach(), target_coords, mask=outputs["mask"]).item()
                        trackers["rmsd"].update(rmsd_val, batch_size)
                        trackers["perplexity"].update(
                            float(outputs["vq_losses"]["perplexity"].detach().item())
                        )

                    samples_since_log += batch_size
                    residues_since_log += residue_count

                    if accum_counter >= stage.accumulation_steps:
                        if self.train_cfg.clip_grad > 0:
                            clip_grad_norm_(self.model.parameters(), self.train_cfg.clip_grad)
                        self.optimizer.step()
                        self.optimizer.zero_grad(set_to_none=True)
                        scheduler.step()
                        accum_counter = 0

                        stage_step += 1
                        self.global_step += 1

                        if (
                            self.train_cfg.checkpoint_interval
                            and self.global_step % int(self.train_cfg.checkpoint_interval) == 0
                        ):
                            self._save_checkpoint(stage)

                        if (
                            self.train_cfg.export.enabled
                            and self.train_cfg.export.every_n_steps
                            and self.global_step % int(self.train_cfg.export.every_n_steps) == 0
                        ):
                            dataset = dataloader.dataset  # type: ignore[assignment]
                            _export_samples(
                                self.model,
                                dataset,
                                self.train_cfg.export,
                                stage.name,
                                self.global_step,
                                self.output_dir,
                                self.logger,
                            )

                        if self.train_cfg.log_interval and stage_step % self.train_cfg.log_interval == 0:
                            now = time.perf_counter()
                            self._log_stage_progress(
                                stage,
                                stage_step,
                                total_updates,
                                trackers,
                                samples_since_log,
                                residues_since_log,
                                now - last_log_time,
                            )
                            samples_since_log = 0
                            residues_since_log = 0
                            last_log_time = now

                        if stage_step >= total_updates:
                            break

                if stage_step >= total_updates:
                    break

            # Flush any remaining gradients (unlikely when accumulation divides batches).
            if accum_counter > 0 and stage_step < total_updates:
                if self.train_cfg.clip_grad > 0:
                    clip_grad_norm_(self.model.parameters(), self.train_cfg.clip_grad)
                self.optimizer.step()
                self.optimizer.zero_grad(set_to_none=True)
                scheduler.step()
                stage_step += 1
                self.global_step += 1

            self._save_checkpoint(stage)

            if self.train_cfg.export.enabled and self.train_cfg.export.on_stage_end:
                dataset = dataloader.dataset  # type: ignore[assignment]
                _export_samples(
                    self.model,
                    dataset,
                    self.train_cfg.export,
                    stage.name,
                    self.global_step,
                    self.output_dir,
                    self.logger,
                )

            self.logger.info("Completed stage %s", stage.name)


def train(config_path: str) -> None:
    """Entry point mirroring the public API."""

    Trainer(config_path).run()


__all__ = ["train", "Trainer"]


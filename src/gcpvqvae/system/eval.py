"""Evaluation utilities for trained checkpoints."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader

from gcpvqvae.data.dataset import BackboneDataset, collate_backbones
from gcpvqvae.geometry.frames import kabsch_align
from gcpvqvae.geometry.metrics import gdt_ts, tm_score
from gcpvqvae.models.gcpvqvae import GCPVQVAE, GCPVQVAEConfig
from gcpvqvae.system.configuration import build_model_config
from gcpvqvae.utils.checkpoint import load_checkpoint
from gcpvqvae.utils.logging import get_logger


Tensor = torch.Tensor


@dataclass
class EvalDataConfig:
    root: str
    chain_ids: Optional[Sequence[str]] = None
    length_cap: int = 2048
    k: int = 16
    num_dataloader_workers: int = 0
    cache: bool = True
    show_progress: bool = True


@dataclass
class EvalRuntimeConfig:
    batch_size: int = 1
    device: Optional[str] = None
    tm_score: bool = True
    gdt_ts: bool = False
    max_batches: Optional[int] = None
    histogram_bins: int = 20
    quantiles: Tuple[float, ...] = (0.05, 0.5, 0.95)


@dataclass
class EvalModelConfig:
    checkpoint: str
    config: Optional[Dict[str, Any]] = None


def _load_config(path: str | Path) -> Dict[str, Any]:
    import yaml

    with open(path, "r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)
    if not isinstance(raw, dict):
        raise ValueError("Configuration file must define a dictionary")
    return raw


def _coerce_config(config: Mapping[str, Any] | str | Path) -> Dict[str, Any]:
    if isinstance(config, Mapping):
        return dict(config)
    return _load_config(config)


def _prepare_data_config(raw: Dict[str, Any]) -> EvalDataConfig:
    if not isinstance(raw, dict) or "root" not in raw:
        raise ValueError("data.root must be provided in the evaluation config")

    chain_ids = raw.get("chain_ids")
    if chain_ids is not None and not isinstance(chain_ids, Sequence):
        raise ValueError("data.chain_ids must be a sequence of chain identifiers")

    return EvalDataConfig(
        root=str(raw["root"]),
        chain_ids=tuple(chain_ids) if chain_ids is not None else None,
        length_cap=int(raw.get("length_cap", 2048)),
        k=int(raw.get("k", 16)),
        num_dataloader_workers=int(
            raw.get("num_dataloader_workers", raw.get("num_workers", 0))
        ),
        cache=bool(raw.get("cache", True)),
        show_progress=bool(raw.get("show_progress", raw.get("progress", True))),
    )


def _prepare_runtime_config(raw: Dict[str, Any]) -> EvalRuntimeConfig:
    if raw is None:
        raw = {}
    if not isinstance(raw, dict):
        raise ValueError("eval section must be a dictionary")

    max_batches = raw.get("max_batches")
    if max_batches is not None:
        max_batches = int(max_batches)
        if max_batches <= 0:
            max_batches = None

    quantiles_raw = raw.get("quantiles", (0.05, 0.5, 0.95))
    if quantiles_raw is None:
        quantiles = ()
    else:
        quantiles = []
        for value in quantiles_raw:
            q = float(value)
            if 0.0 <= q <= 1.0:
                quantiles.append(q)
        quantiles = tuple(quantiles)

    return EvalRuntimeConfig(
        batch_size=int(raw.get("batch_size", 1)),
        device=raw.get("device"),
        tm_score=bool(raw.get("tm_score", True)),
        gdt_ts=bool(raw.get("gdt_ts", False)),
        max_batches=max_batches,
        histogram_bins=max(int(raw.get("histogram_bins", 20)), 1),
        quantiles=quantiles,
    )


def _prepare_model_config(raw: Dict[str, Any]) -> EvalModelConfig:
    if not isinstance(raw, dict) or "checkpoint" not in raw:
        raise ValueError("model.checkpoint must be provided in the evaluation config")

    overrides = raw.get("config")
    if overrides is not None and not isinstance(overrides, dict):
        raise ValueError("model.config must be a dictionary when provided")

    return EvalModelConfig(checkpoint=str(raw["checkpoint"]), config=overrides)


def _build_model_config(raw: Optional[Dict[str, Any]]) -> GCPVQVAEConfig:
    return build_model_config(raw)


def _linear_regression(x: Sequence[float], y: Sequence[float]) -> Tuple[float, float]:
    if len(x) < 2 or len(y) < 2:
        return float("nan"), float("nan")

    x_arr = np.asarray(x, dtype=np.float64)
    y_arr = np.asarray(y, dtype=np.float64)

    x_mean = float(x_arr.mean())
    y_mean = float(y_arr.mean())

    denom = float(np.sum((x_arr - x_mean) ** 2))
    if denom <= 0.0:
        return float("nan"), y_mean

    slope = float(np.sum((x_arr - x_mean) * (y_arr - y_mean)) / denom)
    intercept = float(y_mean - slope * x_mean)
    return slope, intercept


def _distribution_summary(values: Sequence[float], quantiles: Sequence[float], bins: int) -> Dict[str, Any]:
    if not values:
        return {
            "mean": float("nan"),
            "median": float("nan"),
            "std": float("nan"),
            "min": float("nan"),
            "max": float("nan"),
            "quantiles": {},
            "histogram": {"counts": [], "edges": []},
        }

    arr = np.asarray(values, dtype=np.float64)
    stats = {
        "mean": float(arr.mean()),
        "median": float(np.median(arr)),
        "std": float(arr.std(ddof=0)),
        "min": float(arr.min()),
        "max": float(arr.max()),
    }

    if quantiles:
        q_vals = np.quantile(arr, quantiles)
        stats["quantiles"] = {float(q): float(v) for q, v in zip(quantiles, q_vals)}
    else:
        stats["quantiles"] = {}

    counts, edges = np.histogram(arr, bins=bins)
    stats["histogram"] = {"counts": counts.tolist(), "edges": edges.astype(float).tolist()}
    return stats


def _log_distribution(logger, name: str, stats: Dict[str, Any]) -> None:
    logger.info(
        "%s: mean=%.3f Å | median=%.3f Å | std=%.3f Å | min=%.3f Å | max=%.3f Å",
        name,
        stats["mean"],
        stats["median"],
        stats["std"],
        stats["min"],
        stats["max"],
    )
    if stats.get("quantiles"):
        quantiles = ", ".join(
            f"{q * 100:.0f}%%={v:.3f} Å" for q, v in sorted(stats["quantiles"].items())
        )
        logger.info("%s quantiles: %s", name, quantiles)


class Evaluator:
    def __init__(self, config: Mapping[str, Any] | str | Path) -> None:
        self.logger = get_logger()
        if isinstance(config, Mapping):
            self.config_path = None
        else:
            self.config_path = Path(config)
        raw = _coerce_config(config)
        self.data_cfg = _prepare_data_config(raw.get("data", {}))
        self.runtime_cfg = _prepare_runtime_config(raw.get("eval", {}))
        self.model_cfg = _prepare_model_config(raw.get("model", {}))

        device_str = self.runtime_cfg.device or (
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.device = torch.device(device_str)

        self.logger.info(
            "Evaluating checkpoint %s on %s", self.model_cfg.checkpoint, self.data_cfg.root
        )

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
            raise ValueError("Evaluation dataset is empty")
        loader = DataLoader(
            dataset,
            batch_size=self.runtime_cfg.batch_size,
            shuffle=False,
            num_workers=self.data_cfg.num_dataloader_workers,
            pin_memory=self.device.type == "cuda",
            collate_fn=collate_backbones,
            drop_last=False,
        )
        return loader

    def _load_model(self) -> GCPVQVAE:
        checkpoint = load_checkpoint(self.model_cfg.checkpoint, map_location=self.device)
        state_dict = checkpoint.get("model")
        if state_dict is None:
            raise KeyError("Checkpoint does not contain a 'model' state_dict")

        raw_config = self.model_cfg.config or checkpoint.get("config")
        model_config = _build_model_config(raw_config)
        model = GCPVQVAE(model_config).to(self.device)

        incompatible = model.load_state_dict(state_dict, strict=False)
        if hasattr(incompatible, "missing_keys") and incompatible.missing_keys:
            self.logger.warning("Missing keys when loading checkpoint: %s", incompatible.missing_keys)
        if hasattr(incompatible, "unexpected_keys") and incompatible.unexpected_keys:
            self.logger.warning(
                "Unexpected keys when loading checkpoint: %s", incompatible.unexpected_keys
            )

        model.eval()
        return model

    @staticmethod
    def _align_chain(
        predicted: Tensor,
        target: Tensor,
        mask: Tensor,
    ) -> Optional[Tensor]:
        valid = mask.to(torch.bool)
        if valid.sum() <= 0:
            return None

        src = predicted[valid].reshape(-1, 3)
        dst = target[valid].reshape(-1, 3)
        if src.shape[0] < 3 or dst.shape[0] < 3:
            return None

        rotation, translation, _ = kabsch_align(src, dst)
        aligned = predicted.reshape(-1, 3) @ rotation + translation
        return aligned.reshape_as(predicted)

    def run(self) -> Dict[str, Any]:
        model = self._load_model()
        dataloader = self._build_dataloader()
        return run_model_evaluation(
            model,
            dataloader,
            runtime_cfg=self.runtime_cfg,
            logger=self.logger,
        )


def run_model_evaluation(
    model: GCPVQVAE,
    dataloader: DataLoader,
    *,
    runtime_cfg: EvalRuntimeConfig,
    logger=None,
) -> Dict[str, Any]:
    if logger is None:
        logger = get_logger()

    rmsd_values: List[float] = []
    tm_values: List[float] = []
    gdt_values: List[float] = []
    lengths: List[float] = []
    perplexities: List[float] = []

    num_codes = getattr(model.vq, "num_codes", 0)
    code_usage = torch.zeros((num_codes,), dtype=torch.bool)

    total_residues = 0

    was_training = model.training
    model.eval()
    try:
        with torch.no_grad():
            for batch_idx, batch in enumerate(dataloader):
                outputs = model(batch)

                decoded = outputs["decoded"].detach()
                mask = outputs.get("valid_mask")
                if mask is None:
                    mask = outputs.get("mask")
                if mask is None:
                    mask = batch["mask"].to(decoded.device)
                mask = mask.to(torch.bool)

                target = batch["coords"].to(decoded.device, dtype=decoded.dtype)
                atom_mask = batch.get("atom_mask")
                if atom_mask is not None:
                    atom_mask = atom_mask.to(decoded.device).to(torch.bool)
                    residue_mask = mask & atom_mask.all(dim=-1)
                else:
                    residue_mask = mask

                finite_mask = torch.isfinite(target).all(dim=-1).all(dim=-1)
                residue_mask = residue_mask & finite_mask

                batch_size = decoded.shape[0]
                for i in range(batch_size):
                    valid = residue_mask[i]
                    if valid.sum().item() == 0:
                        continue
                    aligned = Evaluator._align_chain(decoded[i], target[i], valid)
                    if aligned is None:
                        continue

                    valid_indices = valid
                    pred_sel = aligned[valid_indices]
                    tgt_sel = target[i][valid_indices]

                    diff = pred_sel.reshape(-1, 3) - tgt_sel.reshape(-1, 3)
                    sq = torch.sum(diff**2, dim=-1)
                    rmsd_val = float(torch.sqrt(sq.mean()).cpu().item())
                    rmsd_values.append(rmsd_val)

                    length = int(valid_indices.sum().item())
                    lengths.append(float(length))
                    total_residues += length

                    if runtime_cfg.tm_score:
                        tm_val = float(tm_score(pred_sel, tgt_sel).cpu().item())
                        tm_values.append(tm_val)

                    if runtime_cfg.gdt_ts:
                        gdt_val = float(gdt_ts(pred_sel, tgt_sel).cpu().item())
                        gdt_values.append(gdt_val)

                indices = outputs.get("indices")
                if indices is not None and num_codes > 0:
                    flat = indices.detach().to(torch.long)
                    mask_indices = mask.detach()
                    flat = torch.where(mask_indices, flat, torch.full_like(flat, -1))
                    flat = flat[flat >= 0]
                    if flat.numel() > 0:
                        unique = torch.unique(flat)
                        valid_unique = unique[(unique >= 0) & (unique < num_codes)]
                        if valid_unique.numel() > 0:
                            code_usage[valid_unique.cpu()] = True

                vq_metrics = outputs.get("vq_metrics")
                if isinstance(vq_metrics, dict) and "perplexity" in vq_metrics:
                    perplexity = vq_metrics["perplexity"]
                    perplexities.append(float(perplexity.detach().cpu().item()))

                if runtime_cfg.max_batches is not None and batch_idx + 1 >= runtime_cfg.max_batches:
                    break
    finally:
        model.train(was_training)

    slope, intercept = _linear_regression(lengths, rmsd_values)
    rmsd_stats = _distribution_summary(
        rmsd_values, runtime_cfg.quantiles, runtime_cfg.histogram_bins
    )

    summary: Dict[str, Any] = {
        "num_chains": len(rmsd_values),
        "num_residues": total_residues,
        "rmsd": rmsd_stats,
        "length_vs_rmsd": {"slope": slope, "intercept": intercept},
        "codebook": {
            "num_codes": num_codes,
            "active_codes": int(code_usage.sum().item()) if num_codes > 0 else 0,
            "utilization": float(code_usage.float().mean().item()) if num_codes > 0 else 0.0,
            "perplexity_mean": float(np.mean(perplexities)) if perplexities else float("nan"),
            "perplexity_std": float(np.std(perplexities)) if perplexities else float("nan"),
        },
    }

    if tm_values:
        summary["tm_score"] = _distribution_summary(
            tm_values, runtime_cfg.quantiles, runtime_cfg.histogram_bins
        )
    if gdt_values:
        summary["gdt_ts"] = _distribution_summary(
            gdt_values, runtime_cfg.quantiles, runtime_cfg.histogram_bins
        )

    _log_distribution(logger, "RMSD", rmsd_stats)
    if tm_values:
        _log_distribution(logger, "TM-score", summary["tm_score"])
    if gdt_values:
        _log_distribution(logger, "GDT-TS", summary["gdt_ts"])

    if math.isfinite(slope):
        logger.info(
            "Length vs RMSD slope: %.3e Å/residue (intercept %.3f Å)", slope, intercept
        )
    else:
        logger.info("Insufficient data to estimate length vs RMSD trend")

    code_summary = summary["codebook"]
    utilization_pct = code_summary["utilization"] * 100.0
    logger.info(
        "Codebook utilization: %d/%d (%.2f%%) | perplexity mean=%.2f std=%.2f",
        code_summary["active_codes"],
        code_summary["num_codes"],
        utilization_pct,
        code_summary["perplexity_mean"],
        code_summary["perplexity_std"],
    )

    return summary


def evaluate(config: Mapping[str, Any] | str | Path) -> Dict[str, Any]:
    """Run evaluation using the specified configuration file."""

    return Evaluator(config).run()


__all__ = ["evaluate", "Evaluator", "run_model_evaluation"]

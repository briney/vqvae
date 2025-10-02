"""Utilities for validating and summarising experiment configuration files."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import yaml

from gcpvqvae.models.gcpvqvae import GCPVQVAE, GCPVQVAEConfig
from gcpvqvae.system.configuration import build_model_config


@dataclass
class ValidationIssue:
    """Represents a problem discovered during validation."""

    severity: str
    message: str
    path: Optional[str] = None


@dataclass
class ModelSummary:
    """Summary statistics describing the instantiated model."""

    total_parameters: int
    component_parameters: Dict[str, int]
    codebook: Dict[str, Any]


@dataclass
class TrainingSummary:
    """High level overview of the training schedule."""

    stages: List[Dict[str, Any]]


@dataclass
class ValidationReport:
    """Combined validation results and derived summaries."""

    valid: bool
    issues: List[ValidationIssue]
    raw: Dict[str, Any]
    model_config: Optional[GCPVQVAEConfig]
    model_summary: Optional[ModelSummary]
    training_summary: Optional[TrainingSummary]


def _load_yaml(path: Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)
    if not isinstance(raw, dict):
        raise ValueError("Configuration file must contain a dictionary at the top level")
    return raw


def _get_nested(mapping: Optional[Dict[str, Any]], *keys: str) -> Any:
    current: Any = mapping
    for key in keys:
        if not isinstance(current, dict) or key not in current:
            return None
        current = current[key]
    return current


def _ensure_positive(value: Any, path: str, issues: List[ValidationIssue]) -> None:
    if value is None:
        return
    if isinstance(value, (int, float)) and value <= 0:
        issues.append(ValidationIssue("error", f"Expected a positive value, got {value}", path))


def _ensure_non_negative(value: Any, path: str, issues: List[ValidationIssue]) -> None:
    if value is None:
        return
    if isinstance(value, (int, float)) and value < 0:
        issues.append(ValidationIssue("error", f"Expected a non-negative value, got {value}", path))


def _check_model_consistency(raw_model: Optional[Dict[str, Any]]) -> List[ValidationIssue]:
    issues: List[ValidationIssue] = []

    latent_dim = _get_nested(raw_model, "gcp", "latent_dim")
    encoder_input = _get_nested(raw_model, "encoder", "input_dim")
    encoder_output = _get_nested(raw_model, "encoder", "output_dim")
    decoder_input = _get_nested(raw_model, "decoder", "input_dim")
    decoder_output = _get_nested(raw_model, "decoder", "output_dim")
    decoder_model_dim = _get_nested(raw_model, "decoder", "model_dim")
    rotation_input = _get_nested(raw_model, "rotation", "input_dim")
    vq_dim = _get_nested(raw_model, "vq", "dim")
    adapter_enabled = _get_nested(raw_model, "adapter", "enabled")
    adapter_output = _get_nested(raw_model, "adapter", "output_dim")

    if latent_dim is not None:
        _ensure_positive(latent_dim, "model.gcp.latent_dim", issues)
    if adapter_enabled:
        if adapter_output is not None:
            _ensure_positive(adapter_output, "model.adapter.output_dim", issues)
        target_input = adapter_output if adapter_output is not None else vq_dim
        if (
            encoder_input is not None
            and target_input is not None
            and encoder_input != target_input
        ):
            issues.append(
                ValidationIssue(
                    "error",
                    (
                        "encoder.input_dim (%s) does not match latent adapter output (%s)"
                        % (encoder_input, target_input)
                    ),
                    "model.encoder.input_dim",
                )
            )
    elif (
        encoder_input is not None
        and latent_dim is not None
        and encoder_input != latent_dim
    ):
        issues.append(
            ValidationIssue(
                "error",
                f"encoder.input_dim ({encoder_input}) does not match gcp.latent_dim ({latent_dim})",
                "model.encoder.input_dim",
            )
        )
    if encoder_output is not None and vq_dim is not None and encoder_output != vq_dim:
        issues.append(
            ValidationIssue(
                "error",
                f"encoder.output_dim ({encoder_output}) does not match vq.dim ({vq_dim})",
                "model.encoder.output_dim",
            )
        )
    if decoder_input is not None and vq_dim is not None and decoder_input != vq_dim:
        issues.append(
            ValidationIssue(
                "error",
                f"decoder.input_dim ({decoder_input}) does not match vq.dim ({vq_dim})",
                "model.decoder.input_dim",
            )
        )
    if rotation_input is not None:
        expected = decoder_output if decoder_output is not None else decoder_model_dim
        if expected is not None and rotation_input != expected:
            issues.append(
                ValidationIssue(
                    "error",
                    f"rotation.input_dim ({rotation_input}) does not match decoder output ({expected})",
                    "model.rotation.input_dim",
                )
            )

    # Sanity checks for transformer hyper-parameters.
    for prefix in ("encoder", "decoder"):
        model_dim = _get_nested(raw_model, prefix, "model_dim")
        num_heads = _get_nested(raw_model, prefix, "num_heads")
        num_kv = _get_nested(raw_model, prefix, "num_kv_heads")
        _ensure_positive(model_dim, f"model.{prefix}.model_dim", issues)
        _ensure_positive(num_heads, f"model.{prefix}.num_heads", issues)
        _ensure_positive(num_kv, f"model.{prefix}.num_kv_heads", issues)
        if isinstance(model_dim, int) and isinstance(num_heads, int) and num_heads > 0:
            if model_dim % num_heads != 0:
                issues.append(
                    ValidationIssue(
                        "error",
                        f"model_dim ({model_dim}) must be divisible by num_heads ({num_heads})",
                        f"model.{prefix}.model_dim",
                    )
                )
            head_dim = model_dim // max(num_heads, 1)
            use_rope = _get_nested(raw_model, prefix, "use_rope")
            if (use_rope is None or bool(use_rope)) and head_dim % 2 != 0:
                issues.append(
                    ValidationIssue(
                        "warning",
                        "Rotary embeddings require an even head dimension; set use_rope=False or adjust model_dim/num_heads",
                        f"model.{prefix}.model_dim",
                    )
                )
        if isinstance(num_heads, int) and isinstance(num_kv, int) and num_heads > 0 and num_kv > 0:
            if num_heads % num_kv != 0:
                issues.append(
                    ValidationIssue(
                        "error",
                        f"num_heads ({num_heads}) must be a multiple of num_kv_heads ({num_kv})",
                        f"model.{prefix}.num_heads",
                    )
                )

    # Vector-quantiser sanity checks
    _ensure_positive(_get_nested(raw_model, "vq", "num_codes"), "model.vq.num_codes", issues)
    _ensure_positive(_get_nested(raw_model, "vq", "dim"), "model.vq.dim", issues)

    return issues


def _summarise_training(raw_train: Optional[Dict[str, Any]]) -> Tuple[Optional[TrainingSummary], List[ValidationIssue]]:
    if not isinstance(raw_train, dict):
        if raw_train is None:
            return None, []
        return None, [ValidationIssue("error", "train section must be a dictionary", "train")]

    stages_raw = raw_train.get("stages")
    if stages_raw is None:
        return TrainingSummary(stages=[]), []
    if not isinstance(stages_raw, Iterable):
        return None, [ValidationIssue("error", "train.stages must be a list", "train.stages")]

    stages: List[Dict[str, Any]] = []
    issues: List[ValidationIssue] = []

    for idx, raw_stage in enumerate(stages_raw):
        if not isinstance(raw_stage, dict):
            issues.append(ValidationIssue("error", "Each stage must be a mapping", f"train.stages[{idx}]"))
            continue

        name = raw_stage.get("name", f"stage{idx + 1}")
        batch_size = raw_stage.get("batch_size")
        length_cap = raw_stage.get("length_cap")
        base_lr = raw_stage.get("base_lr")
        min_lr = raw_stage.get("min_lr")
        total_steps = raw_stage.get("total_steps")
        epochs = raw_stage.get("epochs")
        accumulation = raw_stage.get("accumulation_steps")
        warmup = raw_stage.get("warmup_steps")

        _ensure_positive(batch_size, f"train.stages[{idx}].batch_size", issues)
        _ensure_positive(length_cap, f"train.stages[{idx}].length_cap", issues)
        _ensure_non_negative(warmup, f"train.stages[{idx}].warmup_steps", issues)
        if base_lr is not None and min_lr is not None and isinstance(base_lr, (int, float)) and isinstance(min_lr, (int, float)):
            if min_lr > base_lr:
                issues.append(
                    ValidationIssue(
                        "error",
                        f"min_lr ({min_lr}) must be less than or equal to base_lr ({base_lr})",
                        f"train.stages[{idx}].min_lr",
                    )
                )
        if total_steps is None and epochs is None:
            issues.append(
                ValidationIssue(
                    "error",
                    "Each stage requires total_steps or epochs",
                    f"train.stages[{idx}]",
                )
            )
        _ensure_positive(total_steps, f"train.stages[{idx}].total_steps", issues)
        _ensure_positive(epochs, f"train.stages[{idx}].epochs", issues)
        _ensure_positive(accumulation, f"train.stages[{idx}].accumulation_steps", issues)

        stages.append(
            {
                "name": name,
                "batch_size": batch_size,
                "length_cap": length_cap,
                "base_lr": base_lr,
                "min_lr": min_lr,
                "total_steps": total_steps,
                "epochs": epochs,
                "accumulation_steps": accumulation,
            }
        )

    return TrainingSummary(stages=stages), issues


def _build_model_summary(config: GCPVQVAEConfig) -> Tuple[Optional[ModelSummary], List[ValidationIssue]]:
    try:
        model = GCPVQVAE(config)
    except Exception as exc:  # pragma: no cover - exercised in validation tests
        return None, [ValidationIssue("error", f"Failed to instantiate model: {exc}")]

    def _count_params(module: Any) -> int:
        return sum(param.numel() for param in module.parameters())

    components = {
        "GCP encoder": _count_params(model.encoder_gcp),
        "Encoder transformer": _count_params(model.encoder_transformer),
        "Vector quantiser": _count_params(model.vq),
        "Decoder transformer": _count_params(model.decoder_transformer),
        "Rotation decoder": _count_params(model.rotation_decoder),
    }

    total = sum(components.values())

    codebook_info = {
        "num_codes": config.vq.num_codes,
        "dim": config.vq.dim,
        "beta": config.vq.beta,
        "decay": config.vq.decay,
        "orthogonal_reg_weight": config.vq.orthogonal_reg_weight,
        "orthogonal_reg_max_codes": config.vq.orthogonal_reg_max_codes,
    }

    return ModelSummary(total_parameters=total, component_parameters=components, codebook=codebook_info), []


def validate_config(config: Mapping[str, Any] | str | Path) -> ValidationReport:
    """Validate a configuration mapping or file and collect statistics."""

    if isinstance(config, Mapping):
        raw = dict(config)
    else:
        raw = _load_yaml(Path(config))
    issues: List[ValidationIssue] = []

    model_raw = raw.get("model")
    if model_raw is not None and not isinstance(model_raw, dict):
        issues.append(ValidationIssue("error", "model section must be a dictionary", "model"))
        model_config = None
        model_summary = None
    else:
        issues.extend(_check_model_consistency(model_raw))
        model_config = build_model_config(model_raw)
        summary, model_issues = _build_model_summary(model_config)
        issues.extend(model_issues)
        model_summary = summary

    training_summary, train_issues = _summarise_training(raw.get("train"))
    issues.extend(train_issues)

    valid = all(issue.severity != "error" for issue in issues)

    return ValidationReport(
        valid=valid,
        issues=issues,
        raw=raw,
        model_config=model_config,
        model_summary=model_summary,
        training_summary=training_summary,
    )


def format_report(report: ValidationReport) -> str:
    """Format a :class:`ValidationReport` into a human readable summary."""

    lines: List[str] = []
    status = "VALID" if report.valid else "INVALID"
    lines.append("Configuration Summary")
    lines.append("=====================")
    lines.append(f"Status: {status}")

    lines.append("")
    lines.append("Issues:")
    if not report.issues:
        lines.append("  (none)")
    else:
        for issue in report.issues:
            location = f" {issue.path}" if issue.path else ""
            lines.append(f"  - [{issue.severity.upper()}]{location}: {issue.message}")

    if report.model_config is not None:
        cfg = report.model_config
        lines.append("")
        lines.append("Model Dimensions:")
        lines.append(f"  GCP latent dim: {cfg.gcp.latent_dim}")
        lines.append(f"  Encoder model dim: {cfg.encoder.model_dim}")
        lines.append(f"  Decoder model dim: {cfg.decoder.model_dim}")
        lines.append(f"  Latent code dim: {cfg.vq.dim}")

    if report.model_summary is not None:
        summary = report.model_summary
        lines.append("")
        lines.append("Model Parameters:")
        lines.append(f"  Total: {summary.total_parameters:,}")
        for name, count in summary.component_parameters.items():
            lines.append(f"  {name}: {count:,}")
        lines.append("")
        lines.append("Codebook:")
        for key, value in summary.codebook.items():
            lines.append(f"  {key}: {value}")

    if report.training_summary is not None and report.training_summary.stages:
        lines.append("")
        lines.append("Training Stages:")
        for stage in report.training_summary.stages:
            name = stage.get("name", "(unnamed)")
            total = stage.get("total_steps")
            epochs = stage.get("epochs")
            lines.append(f"  - {name}:")
            lines.append(f"      batch_size: {stage.get('batch_size')}")
            lines.append(f"      length_cap: {stage.get('length_cap')}")
            if total is not None:
                lines.append(f"      total_steps: {total}")
            if epochs is not None:
                lines.append(f"      epochs: {epochs}")
            lines.append(f"      base_lr: {stage.get('base_lr')}")
            lines.append(f"      min_lr: {stage.get('min_lr')}")
            lines.append(f"      accumulation_steps: {stage.get('accumulation_steps')}")

    return "\n".join(lines)


__all__ = [
    "ValidationIssue",
    "ModelSummary",
    "TrainingSummary",
    "ValidationReport",
    "validate_config",
    "format_report",
]


"""Command line interface entry points for the GCP-VQVAE toolkit."""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple

import click

from gcpvqvae.system.configuration import (
    DEFAULT_GCPNET_PRETRAIN_CONFIG_PATH,
    DEFAULT_TRAIN_CONFIG_PATH,
    compose_overrides,
)

try:  # Python 3.10+
    from importlib.metadata import PackageNotFoundError, version
except ImportError:  # pragma: no cover - fallback for very old Python
    from importlib_metadata import PackageNotFoundError, version  # type: ignore

_CONTEXT_SETTINGS = {"help_option_names": ["-h", "--help"]}

try:
    _PACKAGE_VERSION = version("vqvae")
except PackageNotFoundError:
    _PACKAGE_VERSION = "0.0.0"


def _feature_unavailable(feature: str) -> None:
    """Raise a friendly error for CLI features that are not yet implemented."""

    raise click.ClickException(
        f"{feature} is not yet available from the command line. "
        "Use the Python API for now, or keep an eye on future releases."
    )


@click.group(name="gpcvq", context_settings=_CONTEXT_SETTINGS)
@click.version_option(_PACKAGE_VERSION, prog_name="gpcvq")
def gpcvq() -> None:
    """Geometry-complete protein VQ-VAE command line interface.

    The CLI centralizes the major workflows for training, encoding, decoding, and
    evaluating models.  Run ``gpcvq <command> --help`` for details on any command.
    """


@gpcvq.command(
    name="preprocess-dataset",
    short_help="Precompute and cache dataset features.",
    help=(
        "Parse INPUT (a backbone file or directory) and write a cached representation "
        "to OUTPUT. The resulting directory can be supplied to other commands in place "
        "of the original structure files."
    ),
)
@click.argument(
    "input",
    type=click.Path(exists=True, path_type=Path),
    metavar="INPUT",
)
@click.argument(
    "output",
    type=click.Path(path_type=Path),
    metavar="OUTPUT",
)
@click.option(
    "--max-len",
    type=int,
    metavar="N",
    help="Discard chains longer than N residues.",
)
@click.option(
    "--min-len",
    type=int,
    metavar="N",
    help="Discard chains shorter than N residues.",
)
@click.option(
    "--max-workers",
    type=int,
    metavar="N",
    help="Maximum worker processes to use while preprocessing.",
)
@click.option(
    "--no-file-index",
    is_flag=True,
    help="Skip writing the auxiliary file index alongside OUTPUT.",
)
@click.option(
    "--gap-threshold",
    type=float,
    metavar="Å",
    help="Maximum residue-residue gap (in Å) before splitting chains.",
)
def preprocess_dataset_command(
    input: Path,
    output: Path,
    max_len: Optional[int],
    min_len: Optional[int],
    max_workers: Optional[int],
    no_file_index: bool,
    gap_threshold: Optional[float],
) -> None:
    """Preprocess raw backbone data for faster reuse.

    Args:
        input: Path to a backbone file or directory containing structures.
        output: Destination directory for cached tensors and manifest.
        max_len: Optional length cap applied during preprocessing.
        min_len: Optional minimum length filter.
        max_workers: Maximum number of worker processes to spawn.
        no_file_index: Skip generating ``file_index.json`` when ``True``.
        gap_threshold: Maximum gap distance (Å) before splitting chains.
    """

    if max_len is not None and max_len <= 0:
        raise click.ClickException("--max-len must be a positive integer.")
    if min_len is not None and min_len <= 0:
        raise click.ClickException("--min-len must be a positive integer.")
    if (
        max_len is not None
        and min_len is not None
        and min_len > max_len
    ):
        raise click.ClickException("--min-len cannot exceed --max-len.")
    if max_workers is not None and max_workers <= 0:
        raise click.ClickException("--max-workers must be a positive integer.")
    if gap_threshold is not None and gap_threshold <= 0:
        raise click.ClickException("--gap-threshold must be positive.")

    from gcpvqvae.data.preprocess import preprocess_dataset

    try:
        result = preprocess_dataset(
            input,
            output,
            max_len=max_len,
            min_len=min_len,
            max_workers=max_workers,
            file_index=not no_file_index,
            gap_threshold=gap_threshold,
        )
    except (OSError, ValueError) as exc:
        raise click.ClickException(str(exc)) from exc

    if isinstance(result, tuple):
        manifest_path, stats = result
    else:  # pragma: no cover - backwards compatibility
        manifest_path = result
        stats = None

    if stats:
        click.echo(f"Summary: {stats}")
    click.echo(f"Preprocessed dataset written to {manifest_path}")


@gpcvq.command(
    name="train",
    short_help="Train a model using an optional YAML configuration file.",
    help=(
        "Train a GCP-VQVAE model. By default the command loads the settings from "
        "the packaged ``base.yaml`` configuration.  Provide ``--config`` to use an "
        "alternative file.  Additional overrides can be supplied using Hydra's "
        "dotted ``key=value`` syntax."
    ),
)
@click.option(
    "--config",
    "config_path",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    help="Path to a YAML configuration file (defaults to base.yaml).",
)
@click.argument("overrides", nargs=-1, metavar="[OVERRIDE]...", type=str)
def train_command(config_path: Optional[Path], overrides: Tuple[str, ...]) -> None:
    """Kick off model training with the provided configuration file.

    Args:
        config_path: Optional path to a YAML training configuration.
        overrides: Hydra-style ``key=value`` overrides appended on the command line.
    """

    from gcpvqvae.system.train import train as run_train

    config_source = config_path if config_path is not None else DEFAULT_TRAIN_CONFIG_PATH
    raw_config = compose_overrides(config_source, overrides)
    run_train(raw_config)


@gpcvq.command(
    name="train-gpcnet",
    short_help="Pretrain the standalone GCPNet encoder.",
    help=(
        "Run GCPNet pretraining.  The command defaults to the packaged "
        "``gcpnet_pretrain.yaml`` configuration but accepts ``--config`` to specify "
        "an alternative file.  Hydra-style overrides may be appended using "
        "``key=value`` syntax."
    ),
)
@click.option(
    "--config",
    "config_path",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    help="Path to a YAML configuration file (defaults to gcpnet_pretrain.yaml).",
)
@click.argument("overrides", nargs=-1, metavar="[OVERRIDE]...", type=str)
def train_gpcnet_command(config_path: Optional[Path], overrides: Tuple[str, ...]) -> None:
    """Launch GCPNet-only pretraining from the CLI.

    Args:
        config_path: Optional path to the GCPNet pretraining configuration.
        overrides: Hydra-style ``key=value`` overrides appended on the command line.
    """

    from gcpvqvae.system.train_gcpnet import train as run_train_gcpnet

    config_source = (
        config_path if config_path is not None else DEFAULT_GCPNET_PRETRAIN_CONFIG_PATH
    )
    raw_config = compose_overrides(config_source, overrides)
    run_train_gcpnet(raw_config)


@gpcvq.command(
    name="encode",
    short_help="Encode backbone structures into discrete tokens.",
    help=(
        "Encode protein backbone structures into the model's vector-quantized "
        "token representation.  Provide either a single mmCIF/PDB file or a "
        "directory containing structures.  Results are written to OUTPUT as a "
        "NumPy archive containing the tokens and associated metadata. "
        "\n\nNote: the public CLI for encoding is not yet implemented.  Invoke the "
        "Python API (``GCPVQVAE.encode``) for the time being."
    ),
)
@click.argument(
    "input_path",
    type=click.Path(dir_okay=True, path_type=Path),
    metavar="INPUT",
)
@click.option(
    "--output",
    "output_path",
    type=click.Path(dir_okay=False, path_type=Path),
    help="Location to store the encoded tokens (defaults to INPUT with a .npz suffix).",
)
@click.option(
    "--chain-id",
    type=str,
    help="Restrict encoding to a specific chain identifier (optional).",
)
def encode_command(
    input_path: Path, output_path: Optional[Path], chain_id: Optional[str]
) -> None:
    """Placeholder implementation for the encode sub-command.

    Args:
        input_path: File or directory containing structures to encode.
        output_path: Optional location for the token archive.
        chain_id: Optional chain identifier filter.
    """

    _feature_unavailable("Encoding")


@gpcvq.command(
    name="decode",
    short_help="Reconstruct coordinates from discrete tokens.",
    help=(
        "Decode previously generated VQ token sequences back into Cartesian backbone "
        "coordinates.  TOKENS should reference a NumPy archive produced by the "
        "encoder.  The decoded backbone will be written to OUTPUT in mmCIF format. "
        "\n\nNote: CLI decoding is not yet available; please use the Python API "
        "instead."
    ),
)
@click.argument(
    "tokens",
    type=click.Path(dir_okay=False, path_type=Path),
    metavar="TOKENS",
)
@click.option(
    "--output",
    "output_path",
    type=click.Path(dir_okay=False, path_type=Path),
    help="Destination mmCIF file for reconstructed coordinates.",
)
def decode_command(tokens: Path, output_path: Optional[Path]) -> None:
    """Placeholder implementation for the decode sub-command.

    Args:
        tokens: Path to a compressed token archive.
        output_path: Optional destination for reconstructed coordinates.
    """

    _feature_unavailable("Decoding")


@gpcvq.command(
    name="eval",
    short_help="Evaluate a trained checkpoint.",
    help=(
        "Evaluate a trained model checkpoint using the experiment CONFIG file. "
        "The configuration should describe the dataset split, checkpoint to load, "
        "and metrics to compute.\n\n"
        "Overrides can be appended after CONFIG using Hydra's dotted "
        "``key=value`` syntax."
    ),
)
@click.argument(
    "config",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    metavar="CONFIG",
)
@click.argument("overrides", nargs=-1, metavar="[OVERRIDE]...", type=str)
def eval_command(config: Path, overrides: Tuple[str, ...]) -> None:
    """Run model evaluation for a specified configuration.

    Args:
        config: Path to the evaluation YAML file.
        overrides: Hydra-style overrides applied before evaluation.
    """

    from gcpvqvae.system.eval import evaluate

    try:
        raw_config = compose_overrides(config, overrides)
        evaluate(raw_config)
    except NotImplementedError as exc:  # pragma: no cover - pending implementation
        raise click.ClickException(str(exc)) from exc


@gpcvq.command(
    name="validate-config",
    short_help="Validate and summarise a configuration file.",
    help=(
        "Inspect CONFIG for common issues and display model statistics. "
        "The validator checks for incompatible dimensions between sub-modules, "
        "invalid hyper-parameter settings, and reports the parameter counts for "
        "each major component.\n\n"
        "Overrides supplied after CONFIG (using Hydra's ``key=value`` syntax) "
        "will be validated as well."
    ),
)
@click.argument(
    "config",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    metavar="CONFIG",
)
@click.argument("overrides", nargs=-1, metavar="[OVERRIDE]...", type=str)
def validate_config_command(config: Path, overrides: Tuple[str, ...]) -> None:
    """Validate a configuration file and print a detailed report.

    Args:
        config: Path to the configuration YAML file.
        overrides: Hydra-style overrides evaluated alongside ``config``.
    """

    from gcpvqvae.system.config_validation import format_report, validate_config

    raw_config = compose_overrides(config, overrides)
    report = validate_config(raw_config)
    click.echo(format_report(report))
    if not report.valid:
        raise click.ClickException("Configuration validation failed.")


def main() -> None:
    """Entry-point used by setuptools console scripts."""

    gpcvq()  # pylint: disable=no-value-for-parameter


if __name__ == "__main__":  # pragma: no cover
    main()

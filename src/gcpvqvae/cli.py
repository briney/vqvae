"""Command line interface entry points for the GCP-VQVAE toolkit."""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple

import click

from gcpvqvae.system.configuration import compose_overrides

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
    "--chain-id",
    "chain_ids",
    multiple=True,
    help="Restrict preprocessing to specific chain identifiers.",
)
@click.option(
    "--length-cap",
    type=int,
    default=2048,
    show_default=True,
    help="Maximum sequence length to load from the source data.",
)
@click.option(
    "--k",
    type=int,
    default=16,
    show_default=True,
    help="Number of nearest neighbours to use when computing geometric features.",
)
@click.option(
    "--overwrite/--no-overwrite",
    default=False,
    show_default=True,
    help="Overwrite OUTPUT if it already exists.",
)
@click.option(
    "--progress/--no-progress",
    "show_progress",
    default=True,
    show_default=True,
    help="Display progress while preprocessing.",
)
def preprocess_dataset_command(
    input: Path,
    output: Path,
    chain_ids: Tuple[str, ...],
    length_cap: int,
    k: int,
    overwrite: bool,
    show_progress: bool,
) -> None:
    """Preprocess raw backbone data for faster reuse."""

    from gcpvqvae.data.preprocessing import preprocess_dataset

    manifest_path = preprocess_dataset(
        input,
        output,
        chain_ids=chain_ids if chain_ids else None,
        length_cap=length_cap,
        k=k,
        overwrite=overwrite,
        progress=show_progress,
    )
    click.echo(f"Preprocessed dataset written to {manifest_path}")


@gpcvq.command(
    name="train",
    short_help="Train a model from a YAML configuration file.",
    help=(
        "Train a GCP-VQVAE model using the settings defined in CONFIG. "
        "The configuration file should provide ``data``, ``model``, and ``train`` "
        "sections as described in :mod:`gcpvqvae.system.train`.  Template files are "
        "available under ``src/gcpvqvae/configs``.\n\n"
        "Additional overrides can be supplied after CONFIG using Hydra's dotted "
        "``key=value`` syntax."
    ),
)
@click.argument(
    "config",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    metavar="CONFIG",
)
@click.argument("overrides", nargs=-1, metavar="[OVERRIDE]...", type=str)
def train_command(config: Path, overrides: Tuple[str, ...]) -> None:
    """Kick off model training with the provided configuration file."""

    from gcpvqvae.system.train import train as run_train

    raw_config = compose_overrides(config, overrides)
    run_train(raw_config)


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
    """Placeholder implementation for the encode sub-command."""

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
    """Placeholder implementation for the decode sub-command."""

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
    """Run model evaluation for a specified configuration."""

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
    """Validate a configuration file and print a detailed report."""

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

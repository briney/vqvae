# VQ-VAE

Clean PyTorch implementation of the geometry-complete protein VQ-VAE (GCP-VQVAE).

## Features

- GCPNet encoder with equivariant micro-steps and gating for protein backbones
- Transformer context module with vector-quantized latent tokens
- 6D rotation decoder with rigid reconstruction losses
- Click-powered CLI tooling for training, encoding, decoding, and evaluation workflows

## Installation

We recommend working inside a fresh Python environment (3.9 or newer).

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -e .
```

## Command line interface

Installing the package registers a single entry point named `gpcvq`. Run `gpcvq --help`
for a full overview of the available subcommands and global options.

```bash
gpcvq --help
```

### Training models

Use the `train` subcommand to launch an experiment from a YAML configuration file.
The configuration is expected to follow the schema used by
`gcpvqvae.system.train`, with top-level `data`, `model`, and `train` sections.
Template files are available under `src/gcpvqvae/configs`.

```bash
# Train using a configuration file in the repository
# (adjust the path to point at your desired template)
gpcvq train src/gcpvqvae/configs/base.yaml
```

You can inspect the supported options with `gpcvq train --help`:

```text
Usage: gpcvq train [OPTIONS] CONFIG

  Train a GCP-VQVAE model using the settings defined in CONFIG. The
  configuration file should provide ``data``, ``model``, and ``train``
  sections as described in :mod:`gcpvqvae.system.train`. Template files are
  available under ``src/gcpvqvae/configs``.
```

### Evaluating checkpoints

Once training has produced a checkpoint, the `eval` subcommand can be used to
compute metrics defined in an evaluation configuration. The schema mirrors the
training configuration.

```bash
gpcvq eval path/to/eval_config.yaml
```

If evaluation encounters an unimplemented feature, the command will exit with a
human-friendly error explaining the limitation.

### Encoding and decoding structures

The CLI includes placeholders for future encoding and decoding workflows:

```bash
gpcvq encode structure.cif --output tokens.npz
gpcvq decode tokens.npz --output rebuilt_structure.cif
```

At present these commands will raise an informative error directing you to use
the Python API while the CLI implementations are finalized.

## Development

After cloning the repository you can run the test suite to verify your setup.

```bash
pytest
```

## License

This project is provided under the MIT License. See `LICENSE` for details.

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

### Preprocessing datasets

Large training runs benefit from caching dataset features to disk ahead of
time. The `preprocess-dataset` subcommand materialises the contents of an input
mmCIF/PDB directory (or single file) into a reusable format:

```bash
gpcvq preprocess-dataset path/to/raw_structures path/to/preprocessed \
    --length-cap 2048 --k 16
```

Running the command creates `path/to/preprocessed` containing:

- `preprocessed_dataset.json` – a manifest describing the dataset version,
  source path, preprocessing parameters (`length_cap`, `k`, and any
  `chain_id` filters), total number of samples, and an `entries` list. Each
  entry records the relative file name of the saved sample, its originating
  structure path, chain identifier, primary sequence (when available), and the
  residue count used during preprocessing.
- `samples/XXXXXXXX.pt` – a directory of PyTorch files storing the featurised
  chains. Each file contains the same dictionary of tensors produced during
  on-the-fly loading, including coordinates, masks, node/edge features,
  torsion angles, rigid-frame poses, and associated metadata.

These cached datasets can be supplied to other CLI commands in place of the raw
structures, eliminating redundant featurisation work.

### Training models

Use the `train` subcommand to launch an experiment from a YAML configuration file.
Configurations follow a simple schema with top-level `data`, `model`, and `train`
sections.  A detailed reference of every option and its default value is
available in [`src/gcpvqvae/configs/README.md`](src/gcpvqvae/configs/README.md).
Starter templates live alongside that document.

```bash
# Train using a configuration file in the repository
# (adjust the path to point at your desired template)
gpcvq train src/gcpvqvae/configs/base.yaml
```

To train from a preprocessed dataset, point `data.root` at the directory
containing `preprocessed_dataset.json`. This can be done directly in the YAML
file or via a command-line override:

```bash
gpcvq train src/gcpvqvae/configs/base.yaml data.root=path/to/preprocessed
```

Any parameter can be overridden directly from the CLI using Hydra's dotted
syntax.  Append `section.key=value` pairs after the config path to tweak
experiments without editing files:

```bash
gpcvq train src/gcpvqvae/configs/small.yaml \
    train.stages[0].batch_size=8 \
    model.vq.num_codes=128
```

You can inspect the supported options with `gpcvq train --help`:

```text
Usage: gpcvq train [OPTIONS] CONFIG

  Train a GCP-VQVAE model using the settings defined in CONFIG. The
  configuration file should provide ``data``, ``model``, and ``train``
  sections as described in :mod:`gcpvqvae.system.train`. Template files are
  available under ``src/gcpvqvae/configs``.
```

### Experiment logging with Weights & Biases

Training jobs can stream metrics to [Weights & Biases](https://wandb.ai/site) by
setting the `train.log` section of your configuration.  Enable logging and
optionally supply a project, entity, run name, tags, working directory, or mode:

```yaml
train:
  log:
    enabled: true
    project: gcp-vqvae
    run_name: baseline-l1
    tags: [prototype, local]
```

When activated the trainer initialises a run with the provided metadata and
automatically reports the total loss, its reconstruction and vector-quantisation
components, reconstruction sub-metrics, RMSD, codebook utilisation statistics,
learning-rate schedule, and throughput figures each time progress is logged.

### Loss components

The model optimises a composite loss consisting of a weighted reconstruction
term plus several vector-quantisation regularisers:

- **Reconstruction loss** – combines three geometry-aware signals following the
  GCP-VQVAE paper: an aligned mean-squared error between predicted and target
  backbones, a pairwise backbone distance penalty, and a backbone direction
  signature loss. These components are weighted `(5e-3, 1e-2, 5e-2)` to balance
  local fidelity with global structure alignment.
- **VQ commitment loss** – encourages encoder outputs to remain close to their
  assigned codebook vectors so that quantisation remains stable.
- **VQ codebook loss** – pulls codebook entries toward the current encoder
  outputs, ensuring the discrete latent space tracks the data manifold.
- **VQ orthogonality loss** – optional regulariser promoting diverse, nearly
  orthogonal code vectors to improve codebook utilisation.

The total loss is the sum of these terms, and all components are logged when
Weights & Biases tracking is enabled to simplify debugging and comparisons.

### Evaluating checkpoints

Once training has produced a checkpoint, the `eval` subcommand can be used to
compute metrics defined in an evaluation configuration. The schema mirrors the
training configuration.

```bash
gpcvq eval path/to/eval_config.yaml
```

Preprocessed datasets are also valid evaluation inputs. Override the evaluation
configuration to reference the cached directory when needed:

```bash
gpcvq eval path/to/eval_config.yaml data.root=path/to/preprocessed
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

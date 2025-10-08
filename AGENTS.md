# Repository Guidelines

## Project Structure & Module Organization
- `src/gcpvqvae/cli.py` exposes the `gpcvq` Click entry point; subpackages (`data`, `geometry`, `models`, `system`, `utils`) hold the dataset pipeline, SE(3)-aware layers, model blocks, trainers, and helpers.
- `src/gcpvqvae/configs/` stores Hydra-friendly YAML templates for training and encoder pretraining; keep new configs beside the existing `base.yaml` set.
- `tests/` mirrors the runtime modules and ships fixtures in `tests/test_data/`; read `tests/README.md` before adding coverage.

## Build, Test, and Development Commands
- Bootstrap a dev environment with `python -m venv .venv && source .venv/bin/activate && pip install -e .[viz]`.
- Run the suite with `pytest`; narrow scope during iteration via `pytest tests/test_data_pipeline.py -k backbone`.
- Exercise the CLI for end-to-end checks: `gpcvq preprocess-dataset path/to/raw path/to/preprocessed` and `gpcvq train --config src/gcpvqvae/configs/small.yaml data.root=...`.

## Coding Style & Naming Conventions
- Stick to Python 3.10+, 4-space indentation, and explicit type hints (`Tensor`, `BackboneRecord`, etc.) as shown in `src/gcpvqvae/data/dataset.py`.
- Keep modules `snake_case`, classes `CamelCase`, constants `UPPER_SNAKE_CASE`, and reuse existing helper names when extending components.
- Group imports by standard library, third-party, then local modules; preserve module-level docstrings that summarise the file’s intent.

## Testing Guidelines
- Pytest is the reference framework; new features need unit or integration tests under the matching subsystem directory plus updates to `tests/README.md` when coverage shifts.
- Reuse fixtures in `tests/test_data/` and assert tensor shapes, masks, and dtype expectations to guard geometry regressions.
- Run `pytest --maxfail=1 --disable-warnings` before opening a PR and include a CLI smoke test when changing preprocessing or configuration surfaces.

## Commit & Pull Request Guidelines
- Follow the existing history: use short, imperative commit subjects (`Refine dataset trimming`, `Add CLI validation guard`) and include focused changes per commit.
- Pull requests should summarise behaviour changes, reference the relevant issue or configuration, list the executed commands (e.g., `pytest`, `gpcvq train ...`), and attach logs or screenshots when altering CLI output.
- Ensure backwards compatibility for Hydra configs by documenting new keys and providing defaults so older templates remain valid.

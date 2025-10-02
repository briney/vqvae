# Configuration reference

This directory contains Hydra-compatible YAML configuration files used by the
GCP-VQVAE command line interface and training utilities.  A configuration file
is organised into three top-level sections:

- `data`: how structures are loaded and featurised
- `model`: architecture hyper-parameters for the encoder, VQ layer, decoder and
  auxiliary heads
- `train`: optimiser-free training loop settings, logging configuration, and stage schedule

Hydra allows any value to be overridden at runtime using dotted `section.key`
notation (see the project-wide `README.md` for examples).  When a field is
omitted the defaults listed below are applied by the underlying dataclasses.

## `data` section

| key | type | default | description |
| --- | --- | --- | --- |
| `root` | str | **required** | Path to raw backbone data (single mmCIF/PDB file or directory) **or** to a preprocessed dataset directory containing `preprocessed_dataset.json`. |
| `chain_ids` | sequence[str] \| null | `null` | Optional subset of chain identifiers to keep. |
| `k` | int | `16` | Number of nearest neighbours per residue when building the kNN graph. |
| `num_dataloader_workers` | int | `0` | DataLoader worker processes. |
| `show_progress` | bool | `True` | Display a progress bar while preprocessing structures. |
| `cache` | bool | `True` | Whether to cache parsed mmCIF records on disk. |

## `model` section

### `model.gcp` – GCP encoder (`GCPNetConfig`)

| key | default | description |
| --- | --- | --- |
| `node_scalar_dim` | `6` | Input scalar feature channels per residue. |
| `node_vector_dim` | `3` | Input vector feature channels per residue. |
| `edge_scalar_dim` | `8` | Scalar features attached to each edge. |
| `edge_scalar_input_dim` | `null` | Optional raw dimensionality of edge scalar features (defaults to `edge_scalar_dim`). |
| `edge_vector_dim` | `1` | Vector features per edge. |
| `hidden_scalar_dim` | `128` | Scalar width of the hidden representations. |
| `hidden_vector_dim` | `16` | Vector channel count inside the GCP blocks. |
| `latent_dim` | `256` | Output embedding dimension fed to the Transformer. |
| `layers` | `6` | Number of stacked GCP convolution layers. |
| `dropout` | `0.0` | Dropout applied within the convolutions. |
| `displacement_head` | `False` | Enables an auxiliary displacement prediction head. |
| `init` | `"random"` | Select `"random"` for fresh weights or `"pretrained"` to load from a checkpoint. |
| `init_checkpoint` | `null` | Filesystem path to the checkpoint containing pretrained GCPNet weights. |
| `strict_init` | `True` | Whether to enforce an exact key match when loading weights. |

### `model.adapter` – Latent projection (`LatentAdapterConfig`)

| key | default | description |
| --- | --- | --- |
| `enabled` | `False` | Enables the linear adapter between the GCP encoder and Transformer stack. |
| `output_dim` | `null` | Target dimensionality for the adapter projection (defaults to `model.vq.dim` when omitted). |
| `bias` | `False` | Adds a bias term to the projection layer when set to `True`. |

### `model.encoder` and `model.decoder` – Transformer stacks (`TransformerConfig`)

| key | default | description |
| --- | --- | --- |
| `input_dim` | derived | Automatically set to match the upstream module (`latent_dim` for the encoder, VQ dimension for the decoder). |
| `model_dim` | `1024` | Hidden width of the Transformer blocks. |
| `output_dim` | `null` | Optional projection size (defaults to `model_dim`). |
| `num_layers` | `12` | Number of Transformer blocks. |
| `num_heads` | `12` | Attention heads per block. |
| `num_kv_heads` | `3` | Key/value heads (for grouped-query attention). |
| `dropout` | `0.0` | Dropout applied after each block and on the final output. |
| `ffn_multiplier` | `4.0` | Multiplier controlling the feed-forward hidden size. |
| `use_rope` | `True` | Enables rotary positional embeddings. |

### `model.vq` – Vector-quantiser (`VectorQuantizerConfig`)

| key | default | description |
| --- | --- | --- |
| `num_codes` | `4096` | Size of the codebook. |
| `dim` | `256` | Codebook embedding dimension. |
| `beta` | `0.25` | Commitment loss weight. |
| `decay` | `0.99` | EMA decay for codebook updates. |
| `epsilon` | `1e-5` | Numerical stability constant. |
| `kmeans_iters` | `10` | Initial K-means refinement iterations. |
| `rotation_trick` | `True` | Enables the rotation trick from VQ-VAE v2. |
| `orthogonal_reg_weight` | `0.0` | Strength of the optional orthogonality penalty. |
| `orthogonal_reg_max_codes` | `512` | Maximum codes used when computing the penalty. |

### `model.rotation` – Rigid-frame decoder (`RotationHeadConfig`)

| key | default | description |
| --- | --- | --- |
| `input_dim` | derived | Automatically matches the decoder output width. |
| `translation_scale` | `1.0` | Scale factor applied to predicted translations. |
| `template` | `null` | Optional tensor describing a template structure. |

### `model.data` – Training-time preprocessing (`DataPipelineConfig`)

| key | default | description |
| --- | --- | --- |
| `length_cap` | `2048` | Maximum residues per chain fed to the model. |
| `knn` | `16` | Neighbours per residue when re-constructing edges during training. |

## `train` section

### Top-level training settings (`TrainConfig`)

| key | default | description |
| --- | --- | --- |
| `seed` | `42` | Seed for PyTorch, NumPy, and Python RNGs. |
| `device` | `null` | Forces training onto a specific device identifier (defaults to CUDA if available). |
| `amp` | `True` | Enables automatic mixed precision. |
| `clip_grad` | `1.0` | Gradient clipping value (L2 norm). |
| `random_rotation` | `True` | Applies random SO(3) augmentation to each batch. |
| `checkpoint_interval` | `null` | Frequency (in steps) for saving checkpoints; `null` disables periodic checkpoints. |
| `output_dir` | `"runs"` | Base directory for logs, checkpoints, and exports. |
| `log` | see below | Nested configuration controlling experiment logging backends. |
| `export` | see below | Nested configuration controlling structure exports. |
| `stages` | `[]` | List of training stage dictionaries (`StageConfig`). |

### `train.log` (`LogConfig`)

Enable and customise Weights & Biases tracking for a run.  All fields are optional
except `enabled`; CLI overrides such as `train.log.project=my-project` update the
values shown below.

| key | default | description |
| --- | --- | --- |
| `enabled` | `False` | Toggle logging to Weights & Biases. |
| `project` | `null` | W&B project name (required when logging is enabled). |
| `entity` | `null` | Optional W&B entity/organisation. |
| `run_name` | `null` | Custom run display name. |
| `tags` | `[]` | Optional list of tags applied to the run. |
| `dir` | `null` | Directory used for W&B file artefacts (defaults to `<output_dir>`). |
| `mode` | `null` | Advanced W&B init mode (`online`, `offline`, etc.). |
| `interval` | `50` | Number of optimisation steps between progress log updates. |

### `train.export` (`ExportConfig`)

| key | default | description |
| --- | --- | --- |
| `enabled` | `True` | Toggles structure export. |
| `directory` | `null` | Output directory for exported structures (defaults to `<output_dir>/exports`). |
| `every_n_steps` | `null` | If set, export samples every `n` optimisation steps. |
| `on_stage_end` | `True` | Export a batch at the end of each stage. |
| `num_samples` | `1` | Number of structures to export at each trigger. |

### `train.stages[]` (`StageConfig`)

Each stage entry orchestrates one phase of the curriculum.  Provide either
`total_steps` **or** `epochs`:

| key | default | description |
| --- | --- | --- |
| `name` | **required** | Stage identifier used in logs. |
| `length_cap` | **required** | Maximum residues per training sample during the stage. |
| `batch_size` | **required** | Number of chains per optimisation step. |
| `base_lr` | **required** | Peak learning rate reached after warmup. |
| `min_lr` | **required** | Minimum learning rate used at the start/end of cosine decay. |
| `warmup_steps` | **required** | Linear warmup duration in optimisation steps. |
| `total_steps` | `null` | Number of optimisation steps to run (mutually exclusive with `epochs`). |
| `epochs` | `null` | Number of dataset passes (used when `total_steps` is omitted). |
| `accumulation_steps` | `1` | Gradient accumulation factor. |
| `nan_mask_prob` | `0.0` | Probability of applying random NaN masking augmentation. |
| `nan_mask_span` | `[1, 1]` | Range for contiguous residue spans used during NaN masking. |

## Selecting a template

Example templates are provided:

- `base.yaml`: full-sized training schedule mirroring the manuscript.
- `small.yaml`: reduced footprint configuration for local experiments, reusing the base GCPNet with a latent adapter to shrink the Transformer.
- `xsmall.yaml`: minimal configuration matching the continuous integration tests, also using the latent adapter for compact Transformer/VQ settings.
- `gcpnet_pretrain.yaml`: lightweight schedule for pretraining the encoder in isolation.

Feel free to copy these files and customise them using the options described
above.

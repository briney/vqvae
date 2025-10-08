# Test Suite Overview

This document summarises the automated tests that exercise the GCP-VQVAE codebase. The tests are grouped by the major subsystem they validate, with each entry describing the intent of the individual checks.

## CLI and Configuration Validation

### `tests/test_cli_config_validation.py`
- `test_validate_config_reports_success` – Runs the `gpcvq validate-config` CLI command against a representative configuration and asserts that validation succeeds while reporting model details. ([source](test_cli_config_validation.py#L80-L102))
- `test_validate_config_flags_dimension_mismatch` – Confirms the CLI surfaces an error when incompatible model dimensions are specified in the configuration file. ([source](test_cli_config_validation.py#L105-L124))

### `tests/test_cli_preprocess.py`
- `test_preprocess_dataset_help_lists_reference_options` – Verifies that the preprocessing CLI subcommand documents the key command-line flags in its help output. ([source](test_cli_preprocess.py#L12-L33))
- `test_preprocess_dataset_invokes_reference_driver` – Checks that invoking the CLI routes arguments to the dataset preprocessing driver and prints a summary of the work performed. ([source](test_cli_preprocess.py#L36-L86))
- `test_preprocess_dataset_reports_invalid_option_combinations` – Ensures invalid option combinations are rejected with informative error messages. ([source](test_cli_preprocess.py#L89-L117))

### `tests/test_configuration_overrides.py`
- `test_cli_overrides_update_logging_block` – Validates that CLI-provided overrides can toggle logging settings in a base YAML configuration. ([source](test_configuration_overrides.py#L18-L39))
- `test_default_train_config_matches_base_yaml` – Confirms the packaged default training configuration loads exactly as written. ([source](test_configuration_overrides.py#L42-L49))
- `test_default_gcpnet_config_matches_packaged_yaml` – Ensures the reference GCPNet pretraining configuration matches the packaged YAML template. ([source](test_configuration_overrides.py#L52-L60))

## Data Ingestion and Preprocessing

### `tests/test_data_pipeline.py`
- `test_load_mmcif_parses_backbone` – Exercises backbone parsing from mmCIF/PDB files, confirming residue masks, canonical filtering, and coordinate centring. ([source](test_data_pipeline.py#L87-L136))
- `test_load_mmcif_filters_noncanonical_residues` – Verifies that non-canonical residues are excluded when loading structures. ([source](test_data_pipeline.py#L139-L183))
- `test_featurize_backbone_produces_expected_shapes` – Checks backbone featurisation produces normalised node and edge features with consistent shapes. ([source](test_data_pipeline.py#L186-L229))
- `test_backbone_dataset_batch_matches_gcpnet_expectations` – Ensures collated backbone batches interface correctly with the GCPNet encoder and yield finite embeddings. ([source](test_data_pipeline.py#L232-L266))
- `test_preprocessed_dataset_batch_matches_gcpnet_expectations` – Confirms that preprocessed datasets match raw samples and remain compatible with the encoder pipeline. ([source](test_data_pipeline.py#L269-L318))
- `test_protein_batch_filters_edges_from_masked_nodes` – Asserts that masked residues are excluded from the graph connectivity in protein batches. ([source](test_data_pipeline.py#L321-L357))
- `test_dataset_and_collate` – Covers mixed-format dataset loading, batching behaviour, and padding with sentinel indices. ([source](test_data_pipeline.py#L360-L404))
- `test_dataset_repeated_parsing_matches` – Verifies cached dataset parsing is deterministic across multiple instances. ([source](test_data_pipeline.py#L411-L462))
- `test_dataset_skips_chains_without_valid_backbone` – Checks that samples lacking valid residues are omitted from the dataset. ([source](test_data_pipeline.py#L465-L477))

### `tests/test_preprocessing.py`
- `test_preprocess_dataset_roundtrip` – Runs the high-level preprocessing pipeline end-to-end, comparing manifest metadata and processed samples against raw inputs. ([source](test_preprocessing.py#L41-L121))

### `tests/test_reference_preprocessing_module.py`
- `test_validate_length_bounds` – Evaluates length validation helper thresholds for preprocessed chains. ([source](test_reference_preprocessing_module.py#L118-L139))
- `test_validate_missing_thresholds` – Confirms missing-residue heuristics detect excessive gaps and runs. ([source](test_reference_preprocessing_module.py#L142-L167))
- `test_preprocess_dataset_collects_stats` – Processes a mixture of structures and verifies manifest output plus comprehensive statistics tracking. ([source](test_reference_preprocessing_module.py#L170-L233))
- `test_preprocess_dataset_h5_matches_fixture` – Ensures generated HDF5 payloads match a known-good fixture for clean chains. ([source](test_reference_preprocessing_module.py#L236-L252))
- `test_preprocess_dataset_h5_preserves_nans` – Checks that NaN values in coordinates and pLDDT scores are preserved through preprocessing. ([source](test_reference_preprocessing_module.py#L255-L273))
- `test_preprocess_dataset_omits_index_when_requested` – Validates optional suppression of the file index during preprocessing. ([source](test_reference_preprocessing_module.py#L276-L291))

### `tests/test_pdb_hdf5.py`
- `test_three_to_one_contains_expected_mappings` – Spot-checks amino-acid code conversions used during PDB/HDF5 export. ([source](test_pdb_hdf5.py#L79-L86))
- `test_extract_chains_filters_and_tracks_stats` – Confirms chain extraction filters short chains and records summary statistics. ([source](test_pdb_hdf5.py#L89-L131))
- `test_extract_chains_prefers_chain_with_more_ca_atoms` – Ensures duplicate chains favour the instance with the most complete CA atoms. ([source](test_pdb_hdf5.py#L134-L160))
- `test_only_first_model_is_used` – Verifies that only the first model in multi-model structures contributes chains. ([source](test_pdb_hdf5.py#L163-L178))

### `tests/test_pdb_to_hdf5_integration.py`
- `test_preprocess_dataset_generates_hdf5_matching_structures` – Runs the reference preprocessing flow on real PDB chains and checks the produced HDF5 files and dataset entries. ([source](test_pdb_to_hdf5_integration.py#L36-L111))
- `test_training_pipeline_uses_preprocessed_hdf5` – Executes a short training run to confirm the trainer consumes preprocessed HDF5 inputs and emits checkpoints. ([source](test_pdb_to_hdf5_integration.py#L170-L214))

## Geometry, Metrics, and Model Components

### `tests/test_frames.py`
- `test_build_local_frames_right_handed` – Validates the local frame construction returns orthonormal, right-handed bases aligned with backbone tangents. ([source](test_frames.py#L21-L52))
- `test_build_local_frames_equivariant_under_rigid_transform` – Ensures local frames transform equivariantly under rigid motions. ([source](test_frames.py#L55-L90))
- `test_kabsch_align_identity` – Confirms the Kabsch solver returns identity transforms when inputs already align. ([source](test_frames.py#L105-L117))
- `test_kabsch_align_recovers_transform` – Checks recovery of an arbitrary rotation and translation via Kabsch alignment. ([source](test_frames.py#L120-L141))
- `test_kabsch_align_pure_translation` – Verifies translation-only displacements are recovered accurately. ([source](test_frames.py#L144-L162))
- `test_kabsch_align_supports_masks` – Confirms masked correspondences restrict the alignment appropriately. ([source](test_frames.py#L165-L187))
- `test_kabsch_align_requires_three_points` – Ensures the solver rejects inputs with insufficient correspondences. ([source](test_frames.py#L190-L197))
- `test_kabsch_alignment_with_reflection_toggle` – Tests the option to allow or forbid reflections during alignment. ([source](test_frames.py#L200-L226))
- `test_kabsch_align_float32_accuracy` – Measures numerical accuracy when operating in float32 precision. ([source](test_frames.py#L229-L246))
- `test_kabsch_align_low_precision_dtypes` – Exercises support for low-precision (bfloat16) inputs. ([source](test_frames.py#L249-L264))
- `test_kabsch_align_promotes_dtype_for_det` – Verifies determinant checks promote to a safe precision for numerical stability. ([source](test_frames.py#L267-L290))

### `tests/test_metrics.py`
- `test_gdt_ts_perfect_alignment` – Ensures the GDT-TS metric returns a perfect score when structures coincide. ([source](test_metrics.py#L8-L12))
- `test_gdt_ts_thresholds` – Validates the threshold averaging logic by perturbing coordinates within specific cutoffs. ([source](test_metrics.py#L15-L21))

### `tests/test_decoder.py`
- `test_structure_head_identity_template` – Confirms the rotation decoder reproduces the template scaffold for identity parameters. ([source](test_decoder.py#L10-L23))
- `test_structure_head_respects_mask` – Checks that masked positions yield zeroed outputs while preserving rotations for valid entries. ([source](test_decoder.py#L26-L55))
- `test_structure_head_scaling_factor` – Ensures the decoder’s scaling factor scales reconstructed coordinates as expected. ([source](test_decoder.py#L58-L68))
- `test_structure_head_produces_orthonormal_rotations` – Validates that decoded rotations remain orthonormal with determinant one. ([source](test_decoder.py#L71-L86))

### `tests/test_losses.py`
- `test_aligned_mse_supports_fully_masked_batch` – Verifies the aligned MSE loss remains differentiable even when all residues are masked. ([source](test_losses.py#L12-L22))
- `test_backbone_distance_loss_supports_fully_masked_batch` – Confirms the distance-based backbone loss handles empty masks gracefully. ([source](test_losses.py#L25-L35))
- `test_backbone_direction_loss_supports_short_backbones` – Checks the direction loss accommodates extremely short chains with fully masked inputs. ([source](test_losses.py#L38-L48))
- `test_reconstruction_loss_requires_grad_with_empty_mask` – Ensures the composite reconstruction loss still produces gradients when every position is masked. ([source](test_losses.py#L51-L61))

### `tests/test_vq.py`
- `test_vector_quantizer_forward_shapes` – Exercises the vector-quantiser forward pass, verifying shapes, metrics, and gradient flow. ([source](test_vq.py#L23-L44))
- `test_vector_quantizer_supports_masks` – Checks masked positions bypass codebook lookups while producing valid metrics. ([source](test_vq.py#L47-L63))
- `test_get_output_from_indices_matches_quantized_vectors` – Ensures decoding stored indices reproduces the quantised embeddings. ([source](test_vq.py#L66-L81))

## Model API, Integration, and Training

### `tests/test_model_api.py`
- `test_model_forward_runs` – Runs the end-to-end forward pass (including loss computation and gradients) on synthetic structures. ([source](test_model_api.py#L103-L139))
- `test_encode_decode_roundtrip` – Validates encode/decode parity, ensuring round-tripped coordinates and metadata align. ([source](test_model_api.py#L142-L183))
- `test_latent_adapter_projects_embeddings` – Confirms the optional latent adapter reshapes embeddings and that downstream tensors adopt the expected dimensions. ([source](test_model_api.py#L186-L238))
- `test_gcpnet_pretrained_initialisation` – Checks that pretrained GCPNet weights can be reloaded from a saved checkpoint. ([source](test_model_api.py#L241-L263))

### `tests/test_end_to_end.py`
- `test_vq_decoder_pipeline_runs` – Exercises the vector-quantiser and rotation decoder together, ensuring gradients flow through the combined pipeline. ([source](test_end_to_end.py#L69-L96))
- `test_roundtrip_rmsd_after_brief_training` – Performs a brief training loop and verifies the model achieves a low RMSD on a reconstructed structure. ([source](test_end_to_end.py#L99-L152))

### `tests/test_gcpnet.py`
- `test_gcpnet_encoder_projects_edge_scalars_with_default_input_dim` – Ensures the encoder projects edge scalars with the expected dimensionality and produces finite embeddings. ([source](test_gcpnet.py#L33-L76))
- `test_prepare_model_config_defaults_edge_scalar_input_dim` – Verifies configuration helpers inject default edge scalar dimensions when absent. ([source](test_gcpnet.py#L79-L87))
- `test_gcpconv_supports_bfloat16_inputs` – Confirms the GCP convolution operates on bfloat16 tensors without precision loss. ([source](test_gcpnet.py#L90-L125))
- `test_gcpnet_encoder_supports_bfloat16_inputs` – Checks the full encoder supports bfloat16 features and preserves data types in its outputs. ([source](test_gcpnet.py#L128-L177))
- `test_gcpnet_reference_checkpoint_loads` – Validates that the packaged GCP checkpoint contains usable weights that load into a fresh encoder instance. ([source](test_gcpnet.py#L180-L206))
- `test_packaged_configs_initialize_gcpnet_from_pretrained_weights` – Confirms packaged model configurations can map pretrained weights onto the encoder. ([source](test_gcpnet.py#L209-L235))

### `tests/test_eval.py`
- `test_evaluate_reports_summary` – Mocks the evaluation pipeline to ensure evaluation summarises dataset statistics, codebook usage, and structural metrics. ([source](test_eval.py#L86-L153))

### `tests/test_train.py`
- `test_training_harness_runs_single_stage` – Runs the training CLI on raw structures to confirm checkpoint emission for a single-stage schedule. ([source](test_train.py#L43-L120))
- `test_training_with_preprocessed_dataset` – Repeats the training harness against a preprocessed dataset to verify compatibility. ([source](test_train.py#L123-L196))
- `test_training_on_cif_dataset_decreases_loss` – Exercises a short multi-step training session and asserts that loss decreases at least once. ([source](test_train.py#L199-L282))
- `test_training_with_eval_and_export` – Validates evaluation hooks and sample exports trigger during training and write artefacts. ([source](test_train.py#L285-L364))
- `test_multi_stage_training_tracks_global_step` – Ensures multi-stage training tracks the global step correctly and emits checkpoints with consistent numbering. ([source](test_train.py#L367-L456))

## Comprehensive Pipelines

### `tests/test_pdb_to_hdf5_integration.py`
(see Data Ingestion and Preprocessing above for preprocessing and training coverage.)

### `tests/test_end_to_end.py`
(see Model API, Integration, and Training above for end-to-end round-trip checks.)

### `tests/test_pdb_hdf5.py`
(see Data Ingestion and Preprocessing for chain extraction and HDF5 coverage.)

---

These tests collectively ensure the CLI entry points, data handling, geometry utilities, model components, and training/evaluation loops remain reliable across incremental changes.

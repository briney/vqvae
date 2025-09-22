# VQ-VAE

Clean PyTorch implementation of the geometry-complete protein VQ-VAE (GCP-VQVAE).

## Features

- GCPNet encoder with equivariant micro-steps and gating
- Transformer context module with vector-quantized latent tokens
- 6D rotation decoder with rigid reconstruction losses
- CLI tooling for encoding, decoding, training, and evaluation

## Installation

```bash
pip install -e .
```

## License

This project is provided under the MIT License. See `LICENSE` for details.

# Minimus SDK

Optimized ML inference SDK for mobile and embedded devices.

## Overview

Minimus SDK provides pre-trained, optimized machine learning models that "just work" on resource-constrained devices. No ML expertise required - download, load, predict.

## Features

- **Curated Model Registry** - Discover and search optimized models
- **Automatic Downloading** - Models download on first use and cache locally
- **Checksum Verification** - Ensures model integrity
- **Optimized for Edge** - Quantized models for mobile/embedded deployment
- **Simple API** - Load and predict in just a few lines of code

## Available Models

### Agriculture

| Model ID | Description | Classes | Size |
|----------|-------------|---------|------|
| `plant-disease-v1` | Identifies plant diseases from leaf images | 38 | 12.5 MB |

### Livestock (Coming Soon)

| Model ID | Description | Classes | Size |
|----------|-------------|---------|------|
| `poultry-disease-v1` | Detects poultry diseases from fecal images | 4 | TBD |

### Health (Coming Soon)

| Model ID | Description | Input | Size |
|----------|-------------|-------|------|
| `cough-detector-v1` | Detects cough sounds for health screening | Audio | TBD |

## Why Minimus?

### Built for African Challenges

- **Agriculture** - Detect crop diseases early to prevent losses
- **Livestock** - Monitor animal health in rural areas
- **Healthcare** - Enable screening where specialists are scarce

### Built for Constraints

- **Low bandwidth** - Models download once, cache forever
- **Limited storage** - Quantized models, small footprint
- **Offline-first** - Works without internet after first download
- **Low-power devices** - Optimized inference

## Roadmap

- [x] Model registry with metadata
- [x] Automatic downloading and caching
- [x] Checksum verification
- [x] Plant disease detection model
- [ ] Poultry disease detection model
- [ ] Cough detection model
- [ ] Model quantization utilities
- [ ] Android/iOS examples
- [ ] WASM support



## License

MIT License - see [LICENSE](LICENSE) for details.

---

*Minimus: Minimum footprint, maximum impact.*
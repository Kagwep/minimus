# Minimus SDK

#RustAfricaHackathon

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
| `plant-disease-v1` | Identifies plant diseases from leaf images | 38 | 25 MB |

### Livestock (Coming Soon)

| Model ID | Description | Classes | Size |
|----------|-------------|---------|------|
| `poultry-disease-v1` | Detects poultry diseases from fecal images | 4 | TBD |

### Health (Coming Soon)

| Model ID | Description | Input | Size |
|----------|-------------|-------|------|
| `cough-detector-v1` | Detects cough sounds for health screening | Audio | TBD |

## Why Minimus?

### Built for

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


## Demo Application

We've built a proof-of-concept Android app that demonstrates the Minimus SDK in action.

### Plant Disease Detector App

**Download:** [minimise.apk](https://github.com/Kagwep/minimus/releases/download/models-v1.0.0/minimise.apk)

This demo app showcases real-world usage of the Minimus SDK:

- **Uses the SDK** - Integrates Minimus SDK to load the `plant-disease-v1` model
- **On-device inference** - Performs plant disease detection entirely on your phone
- **Offline-capable** - Works without internet after initial model download
- **Simple UX** - Take a photo or select from gallery, get instant diagnosis

### Features Demonstrated

- ✅ Automatic model downloading and caching
- ✅ Real-time image classification
- ✅ 38 plant disease classes detection
- ✅ Low-latency inference on mobile hardware
- ✅ Offline-first operation

### Try It

1. Download the APK to your Android device
2. Install (you may need to enable "Install from unknown sources")
3. Grant camera permissions
4. Take a photo of a plant leaf
5. Get instant disease detection results

> **Note:** First launch will download the model (~25 MB). Subsequent uses are fully offline.

## Test Images

Can't find a diseased plant to test with? No problem! We've included sample test images in the root of this repository that you can use to try out the SDK and demo app.

### Available Test Images

Browse the test images in the repository root to see examples of various plant diseases that the model can detect. These images are perfect for:

- Testing the SDK integration
- Trying out the demo app
- Validating your implementation
- Understanding the model's capabilities

### Using Test Images

**With the Demo App:**
1. Download test images to your phone
2. Open the app and select "Choose from Gallery"
3. Select a test image to see instant disease detection


## Quick Start

### For Developers
```rust
use minimus_sdk::ModelRegistry;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Initialize registry
    let registry = ModelRegistry::new().await?;
    
    // Load model (downloads automatically on first use)
    let model = registry.get_model("plant-disease-v1").await?;
    
    // Run inference
    let result = model.predict("path/to/leaf.jpg").await?;
    
    println!("Disease detected: {} ({:.2}% confidence)", 
             result.class_name, 
             result.confidence * 100.0);
    
    Ok(())
}
```

### Installation

Add to your `Cargo.toml`:
```toml
[dependencies]
minimus-sdk = "0.1.0"
```

Or install via cargo:
```bash
cargo add minimus-sdk
```

## API Reference

### ModelRegistry

- `new() -> Result<ModelRegistry>` - Initialize registry
- `list_models() -> Vec<ModelInfo>` - List available models
- `get_model(id: &str) -> Result<Model>` - Load/download model
- `clear_cache()` - Remove cached models

### Model

- `predict(image_path: &str) -> Result<Prediction>` - Run inference
- `predict_batch(images: Vec<&str>) -> Result<Vec<Prediction>>` - Batch inference
- `info() -> ModelInfo` - Get model metadata

### Prediction

- `class_name: String` - Detected class
- `confidence: f32` - Prediction confidence (0.0-1.0)
- `all_scores: Vec<(String, f32)>` - All class probabilities

## Contributing

We welcome contributions, especially:

- Optimizations for specific hardware
- Documentation and examples
- Bug reports and fixes

## License

MIT License - see [LICENSE](LICENSE) for details.

---

*Minimus: Minimum footprint, maximum impact.*
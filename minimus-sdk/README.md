# Minimus SDK

Optimized ML inference SDK for embedded and mobile devices.

## Features

- 🚀 **Lightweight**: Minimal dependencies, small binary size
- 📱 **Cross-platform**: Works on desktop, mobile (via Tauri), and embedded
- 🔌 **Simple API**: Load models and run predictions in a few lines
- 📦 **Model Registry**: Built-in catalog of optimized models
- ⬇️ **On-demand Downloads**: Only download models you need
- 💾 **Local Caching**: Models are cached for offline use

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

### Feature Flags

| Feature | Default | Description |
|---------|---------|-------------|
| `download` | ✓ | Enable model downloading from remote URLs |

To disable downloads (embedded-only mode):

```toml
[dependencies]
minimus-sdk = { version = "0.1", default-features = false }
```

## Loading Models

### 1. From Registry (Auto-download)

```rust
let model = minimus.load("plant-disease-v1").await?;
```

### 2. From Embedded Bytes

```rust
let bytes = include_bytes!("../models/plant-disease.onnx");
let model = minimus.load_from_bytes("plant-disease-v1", bytes)?;
```

### 3. Custom Model

```rust
use minimus_sdk::{ModelInfo, ModelType, MinimusModel};

let info = ModelInfo::builder("my-classifier")
    .name("My Custom Classifier")
    .description("Classifies things")
    .model_type(ModelType::ImageClassification)
    .input_size(224, 224)
    .classes_static(&["cat", "dog", "bird"])
    .build();

let bytes = std::fs::read("my-model.onnx")?;
let model = MinimusModel::load_custom(&bytes, info)?;
```

## Running Predictions

### Single Prediction

```rust
let result = model.predict(&image_bytes)?;
println!("{}: {:.1}%", result.label, result.confidence_percent());
```

### Top-K Predictions

```rust
let results = model.predict_topk(&image_bytes, 5)?;
for pred in results {
    println!("{}: {:.1}%", pred.display_label(), pred.confidence_percent());
}
```

## Cache Management

```rust
// Check cache size
println!("Cache: {:.1} MB", minimus.cache_size_mb());

// Clear specific model
minimus.clear_model("plant-disease-v1")?;

// Clear all cache
minimus.clear_cache()?;
```

## Available Models

| ID | Name | Type | Size |
|----|------|------|------|
| `plant-disease-v1` | Plant Disease Detector | Classification | 12.5 MB |

## Use with Tauri

```rust
use minimus_sdk::{Minimus, MinimusModel};
use std::sync::Arc;
use tauri::State;
use tokio::sync::RwLock;

struct AppState {
    model: RwLock<Option<MinimusModel>>,
}

#[tauri::command]
async fn predict(
    state: State<'_, AppState>,
    image_bytes: Vec<u8>,
) -> Result<String, String> {
    let model = state.model.read().await;
    let model = model.as_ref().ok_or("Model not loaded")?;
    
    let result = model.predict(&image_bytes).map_err(|e| e.to_string())?;
    Ok(result.display_label())
}
```

## License

MIT

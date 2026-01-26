use once_cell::sync::Lazy;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Type of ML model
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum ModelType {
    ImageClassification,
    ObjectDetection,
    PlantDisease,
    Custom,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum ModelFormat {
    NnefTar,
    NnefDirectory,
    Onnx,
}

/// Metadata about a model (no weights included)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelInfo {
    /// Unique identifier (e.g., "plant-disease-v1")
    pub id: String,
    /// Human-readable name
    pub name: String,
    /// Description of what the model does
    pub description: String,
    /// Type of model
    pub model_type: ModelType,
    /// Input dimensions (width, height)
    pub format: ModelFormat,
    pub input_size: (u32, u32),
    /// Number of channels (usually 3 for RGB)
    pub channels: u32,
    /// Class labels for classification models
    pub classes: Vec<String>,
    /// Model version
    pub version: String,
    /// Approximate size in MB
    pub size_mb: f32,
    /// URL to download model weights
    pub download_url: Option<String>,
    /// SHA256 hash for integrity verification
    pub sha256: Option<String>,
}

impl ModelInfo {
    /// Create a new ModelInfo builder
    pub fn builder(id: impl Into<String>) -> ModelInfoBuilder {
        ModelInfoBuilder::new(id)
    }
}

/// Builder for ModelInfo
pub struct ModelInfoBuilder {
    info: ModelInfo,
}

impl ModelInfoBuilder {
    pub fn new(id: impl Into<String>) -> Self {
        Self {
            info: ModelInfo {
                id: id.into(),
                name: String::new(),
                description: String::new(),
                model_type: ModelType::Custom,
                input_size: (224, 224),
                channels: 3,
                classes: Vec::new(),
                version: "1.0.0".into(),
                format:ModelFormat::NnefTar,
                size_mb: 0.0,
                download_url: None,
                sha256: None,
            },
        }
    }

    pub fn name(mut self, name: impl Into<String>) -> Self {
        self.info.name = name.into();
        self
    }

    pub fn description(mut self, desc: impl Into<String>) -> Self {
        self.info.description = desc.into();
        self
    }

    pub fn model_type(mut self, t: ModelType) -> Self {
        self.info.model_type = t;
        self
    }

    pub fn input_size(mut self, width: u32, height: u32) -> Self {
        self.info.input_size = (width, height);
        self
    }

    pub fn channels(mut self, c: u32) -> Self {
        self.info.channels = c;
        self
    }

    pub fn classes(mut self, classes: Vec<String>) -> Self {
        self.info.classes = classes;
        self
    }

    pub fn classes_static(mut self, classes: &[&str]) -> Self {
        self.info.classes = classes.iter().map(|s| s.to_string()).collect();
        self
    }

    pub fn version(mut self, v: impl Into<String>) -> Self {
        self.info.version = v.into();
        self
    }

    pub fn size_mb(mut self, size: f32) -> Self {
        self.info.size_mb = size;
        self
    }

    pub fn download_url(mut self, url: impl Into<String>) -> Self {
        self.info.download_url = Some(url.into());
        self
    }

    pub fn sha256(mut self, hash: impl Into<String>) -> Self {
        self.info.sha256 = Some(hash.into());
        self
    }

    pub fn build(self) -> ModelInfo {
        self.info
    }
}

/// Registry of available models
pub struct ModelRegistry {
    models: HashMap<String, ModelInfo>,
}

impl ModelRegistry {
    /// Create a new empty registry
    pub fn new() -> Self {
        Self {
            models: HashMap::new(),
        }
    }

    /// Create registry with built-in models
    pub fn with_defaults() -> Self {
        let mut registry = Self::new();
        registry.register_defaults();
        registry
    }

    /// Register the default built-in models
    fn register_defaults(&mut self) {
        // Plant Disease Model
        self.register(
            ModelInfo::builder("plant-disease-v1")
                .name("Plant Disease Detector")
                .description("Identifies 38 plant diseases from leaf images")
                .model_type(ModelType::PlantDisease)
                .input_size(256, 256)
                .classes_static(&crate::models::plant_disease::CLASSES)
                .version("1.0.0")
                .size_mb(12.5)
                .download_url("https://github.com/Kagwep/minimus/releases/download/models-v1.0.0/plant_disease.nnef.tar")
                .sha256("sha256:a80789a823be847c1b17a35f47b3dd3faa4ebd9bcbf3b954276895eae2a08cf1") 
                .build()
        );

        // Add more default models here as they become available
    }

    /// Register a model
    pub fn register(&mut self, info: ModelInfo) {
        self.models.insert(info.id.clone(), info);
    }

    /// Unregister a model
    pub fn unregister(&mut self, id: &str) -> Option<ModelInfo> {
        self.models.remove(id)
    }

    /// List all available models
    pub fn list(&self) -> Vec<&ModelInfo> {
        self.models.values().collect()
    }

    /// List models by type
    pub fn list_by_type(&self, model_type: ModelType) -> Vec<&ModelInfo> {
        self.models
            .values()
            .filter(|m| m.model_type == model_type)
            .collect()
    }

    /// Get model info by ID
    pub fn get(&self, id: &str) -> Option<&ModelInfo> {
        self.models.get(id)
    }

    /// Check if model exists
    pub fn exists(&self, id: &str) -> bool {
        self.models.contains_key(id)
    }

    /// Search models by name or description
    pub fn search(&self, query: &str) -> Vec<&ModelInfo> {
        let query_lower = query.to_lowercase();
        self.models
            .values()
            .filter(|m| {
                m.name.to_lowercase().contains(&query_lower)
                    || m.description.to_lowercase().contains(&query_lower)
                    || m.id.to_lowercase().contains(&query_lower)
            })
            .collect()
    }

    /// Get number of registered models
    pub fn len(&self) -> usize {
        self.models.len()
    }

    /// Check if registry is empty
    pub fn is_empty(&self) -> bool {
        self.models.is_empty()
    }
}

impl Default for ModelRegistry {
    fn default() -> Self {
        Self::with_defaults()
    }
}

/// Global registry instance with default models
pub static REGISTRY: Lazy<ModelRegistry> = Lazy::new(ModelRegistry::with_defaults);

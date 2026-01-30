
mod error;
mod inference;
mod loader;
mod model;
pub mod models;
mod registry;

pub use error::MinimusError;
pub use inference::{InferenceConfig, Prediction};
pub use loader::ModelLoader;
pub use model::MinimusModel;
pub use registry::{ModelInfo, ModelInfoBuilder, ModelRegistry, ModelType, REGISTRY,ModelFormat};

use std::path::PathBuf;

/// Main SDK interface
pub struct Minimus {
    loader: ModelLoader,
    work_dir: PathBuf,
}

impl Minimus {
    /// Create a new SDK instance with default cache directory
/// Initialize with explicit paths (The safe way for Cross-Platform)
    pub fn new(cache_dir: PathBuf, work_dir: PathBuf) -> Self {
        // Ensure paths exist
        std::fs::create_dir_all(&cache_dir).ok();
        std::fs::create_dir_all(&work_dir).ok();
        
        Self {
            loader: ModelLoader::new(cache_dir),
            work_dir,
        }
    }

    /// Create SDK with a custom cache directory
    pub fn with_cache_dir(cache_dir: PathBuf) -> Self {
        let work_dir = cache_dir.join(".tmp");
        Self::new(cache_dir, work_dir)
    }

    /// Get the model loader
    pub fn loader(&self) -> &ModelLoader {
        &self.loader
    }

    // ========== Model Discovery ==========

    /// List all available models in the registry
    pub fn available_models(&self) -> Vec<&ModelInfo> {
        REGISTRY.list()
    }

    /// List models by type
    pub fn models_by_type(&self, model_type: ModelType) -> Vec<&ModelInfo> {
        REGISTRY.list_by_type(model_type)
    }

    pub fn model_format(&self, model_id: &str) -> Option<ModelFormat> {
    REGISTRY.get(model_id).map(|info| info.format.clone())
    }

    /// Search models by name or description
    pub fn search_models(&self, query: &str) -> Vec<&ModelInfo> {
        REGISTRY.search(query)
    }

    /// Get info for a specific model
    pub fn model_info(&self, model_id: &str) -> Option<&ModelInfo> {
        REGISTRY.get(model_id)
    }

    /// Check if a model exists in the registry
    pub fn model_exists(&self, model_id: &str) -> bool {
        REGISTRY.exists(model_id)
    }

    // ========== Model Status ==========

    /// Check if a model is downloaded/cached
    pub fn is_downloaded(&self, model_id: &str) -> bool {
        REGISTRY
            .get(model_id)
            .map(|info| self.loader.is_cached(info))
            .unwrap_or(false)
    }

    /// Get download status for all models
    pub fn download_status(&self) -> Vec<(&ModelInfo, bool)> {
        REGISTRY
            .list()
            .into_iter()
            .map(|info| (info, self.loader.is_cached(info)))
            .collect()
    }

    // ========== Model Loading ==========

    /// Load a model by ID (downloads if needed and `download` feature is enabled)
    #[cfg(feature = "download")]
    pub async fn load(&self, model_id: &str) -> Result<MinimusModel, MinimusError> {
        let info = REGISTRY
            .get(model_id)
            .ok_or_else(|| MinimusError::ModelNotFound(model_id.to_string()))?;

        let bytes = self.loader.load_bytes(info).await?;
        MinimusModel::load(model_id, &bytes,self.work_dir.clone())
    }

    /// Load a model by ID (cache-only when download feature is disabled)
    #[cfg(not(feature = "download"))]
    pub async fn load(&self, model_id: &str) -> Result<MinimusModel, MinimusError> {
        let info = REGISTRY
            .get(model_id)
            .ok_or_else(|| MinimusError::ModelNotFound(model_id.to_string()))?;

        let bytes = self.loader.load_from_cache(info)?;
        MinimusModel::load(model_id, &bytes)
    }

    /// Load a model from provided bytes (no download)
    pub fn load_from_bytes(
        &self,
        model_id: &str,
        bytes: &[u8],
    ) -> Result<MinimusModel, MinimusError> {
        MinimusModel::load(model_id, bytes,self.work_dir.clone())
    }

    /// Load a model from bytes with custom config
    pub fn load_from_bytes_with_config(
        &self,
        model_id: &str,
        bytes: &[u8],
        config: InferenceConfig,
    ) -> Result<MinimusModel, MinimusError> {
        MinimusModel::load_with_config(model_id, bytes, config,self.work_dir.clone())
    }

    /// Load a custom model (not in registry)
    pub fn load_custom(
        &self,
        bytes: &[u8],
        info: ModelInfo,
    ) -> Result<MinimusModel, MinimusError> {
        MinimusModel::load_custom(bytes, info,self.work_dir.clone())
    }

    // ========== Download Management ==========

    /// Download a model without loading it (for pre-caching)
    #[cfg(feature = "download")]
    pub async fn download(&self, model_id: &str) -> Result<(), MinimusError> {
        let info = REGISTRY
            .get(model_id)
            .ok_or_else(|| MinimusError::ModelNotFound(model_id.to_string()))?;

        let bytes = self.loader.download(info).await?;
        self.loader.save_to_cache(info, &bytes)?;
        Ok(())
    }

    /// Download multiple models
    #[cfg(feature = "download")]
    pub async fn download_many(&self, model_ids: &[&str]) -> Vec<Result<(), MinimusError>> {
        let mut results = Vec::new();
        for id in model_ids {
            results.push(self.download(id).await);
        }
        results
    }

    // ========== Cache Management ==========

    /// Get total cache size in MB
    pub fn cache_size_mb(&self) -> f32 {
        self.loader.cache_size_mb()
    }

    /// Clear a specific model from cache
    pub fn clear_model(&self, model_id: &str) -> Result<(), MinimusError> {
        let info = REGISTRY
            .get(model_id)
            .ok_or_else(|| MinimusError::ModelNotFound(model_id.to_string()))?;
        self.loader.clear(info)
    }

    /// Clear all cached models
    pub fn clear_cache(&self) -> Result<(), MinimusError> {
        self.loader.clear_all()
    }
}

impl Default for Minimus {
    fn default() -> Self {
        // Use the OS-provided cache dir, or fall back to a local folder
        let base_dir = dirs::cache_dir()
            .unwrap_or_else(|| std::env::current_dir().unwrap_or_default())
            .join("minimus");
            
        let cache_dir = base_dir.join("models");
        let work_dir = base_dir.join("runtime");
        
        Self::new(cache_dir, work_dir)
    }
}

// ========== Convenience Functions ==========

/// Quick prediction without managing SDK instance
pub fn quick_predict(
    model_id: &str,
    model_bytes: &[u8],
    image_bytes: &[u8],
) -> Result<Prediction, MinimusError> {
        // Use the OS-provided cache dir, or fall back to a local folder
    let base_dir = dirs::cache_dir()
        .unwrap_or_else(|| std::env::current_dir().unwrap_or_default())
        .join("minimus");
        
    let cache_dir = base_dir.join("models");
    let work_dir = base_dir.join("runtime");
    let model = MinimusModel::load(model_id, model_bytes,work_dir)?;
    model.predict(image_bytes)
}

/// Get version of the SDK
pub fn version() -> &'static str {
    env!("CARGO_PKG_VERSION")
}

//! Model downloading and caching functionality
//! 
//! This module is only available with the `download` feature.

use std::path::PathBuf;
use crate::error::MinimusError;
use crate::registry::ModelInfo;

#[cfg(feature = "download")]
use sha2::{Sha256, Digest};

/// Handles model downloading and local caching
pub struct ModelLoader {
    cache_dir: PathBuf,
}

impl ModelLoader {
    /// Create a new loader with a specific cache directory
    pub fn new(cache_dir: PathBuf) -> Self {
        // Ensure cache directory exists
        std::fs::create_dir_all(&cache_dir).ok();
        Self { cache_dir }
    }

    /// Create loader with platform-specific default cache directory
    #[cfg(feature = "download")]
    pub fn with_default_cache() -> Self {
        let cache_dir = dirs::cache_dir()
            .unwrap_or_else(|| PathBuf::from("."))
            .join("minimus-models");
        Self::new(cache_dir)
    }

    /// Create loader with platform-specific default cache directory
    #[cfg(not(feature = "download"))]
    pub fn with_default_cache() -> Self {
        Self::new(PathBuf::from(".minimus-cache"))
    }

    /// Get the cache directory path
    pub fn cache_dir(&self) -> &PathBuf {
        &self.cache_dir
    }

    /// Get the local file path for a model
    pub fn model_path(&self, info: &ModelInfo) -> PathBuf {
        self.cache_dir
            .join(format!("{}-{}.onnx", info.id, info.version))
    }

    /// Check if a model is cached locally
    pub fn is_cached(&self, info: &ModelInfo) -> bool {
        self.model_path(info).exists()
    }

    /// Load model bytes from cache
    pub fn load_from_cache(&self, info: &ModelInfo) -> Result<Vec<u8>, MinimusError> {
        let path = self.model_path(info);
        if !path.exists() {
            return Err(MinimusError::ModelNotFound(info.id.clone()));
        }
        
        let bytes = std::fs::read(&path)?;
        
        #[cfg(feature = "download")]
        if let Some(expected_hash) = &info.sha256 {
            self.verify_checksum(&bytes, expected_hash)?;
        }
        
        Ok(bytes)
    }

    /// Save model bytes to cache
    pub fn save_to_cache(&self, info: &ModelInfo, bytes: &[u8]) -> Result<(), MinimusError> {
        let path = self.model_path(info);
        std::fs::write(&path, bytes)?;
        Ok(())
    }

    /// Download model from URL (requires `download` feature)
    #[cfg(feature = "download")]
    pub async fn download(&self, info: &ModelInfo) -> Result<Vec<u8>, MinimusError> {
        let url = info
            .download_url
            .as_ref()
            .ok_or_else(|| MinimusError::InvalidInput("No download URL specified".into()))?;

        let response = reqwest::get(url).await?;
        
        if !response.status().is_success() {
            return Err(MinimusError::Download(
                response.error_for_status().unwrap_err()
            ));
        }

        let bytes = response.bytes().await?.to_vec();

        // Verify checksum if provided
        if let Some(expected_hash) = &info.sha256 {
            self.verify_checksum(&bytes, expected_hash)?;
        }

        Ok(bytes)
    }

    /// Download model (stub for when download feature is disabled)
    #[cfg(not(feature = "download"))]
    pub async fn download(&self, _info: &ModelInfo) -> Result<Vec<u8>, MinimusError> {
        Err(MinimusError::DownloadDisabled)
    }

    /// Load model bytes - from cache if available, otherwise download
    #[cfg(feature = "download")]
    pub async fn load_bytes(&self, info: &ModelInfo) -> Result<Vec<u8>, MinimusError> {
        // Try cache first
        if self.is_cached(info) {
            return self.load_from_cache(info);
        }

        // Download and cache
        let bytes = self.download(info).await?;
        self.save_to_cache(info, &bytes)?;
        
        Ok(bytes)
    }

    /// Load model bytes (non-download version)
    #[cfg(not(feature = "download"))]
    pub async fn load_bytes(&self, info: &ModelInfo) -> Result<Vec<u8>, MinimusError> {
        self.load_from_cache(info)
    }

    /// Verify SHA256 checksum
    #[cfg(feature = "download")]
    fn verify_checksum(&self, bytes: &[u8], expected: &str) -> Result<(), MinimusError> {
        let hash = format!("{:x}", Sha256::digest(bytes));
        if hash != expected.to_lowercase() {
            return Err(MinimusError::ChecksumMismatch);
        }
        Ok(())
    }

    /// Delete a cached model
    pub fn clear(&self, info: &ModelInfo) -> Result<(), MinimusError> {
        let path = self.model_path(info);
        if path.exists() {
            std::fs::remove_file(path)?;
        }
        Ok(())
    }

    /// Clear all cached models
    pub fn clear_all(&self) -> Result<(), MinimusError> {
        if self.cache_dir.exists() {
            std::fs::remove_dir_all(&self.cache_dir)?;
            std::fs::create_dir_all(&self.cache_dir)?;
        }
        Ok(())
    }

    /// Get total cache size in bytes
    pub fn cache_size(&self) -> u64 {
        if !self.cache_dir.exists() {
            return 0;
        }

        std::fs::read_dir(&self.cache_dir)
            .map(|entries| {
                entries
                    .filter_map(|e| e.ok())
                    .filter_map(|e| e.metadata().ok())
                    .map(|m| m.len())
                    .sum()
            })
            .unwrap_or(0)
    }

    /// Get cache size in MB
    pub fn cache_size_mb(&self) -> f32 {
        self.cache_size() as f32 / (1024.0 * 1024.0)
    }

    /// List all cached model files
    pub fn list_cached(&self) -> Vec<PathBuf> {
        if !self.cache_dir.exists() {
            return Vec::new();
        }

        std::fs::read_dir(&self.cache_dir)
            .map(|entries| {
                entries
                    .filter_map(|e| e.ok())
                    .map(|e| e.path())
                    .filter(|p| p.extension().map(|e| e == "onnx").unwrap_or(false))
                    .collect()
            })
            .unwrap_or_default()
    }
}

impl Default for ModelLoader {
    fn default() -> Self {
        Self::with_default_cache()
    }
}

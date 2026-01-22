use thiserror::Error;
use tract_onnx::prelude::TractError;

#[derive(Error, Debug)]
pub enum MinimusError {
    #[error("Model not found in registry: {0}")]
    ModelNotFound(String),


    #[error("Image processing failed: {0}")]
    ImageError(#[from] image::ImageError),

    #[error("No prediction result")]
    NoPrediction,

    #[error("Invalid input: {0}")]
    InvalidInput(String),

    #[error("Checksum mismatch - model file may be corrupted")]
    ChecksumMismatch,

    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),

    #[cfg(feature = "download")]
    #[error("Download failed: {0}")]
    Download(#[from] reqwest::Error),

    #[error("Model not downloaded and download feature is disabled")]
    DownloadDisabled,

    #[error("Model loading failed: {0}")]
    ModelLoad(String),
}

impl From<TractError> for MinimusError {
    fn from(e: TractError) -> Self {
        MinimusError::ModelLoad(format!("{e:?}"))
    }
}
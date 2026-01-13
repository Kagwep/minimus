use serde::{Deserialize, Serialize};

/// A single prediction result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Prediction {
    /// Human-readable label
    pub label: String,
    /// Confidence score (0.0 - 1.0)
    pub confidence: f32,
    /// Index in the classes array
    pub class_index: usize,
}

impl Prediction {
    /// Create a new prediction
    pub fn new(label: impl Into<String>, confidence: f32, class_index: usize) -> Self {
        Self {
            label: label.into(),
            confidence,
            class_index,
        }
    }

    /// Get confidence as percentage (0 - 100)
    pub fn confidence_percent(&self) -> f32 {
        self.confidence * 100.0
    }

    /// Format label for display (replace underscores, etc.)
    pub fn display_label(&self) -> String {
        self.label
            .replace("___", " - ")
            .replace('_', " ")
    }
}

/// Configuration for model inference
#[derive(Debug, Clone)]
pub struct InferenceConfig {
    /// Normalize pixel values to 0-1 range
    pub normalize: bool,
    /// Channel ordering: true = NCHW (PyTorch), false = NHWC (TensorFlow)
    pub channels_first: bool,
    /// Mean values for normalization (per channel)
    pub mean: Option<[f32; 3]>,
    /// Std values for normalization (per channel)
    pub std: Option<[f32; 3]>,
}

impl Default for InferenceConfig {
    fn default() -> Self {
        Self {
            normalize: true,
            channels_first: true,
            mean: None,
            std: None,
        }
    }
}

impl InferenceConfig {
    /// Standard ImageNet normalization
    pub fn imagenet() -> Self {
        Self {
            normalize: true,
            channels_first: true,
            mean: Some([0.485, 0.456, 0.406]),
            std: Some([0.229, 0.224, 0.225]),
        }
    }

    /// Simple 0-1 normalization
    pub fn simple() -> Self {
        Self::default()
    }
}

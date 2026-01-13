use std::sync::Arc;
use tract_onnx::prelude::*;

use crate::error::MinimusError;
use crate::inference::{InferenceConfig, Prediction};
use crate::registry::{ModelInfo, REGISTRY};

/// Type alias for the tract model plan
type ModelPlan = SimplePlan<TypedFact, Box<dyn TypedOp>, Graph<TypedFact, Box<dyn TypedOp>>>;

/// A loaded ML model ready for inference
pub struct MinimusModel {
    plan: Arc<ModelPlan>,
    info: ModelInfo,
    config: InferenceConfig,
}

impl MinimusModel {
    /// Load a registered model by ID
    pub fn load(model_id: &str, onnx_bytes: &[u8]) -> Result<Self, MinimusError> {
        Self::load_with_config(model_id, onnx_bytes, InferenceConfig::default())
    }

    /// Load a registered model with custom inference config
    pub fn load_with_config(
        model_id: &str,
        onnx_bytes: &[u8],
        config: InferenceConfig,
    ) -> Result<Self, MinimusError> {
        let info = REGISTRY
            .get(model_id)
            .ok_or_else(|| MinimusError::ModelNotFound(model_id.to_string()))?
            .clone();

        Self::load_with_info(onnx_bytes, info, config)
    }

    /// Load a custom model with provided info
    pub fn load_custom(
        onnx_bytes: &[u8],
        info: ModelInfo,
    ) -> Result<Self, MinimusError> {
        Self::load_with_info(onnx_bytes, info, InferenceConfig::default())
    }

    /// Load a custom model with provided info and config
    pub fn load_with_info(
        onnx_bytes: &[u8],
        info: ModelInfo,
        config: InferenceConfig,
    ) -> Result<Self, MinimusError> {
        let (w, h) = info.input_size;
        let c = info.channels as i64;

        let mut cursor = std::io::Cursor::new(onnx_bytes);

        let input_shape = if config.channels_first {
            [1, c, h as i64, w as i64]
        } else {
            [1, h as i64, w as i64, c]
        };

        let plan = tract_onnx::onnx()
            .model_for_read(&mut cursor)?
            .with_input_fact(0, f32::fact(&input_shape).into())?
            .into_typed()?
            .into_runnable()?;

        Ok(Self {
            plan: Arc::new(plan),
            info,
            config,
        })
    }

    /// Get model info
    pub fn info(&self) -> &ModelInfo {
        &self.info
    }

    /// Get model ID
    pub fn id(&self) -> &str {
        &self.info.id
    }

    /// Get number of classes
    pub fn num_classes(&self) -> usize {
        self.info.classes.len()
    }

    /// Run prediction on image bytes, returns top result
    pub fn predict(&self, image_bytes: &[u8]) -> Result<Prediction, MinimusError> {
        let predictions = self.predict_topk(image_bytes, 1)?;
        predictions
            .into_iter()
            .next()
            .ok_or(MinimusError::NoPrediction)
    }

    /// Run prediction and return top-k results
    pub fn predict_topk(
        &self,
        image_bytes: &[u8],
        k: usize,
    ) -> Result<Vec<Prediction>, MinimusError> {
        let tensor = self.preprocess(image_bytes)?;
        let probs = self.run_inference(tensor)?;
        let predictions = self.postprocess(probs, k);
        Ok(predictions)
    }

    /// Run prediction on pre-processed tensor
    pub fn predict_tensor(&self, tensor: Tensor) -> Result<Prediction, MinimusError> {
        let probs = self.run_inference(tensor)?;
        let predictions = self.postprocess(probs, 1);
        predictions
            .into_iter()
            .next()
            .ok_or(MinimusError::NoPrediction)
    }

    /// Preprocess image bytes into a tensor
    fn preprocess(&self, image_bytes: &[u8]) -> Result<Tensor, MinimusError> {
        let (w, h) = self.info.input_size;

        // Load and resize image
        let img = image::load_from_memory(image_bytes)?;
        let resized = img.resize_exact(w, h, image::imageops::FilterType::Triangle);
        let rgb = resized.to_rgb8();

        // Create tensor based on config
        let tensor: Tensor = if self.config.channels_first {
            tract_ndarray::Array4::from_shape_fn(
                (1, 3, h as usize, w as usize),
                |(_, c, y, x)| self.normalize_pixel(rgb.get_pixel(x as u32, y as u32)[c], c),
            )
            .into()
        } else {
            tract_ndarray::Array4::from_shape_fn(
                (1, h as usize, w as usize, 3),
                |(_, y, x, c)| self.normalize_pixel(rgb.get_pixel(x as u32, y as u32)[c], c),
            )
            .into()
        };

        Ok(tensor)
    }

    /// Normalize a single pixel value
    fn normalize_pixel(&self, value: u8, channel: usize) -> f32 {
        let mut v = value as f32;

        if self.config.normalize {
            v /= 255.0;
        }

        if let Some(mean) = self.config.mean {
            v -= mean[channel];
        }

        if let Some(std) = self.config.std {
            v /= std[channel];
        }

        v
    }

    /// Run the model inference
    fn run_inference(&self, tensor: Tensor) -> Result<Vec<f32>, MinimusError> {
        let result = self.plan.run(tvec!(tensor.into()))?;
        let probs = result[0]
            .to_array_view::<f32>()
            .map_err(|e| MinimusError::ModelLoad(e.to_string()))?;

        Ok(probs.iter().copied().collect())
    }

    /// Convert raw outputs to predictions
    fn postprocess(&self, probs: Vec<f32>, k: usize) -> Vec<Prediction> {
        let mut indexed: Vec<(usize, f32)> = probs.into_iter().enumerate().collect();
        indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        indexed
            .into_iter()
            .take(k)
            .map(|(idx, confidence)| {
                let label = self
                    .info
                    .classes
                    .get(idx)
                    .cloned()
                    .unwrap_or_else(|| format!("class_{}", idx));

                Prediction::new(label, confidence, idx)
            })
            .collect()
    }
}

// Allow cloning the model (shares the underlying plan)
impl Clone for MinimusModel {
    fn clone(&self) -> Self {
        Self {
            plan: Arc::clone(&self.plan),
            info: self.info.clone(),
            config: self.config.clone(),
        }
    }
}

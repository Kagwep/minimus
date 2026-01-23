#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

use image::{GenericImageView, imageops::FilterType};
use ndarray::{Array, Array4};
use tract_onnx::prelude::*;
use std::path::Path;
use minimus_sdk::Minimus;
use std::sync::OnceLock;
use once_cell::sync::Lazy;

type Model = SimplePlan<TypedFact, Box<dyn TypedOp>, Graph<TypedFact, Box<dyn TypedOp>>>;

static MODEL: Lazy<Result<Model, String>> = Lazy::new(|| {
    tract_nnef::nnef()
        .with_tract_core()
        .model_for_path("plant_disease_nnef")
        .map_err(|e| format!("Load failed: {:#}", e))?
        .into_optimized()
        .map_err(|e| format!("Optimization failed: {:#}", e))?
        .into_runnable()
        .map_err(|e| format!("Runnable failed: {:#}", e))
});

fn get_model() -> Result<&'static Model, String> {
    MODEL.as_ref().map_err(|e| e.clone())
}

static CLASSES: &[&str] = &[
    "Apple___Apple_scab", "Apple___Black_rot", "Apple___Cedar_apple_rust", "Apple___healthy",
    "Blueberry___healthy", "Cherry_(including_sour)___healthy", "Cherry_(including_sour)___Powdery_mildew",
    "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot", "Corn_(maize)___Common_rust_",
    "Corn_(maize)___healthy", "Corn_(maize)___Northern_Leaf_Blight", "Grape___Black_rot",
    "Grape___Esca_(Black_Measles)", "Grape___healthy", "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)",
    "Orange___Haunglongbing_(Citrus_greening)", "Peach___Bacterial_spot", "Peach___healthy",
    "Pepper,_bell___Bacterial_spot", "Pepper,_bell___healthy", "Potato___Early_blight",
    "Potato___healthy", "Potato___Late_blight", "Raspberry___healthy", "Soybean___healthy",
    "Squash___Powdery_mildew", "Strawberry___healthy", "Strawberry___Leaf_scorch",
    "Tomato___Bacterial_spot", "Tomato___Early_blight", "Tomato___healthy", "Tomato___Late_blight",
    "Tomato___Leaf_Mold", "Tomato___Septoria_leaf_spot", "Tomato___Spider_mites Two-spotted_spider_mite",
    "Tomato___Target_Spot", "Tomato___Tomato_mosaic_virus", "Tomato___Tomato_Yellow_Leaf_Curl_Virus",
];



static MINIMUS: OnceLock<Minimus> = OnceLock::new();

fn get_minimus() -> &'static Minimus {
    MINIMUS.get_or_init(Minimus::new)
}

#[tauri::command]
async fn predict(image_path: String) -> Result<String, String> {
    let model = get_model()?;
    // 2. Preprocess image into a Tract Tensor
    let img = image::open(&image_path).map_err(|e| e.to_string())?;
    let resized = img.resize_exact(256, 256, image::imageops::FilterType::Triangle);
    let rgb = resized.to_rgb8();

    // Create a tensor with shape [1, 3, 256, 256]
    let tensor: Tensor = tract_ndarray::Array4::from_shape_fn((1, 3, 256, 256), |(_, c, y, x)| {
        rgb.get_pixel(x as u32, y as u32)[c] as f32 / 255.0
    }).into();

    // 3. Run inference
    let result = model.run(tvec!(tensor.into()))
        .map_err(|e| format!("Inference failed: {}", e))?;

    // 4. Extract results
    // result[0] is the output tensor, we convert it back to a slice of f32
    let probabilities = result[0]
        .to_array_view::<f32>()
        .map_err(|e| e.to_string())?;

    let mut max_idx = 0;
    let mut max_val = f32::NEG_INFINITY;
    for (i, &val) in probabilities.iter().enumerate() {
        if val > max_val {
            max_val = val;
            max_idx = i;
        }
    }

    Ok(CLASSES[max_idx].to_string())
}
fn main() {
    tauri::Builder::default()
        .plugin(tauri_plugin_fs::init())
        .plugin(tauri_plugin_dialog::init())
        .invoke_handler(tauri::generate_handler![predict])
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}
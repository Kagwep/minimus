use tauri::{Manager, State};
use tract_onnx::prelude::*;
use std::sync::Arc;
use std::sync::{ RwLock};
use tauri_plugin_fs::FsExt;
use std::path::Path;
// We store the model in a shared state so it only loads once
struct ModelState {
    // We use Option so we can initialize it as None
    model: RwLock<Option<Arc<SimplePlan<TypedFact, Box<dyn TypedOp>, Graph<TypedFact, Box<dyn TypedOp>>>>>>,
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

#[tauri::command]
async fn predict(
    state: State<'_, ModelState>, 
    image_bytes: Vec<u8>
) -> Result<String, String> {
    // 1. Check if model is ready
    let model_lock = state.model.read().unwrap();
    let model = model_lock.as_ref().ok_or("Model is still loading...")?;

    // 2. Convert bytes to image (bytes come directly from frontend now)
    let img = image::load_from_memory(&image_bytes)
        .map_err(|e| format!("Image Load Error: {}", e))?;
    
    // 3. Processing
    let resized = img.resize_exact(256, 256, image::imageops::FilterType::Triangle);
    let rgb = resized.to_rgb8();

    let tensor: Tensor = tract_ndarray::Array4::from_shape_fn((1, 3, 256, 256), |(_, c, y, x)| {
        rgb.get_pixel(x as u32, y as u32)[c] as f32 / 255.0
    }).into();

    let result = model.run(tvec!(tensor.into())).map_err(|e| e.to_string())?;
    let probs = result[0].to_array_view::<f32>().map_err(|e| e.to_string())?;

    let mut max_idx = 0;
    let mut max_val = f32::NEG_INFINITY;
    for (i, &val) in probs.iter().enumerate() {
        if val > max_val {
            max_val = val;
            max_idx = i;
        }
    }

    Ok(CLASSES[max_idx].to_string())
}

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    tauri::Builder::default()
        .plugin(tauri_plugin_dialog::init())
        .plugin(tauri_plugin_fs::init())
        .plugin(tauri_plugin_opener::init())
        .setup(|app| {

            let state = ModelState { model: RwLock::new(None) };
            app.manage(state);

            let handle = app.handle().clone();
            
            std::thread::spawn(move || {
                // On Android, Tauri prefixes bundled resources with "resources/"
                let model_bytes: &[u8] = include_bytes!("../plant-disease.onnx");

                let mut cursor = std::io::Cursor::new(model_bytes);
                
                let model = tract_onnx::onnx()
                    .model_for_read(&mut cursor)
                    .expect("Failed to parse ONNX bytes")
                    .with_input_fact(0, f32::fact(&[1, 3, 256, 256]).into()).unwrap()
                    .into_typed().unwrap()
                    .into_runnable().unwrap();

                let state = handle.state::<ModelState>();
                let mut model_lock = state.model.write().unwrap();
                *model_lock = Some(Arc::new(model));
                
                println!("✅ MODEL LOADED SUCCESSFULLY");
            });

            Ok(())
        })
        .invoke_handler(tauri::generate_handler![predict])
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}
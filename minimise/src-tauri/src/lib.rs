use tauri::{Manager, State};
use tract_nnef::prelude::*;
use std::sync::Arc;
use std::sync::RwLock;
use std::io::Cursor;

type Model = SimplePlan<TypedFact, Box<dyn TypedOp>, Graph<TypedFact, Box<dyn TypedOp>>>;

struct ModelState {
    model: RwLock<Option<Arc<Model>>>,
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
    let model_lock = state.model.read().unwrap();
    let model = model_lock.as_ref().ok_or("Model is still loading...")?;

    let img = image::load_from_memory(&image_bytes)
        .map_err(|e| format!("Image Load Error: {}", e))?;
    
    let resized = img.resize_exact(256, 256, image::imageops::FilterType::Triangle);
    let rgb = resized.to_rgb8();
    
    let tensor: Tensor = tract_ndarray::Array4::from_shape_fn((1, 3, 256, 256), |(_, c, y, x)| {
        rgb.get_pixel(x as u32, y as u32)[c] as f32 / 255.0
    }).into();

    let result = model.run(tvec!(tensor.into()))
        .map_err(|e| format!("Inference failed: {:#}", e))?;

    let probs = result[0].to_array_view::<f32>().map_err(|e| e.to_string())?;
    
    let (max_idx, _) = probs
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
        .unwrap();

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
                // Include NNEF tar as bytes
                let model_bytes: &[u8] = include_bytes!("../plant_disease.nnef.tar");
                let cursor = Cursor::new(model_bytes);
                
                let model = tract_nnef::nnef()
                    .with_tract_core()
                    .model_for_read(&mut std::io::BufReader::new(cursor))
                    .expect("Failed to parse NNEF")
                    .into_optimized()
                    .expect("Failed to optimize")
                    .into_runnable()
                    .expect("Failed to make runnable");

                let state = handle.state::<ModelState>();
                let mut model_lock = state.model.write().unwrap();
                *model_lock = Some(Arc::new(model));
                
                println!("✅ NNEF MODEL LOADED SUCCESSFULLY");
            });
            
            Ok(())
        })
        .invoke_handler(tauri::generate_handler![predict])
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}
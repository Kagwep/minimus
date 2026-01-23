use tract_onnx::prelude::*;
use tract_nnef::prelude::*;

fn main() -> TractResult<()> {
    // 1. Load ONNX
    let mut model = tract_onnx::onnx()
        .model_for_path("plant-disease.onnx")?
        .with_input_fact(0, f32::fact(&[1, 3, 256, 256]).into())?
        .into_typed()?;

    // 2. SANITIZE FILENAMES (The Fix)
    // We loop through every node and replace ":" with "_"
    // Sanitize ALL node names - remove leading slashes and replace problematic chars
    for node in model.nodes_mut() {
        node.name = node.name
            .trim_start_matches('/')  // Remove leading slash
            .replace("/", "_")
            .replace(":", "_")
            .replace(".", "_");
    }
    // 3. Clean up the graph
    let model = model.into_decluttered()?;

    // 4. Export to NNEF
    let nnef_path = "plant_disease_nnef";
    
    // Create the directory manually if it helps, 
    // but write_to_dir usually handles it.
    tract_nnef::nnef().write_to_dir(&model, nnef_path)?;

    println!("✅ Fixed! NNEF exported to: {}", nnef_path);
    Ok(())
}
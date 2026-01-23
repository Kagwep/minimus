use tract_onnx::prelude::*;
use tract_nnef::prelude::*;
use std::fs::File;

fn main() -> TractResult<()> {
    let mut model = tract_onnx::onnx()
        .model_for_path("plant-disease.onnx")?
        .with_input_fact(0, f32::fact(&[1, 3, 256, 256]).into())?
        .into_typed()?;

    for node in model.nodes_mut() {
        node.name = node.name.trim_start_matches('/').to_string();
    }

    let model = model.into_decluttered()?;

    // Export as tar for easy bundling
    let file = File::create("plant_disease.nnef.tar")?;
    tract_nnef::nnef()
        .with_tract_core()
        .write_to_tar(&model, file)?;

    println!("✅ Exported plant_disease.nnef.tar");
    Ok(())
}
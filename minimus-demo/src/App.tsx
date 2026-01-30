import { useState } from "react";
import { invoke } from "@tauri-apps/api/core";
import { open } from "@tauri-apps/plugin-dialog";
import { readFile } from "@tauri-apps/plugin-fs";
import "./App.css"

function App() {
  const [imagePath, setImagePath] = useState<string | null>(null);
  const [preview, setPreview] = useState<string | null>(null);
  const [prediction, setPrediction] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);

  const selectImage = async () => {
    try {
      const file = await open({
        multiple: false,
        filters: [{ name: "Images", extensions: ["jpg", "jpeg", "png"] }],
      });

      if (file && typeof file === "string") {
        setImagePath(file);
        
        // 1. Get the raw bytes (We know this works because your prediction uses it!)
        const contents = await readFile(file);
        
        // 2. Wrap them in a Blob (Much faster than Base64)
        const blob = new Blob([contents], { type: "image/jpeg" });
        
        // 3. Create a URL that the <img> tag can actually use
        const url = URL.createObjectURL(blob);
        
        setPreview(url);
      }
    } catch (err) {
      console.error("Selection error:", err);
    }
  };

  const runPrediction = async () => {
    if (!imagePath) return;
    setLoading(true);
    try {
      // Read the bytes in JS (which already works for content://)
      const bytes = await readFile(imagePath);
      // Send bytes to Rust instead of the path
      const result = await invoke<string>("predict", { imageBytes: Array.from(bytes) });
      setPrediction(result);
    } catch (err) {
      setPrediction(`Status: ${err}`);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div style={styles.container}>
      <h1 style={styles.title}>🌿 Minimus Plant Disease Detector</h1>


      <button onClick={selectImage} style={styles.button}>
        Select Image
      </button>

      {preview && (
        <div style={styles.imageContainer}>
          <img 
            src={preview} 
            style={styles.image} 
            alt="Selected" 
            onError={() => console.error("Image failed to load at protocol: ", preview)}
          />
        </div>
      )}

      {imagePath && (
        <div style={{ marginTop: 20 }}>
          <button 
            onClick={runPrediction} 
            style={{...styles.buttonPrimary, opacity: loading ? 0.7 : 1}} 
            disabled={loading}
          >
            {loading ? <span className="spinner">Analyzing...</span> : "Detect Disease"}
          </button>
        </div>
      )}

      {prediction && (
        <div style={styles.result}>
          <h2>Result:</h2>
          <p style={styles.prediction}>{prediction.replace(/___/g, " - ")}</p>
        </div>
      )}
    </div>
  );
}


const styles: Record<string, React.CSSProperties> = {
  container: { maxWidth: 500, margin: "0 auto", padding: 40, textAlign: "center", fontFamily: "system-ui, sans-serif" },
  title: { fontSize: 28, marginBottom: 8 },
  subtitle: { color: "#666", marginBottom: 32 },
  button: { padding: "12px 24px", fontSize: 16, cursor: "pointer", border: "1px solid #ccc", borderRadius: 8, background: "#fff" },
  buttonPrimary: { padding: "12px 24px", fontSize: 16, cursor: "pointer", border: "none", borderRadius: 8, background: "#22c55e", color: "#fff" },
  imageContainer: { marginTop: 20, display: 'flex', justifyContent: 'center' },
  image: { maxWidth: "100%", height: "auto", maxHeight: 300, borderRadius: 8, border: "2px solid #eee" },
  result: { marginTop: 24, padding: 20, background: "#f0fdf4", borderRadius: 8 },
  prediction: { fontSize: 20, fontWeight: "bold", color: "#166534" },
};

export default App;
import { useState } from "react";
import { invoke, convertFileSrc } from "@tauri-apps/api/core";
import { open } from "@tauri-apps/plugin-dialog";

import "./App.css"

function App() {
  const [imagePath, setImagePath] = useState<string | null>(null);
  const [preview, setPreview] = useState<string | null>(null);
  const [prediction, setPrediction] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);

    const selectImage = async () => {
    const file = await open({
        multiple: false,
        filters: [{ name: "Images", extensions: ["jpg", "jpeg", "png"] }],
    });

    if (file) {
        setImagePath(file); // This raw path is for the Rust backend
        
        // THIS is the fix for the broken icon
        const assetUrl = convertFileSrc(file); 
        setPreview(assetUrl); 
        
        setPrediction(null);
    }
    };
  const runPrediction = async () => {
    if (!imagePath) return;
    setLoading(true);
    try {
      const result = await invoke<string>("predict", { imagePath });
      setPrediction(result);
    } catch (err) {
      // If the model is still loading, the Rust error from my previous message 
      // ("Model is still loading...") will appear here.
      setPrediction(`Status: ${err}`);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div style={styles.container}>
      <h1 style={styles.title}>🌿 Minimus Plant Disease Detector</h1>
      <p style={styles.subtitle}>Optimized ML for low-end devices</p>

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
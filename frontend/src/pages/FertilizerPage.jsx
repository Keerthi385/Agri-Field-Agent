import React, { useState } from "react";
import axios from "axios";

export default function FertilizerPage() {
  const [crop, setCrop] = useState("");
  const [soil, setSoil] = useState("");
  const [condition, setCondition] = useState("");
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [voiceUrl, setVoiceUrl] = useState(""); // 🎧 for voice note

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (!crop || !soil || !condition) {
      alert("Please fill all fields!");
      return;
    }

    setLoading(true);
    try {
      // 1️⃣ Get fertilizer recommendation
      const res = await axios.post("http://127.0.0.1:8000/fertilizer", {
        crop,
        soil,
        condition,
      });
      setResult(res.data);

      // 2️⃣ Get Telugu voice note for this crop context
      const voiceRes = await axios.post("http://127.0.0.1:8000/voice", {
        crop,
        disease: `${crop} healthy`, // default context if no disease detected
        soil,
        condition,
      });
      setVoiceUrl(voiceRes.data.audio_url);
    } catch (err) {
      console.error(err);
      alert("Error fetching recommendation or voice note!");
    }
    setLoading(false);
  };

  return (
    <div>
      <h1 className="page-title">🌾 Fertilizer Recommendation</h1>
      <div className="card half">
        <form onSubmit={handleSubmit} className="stack">
          <input
            className="input"
            type="text"
            placeholder="Crop (e.g., Rice)"
            value={crop}
            onChange={(e) => setCrop(e.target.value)}
          />
          <input
            className="input"
            type="text"
            placeholder="Soil Type (e.g., Red)"
            value={soil}
            onChange={(e) => setSoil(e.target.value)}
          />
          <input
            className="input"
            type="text"
            placeholder="Condition (e.g., Normal)"
            value={condition}
            onChange={(e) => setCondition(e.target.value)}
          />
          <button type="submit" className="btn">
            {loading ? "Fetching..." : "Get Recommendation"}
          </button>
        </form>
      </div>

      {result && (
        <div className="card" style={{ marginTop: "20px" }}>
          <h2>
            🌱 Recommended Fertilizer:{" "}
            <span className="pill">{result.fertilizer}</span>
          </h2>
          <p className="muted">{result.advice}</p>

          {voiceUrl && (
            <div style={{ marginTop: "20px" }}>
              <h3>🎧 Listen to Telugu Advisory:</h3>
              <audio controls src={voiceUrl}></audio>
            </div>
          )}
        </div>
      )}
    </div>
  );
}

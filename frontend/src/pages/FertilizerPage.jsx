import React, { useState } from "react";
import axios from "axios";

export default function FertilizerPage() {
  const [crop, setCrop] = useState("");
  const [soil, setSoil] = useState("");
  const [condition, setCondition] = useState("");
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (!crop || !soil || !condition) {
      alert("Please fill all fields!");
      return;
    }

    setLoading(true);
    try {
      const res = await axios.post("http://127.0.0.1:8000/fertilizer", {
        crop,
        soil,
        condition,
      });
      setResult(res.data);
    } catch (err) {
      console.error(err);
      alert("Error fetching recommendation!");
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
          <h2>🌱 Recommended Fertilizer: <span className="pill">{result.fertilizer}</span></h2>
          <p className="muted">{result.advice}</p>
        </div>
      )}
    </div>
  );
}

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
    <div style={{ textAlign: "center", marginTop: "80px" }}>
      <h1>🌾 Fertilizer Recommendation</h1>
      <form onSubmit={handleSubmit} style={{ marginTop: "20px" }}>
        <input
          type="text"
          placeholder="Crop (e.g., Rice)"
          value={crop}
          onChange={(e) => setCrop(e.target.value)}
          style={{ margin: "5px", padding: "8px" }}
        />
        <input
          type="text"
          placeholder="Soil Type (e.g., Red)"
          value={soil}
          onChange={(e) => setSoil(e.target.value)}
          style={{ margin: "5px", padding: "8px" }}
        />
        <input
          type="text"
          placeholder="Condition (e.g., Normal)"
          value={condition}
          onChange={(e) => setCondition(e.target.value)}
          style={{ margin: "5px", padding: "8px" }}
        />
        <button type="submit" style={{ marginTop: "10px" }}>
          {loading ? "Fetching..." : "Get Recommendation"}
        </button>
      </form>

      {result && (
        <div style={{ marginTop: "30px" }}>
          <h2>🌱 Recommended Fertilizer: {result.fertilizer}</h2>
          <p>{result.advice}</p>
        </div>
      )}
    </div>
  );
}

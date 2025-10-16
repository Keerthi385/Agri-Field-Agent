import React, { useState } from "react";
import axios from "axios";

export default function ContextPage() {
  const [context, setContext] = useState(null);
  const [fertilizer, setFertilizer] = useState("");
  const [loading, setLoading] = useState(false);

  // Fetch location + weather context
  const getContext = async () => {
    if (!navigator.geolocation) {
      alert("Geolocation not supported!");
      return;
    }

    navigator.geolocation.getCurrentPosition(async (pos) => {
      const lat = pos.coords.latitude;
      const lon = pos.coords.longitude;
      try {
        const res = await axios.get(`http://127.0.0.1:8000/context?lat=${lat}&lon=${lon}`);
        setContext(res.data);
      } catch (err) {
        console.error(err);
        alert("Error fetching context data");
      }
    });
  };

  // Request fertilizer recommendation
  const getFertilizer = async () => {
    if (!context) return alert("Please fetch context first!");

    setLoading(true);
    try {
      const res = await axios.post("http://127.0.0.1:8000/fertilizer", {
        crop: "potato",                     // you can change this or make input-based later
        soil: context.soil || "red",
        condition: context.condition || "normal",
      });
      setFertilizer(res.data.fertilizer + " — " + res.data.advice);
    } catch (err) {
      console.error(err);
      alert("Error getting fertilizer suggestion");
    }
    setLoading(false);
  };

  return (
    <div style={{ textAlign: "center", marginTop: "80px" }}>
      <h1>📍 Context Data</h1>

      {!context && (
        <button onClick={getContext}>Detect My Location & Context</button>
      )}

      {context && (
        <div style={{ marginTop: "20px" }}>
          <p>Temperature: {context.temperature}°C</p>
          <p>Wind Speed: {context.windspeed} km/h</p>
          <p>Soil Type: {context.soil}</p>
          <p>Condition: {context.condition}</p>
        </div>
      )}

      {context && (
        <button
          onClick={getFertilizer}
          style={{ marginTop: "20px", display: "block", marginLeft: "auto", marginRight: "auto" }}
        >
          {loading ? "Analyzing..." : "Get Fertilizer Suggestion"}
        </button>
      )}

      {fertilizer && (
        <div style={{ marginTop: "30px" }}>
          <h2>🌾 Recommended: {fertilizer}</h2>
        </div>
      )}
    </div>
  );
}

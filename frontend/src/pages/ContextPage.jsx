import React, { useState } from "react";
import axios from "axios";

export default function ContextPage() {
  const [context, setContext] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  const getContext = async () => {
    setError("");
    setContext(null);
    setLoading(true);

    if (!navigator.geolocation) {
      setError("Geolocation not supported in your browser.");
      setLoading(false);
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
        setError("Error fetching weather data.");
      }
      setLoading(false);
    });
  };

  return (
    <div>
      <h1 className="page-title">📍 Field Context</h1>

      {!context && !error && (
        <button onClick={getContext} className="btn">
          {loading ? "Fetching..." : "Detect My Location & Context"}
        </button>
      )}

      {error && (
        <p style={{ color: "#a40000", marginTop: "20px" }}>{error}</p>
      )}

      {context && (
        <div className="card" style={{ marginTop: "20px" }}>
          <p>🌡️ <b>Temperature:</b> {context.temperature}°C</p>
          <p>💨 <b>Wind Speed:</b> {context.windspeed} km/h</p>
          <p>🌱 <b>Soil Type:</b> {context.soil}</p>
          <p>☁️ <b>Condition:</b> {context.condition}</p>
        </div>
      )}
    </div>
  );
}

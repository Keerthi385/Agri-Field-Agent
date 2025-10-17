import React from "react";
import { Link } from "react-router-dom";

export default function DashboardPage() {
  return (
    <div style={{ textAlign: "center", marginTop: "70px" }}>
      <h1>🌾 Agri Advisory Field Agent Dashboard</h1>
      <p style={{ color: "gray", fontSize: "18px" }}>
        Smart farming assistant to detect plant diseases, suggest fertilizers, and analyze field context.
      </p>

      <div
        style={{
          display: "flex",
          justifyContent: "center",
          gap: "30px",
          marginTop: "60px",
        }}
      >
        <Link to="/disease" style={cardStyle}>
          <h2>🩺 Disease Detection</h2>
          <p>Upload a leaf image and detect the disease instantly.</p>
        </Link>

        <Link to="/fertilizer" style={cardStyle}>
          <h2>🌱 Fertilizer Suggestion</h2>
          <p>Get fertilizer recommendations based on soil and crop type.</p>
        </Link>

        <Link to="/context" style={cardStyle}>
          <h2>📍 Field Context</h2>
          <p>Analyze your local weather and soil data automatically.</p>
        </Link>
      </div>
    </div>
  );
}

const cardStyle = {
  border: "1px solid #ddd",
  borderRadius: "12px",
  padding: "25px",
  width: "260px",
  textDecoration: "none",
  color: "black",
  boxShadow: "0 4px 8px rgba(0,0,0,0.1)",
  transition: "transform 0.2s ease-in-out",
};

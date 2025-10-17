import React, { useState } from "react";
import axios from "axios";
import { BrowserRouter as Router, Routes, Route, Link } from "react-router-dom";
import FertilizerPage from "./pages/FertilizerPage";
import ContextPage from "./pages/ContextPage";
import DashboardPage from "./pages/DashboardPage";
import MarketPage from "./pages/MarketPage";
import "./App.css";



function DiseaseDetectionPage() {
  const [file, setFile] = useState(null);
  const [prediction, setPrediction] = useState("");
  const [loading, setLoading] = useState(false);
  const [details, setDetails] = useState(null);


  const handleFileChange = (e) => {
    setFile(e.target.files[0]);
  };

  const handleSubmit = async () => {
    if (!file) {
      alert("Please select an image first!");
      return;
    }

    const formData = new FormData();
    formData.append("file", file);
    setLoading(true);

    try {
      const res = await axios.post("http://127.0.0.1:8000/predict", formData, {
        headers: { "Content-Type": "multipart/form-data" },
      });
      setPrediction(res.data.prediction);
      setDetails(res.data.details);

    } catch (err) {
      console.error(err);
      alert("Error uploading image!");
    }
    setLoading(false);
  };

  return (
    <div style={{ textAlign: "center", marginTop: "80px" }}>
      <h1>🌿 Plant Disease Detection</h1>
      <input type="file" onChange={handleFileChange} />
      <br />
      <button onClick={handleSubmit} style={{ marginTop: "20px" }}>
        {loading ? "Predicting..." : "Predict Disease"}
      </button>
      <div style={{ marginTop: "30px" }}>
        {/* {prediction && <h2>🩺 Result: {prediction}</h2>} */}
        {prediction && (
          <div style={{ marginTop: "30px", lineHeight: "1.8" }}>
            <h2>🩺 Result: {prediction}</h2>
            {details && (
              <div style={{ marginTop: "15px", textAlign: "left", width: "60%", margin: "auto" }}>
                <p><b>Cause:</b> {details.cause}</p>
                <p><b>Treatment:</b> {details.treatment}</p>
                <p><b>Prevention:</b> {details.prevention}</p>
              </div>
            )}
          </div>
        )}

      </div>
    </div>
  );
}

function App() {
  return (
    <Router>
      <nav className="nav">
        <div className="brand">
          <span>🌿</span>
          <span>Agri Field Agent</span>
        </div>
        <div className="nav-links">
          <Link to="/" className="nav-link">Dashboard</Link>
          <Link to="/disease" className="nav-link">Disease</Link>
          <Link to="/fertilizer" className="nav-link">Fertilizer</Link>
          <Link to="/context" className="nav-link">Context</Link>
          <Link to="/market" className="nav-link">Market</Link>
        </div>
      </nav>

      <div className="container">
        <Routes>
          <Route path="/" element={<DashboardPage />} />
          <Route path="/disease" element={<DiseaseDetectionPage />} />
          <Route path="/fertilizer" element={<FertilizerPage />} />
          <Route path="/context" element={<ContextPage />} />
          <Route path="/market" element={<MarketPage />} />
        </Routes>
      </div>
    </Router>
  );
}

export default App;

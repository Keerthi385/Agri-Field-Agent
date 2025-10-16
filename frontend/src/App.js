import React, { useState } from "react";
import axios from "axios";
import { BrowserRouter as Router, Routes, Route, Link } from "react-router-dom";
import FertilizerPage from "./pages/FertilizerPage";
import ContextPage from "./pages/ContextPage";
import DashboardPage from "./pages/DashboardPage";


function DiseaseDetectionPage() {
  const [file, setFile] = useState(null);
  const [prediction, setPrediction] = useState("");
  const [loading, setLoading] = useState(false);

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
        {prediction && <h2>🩺 Result: {prediction}</h2>}
      </div>
    </div>
  );
}

function App() {
  return (
    <Router>
      <div style={{ textAlign: "center", marginTop: "30px" }}>
        {/* <Link to="/" style={{ marginRight: "20px" }}>Disease Detection</Link>
        <Link to="/fertilizer" style={{ marginRight: "20px" }}>Fertilizer</Link>
        <Link to="/context">Context Info</Link> */}

        <Link to="/" style={{ marginRight: "20px" }}>Dashboard</Link>
        <Link to="/disease" style={{ marginRight: "20px" }}>Disease Detection</Link>
        <Link to="/fertilizer" style={{ marginRight: "20px" }}>Fertilizer</Link>
        <Link to="/context">Context</Link>



        {/* <Routes>
          <Route path="/" element={<DiseaseDetectionPage />} />
          <Route path="/fertilizer" element={<FertilizerPage />} />
          <Route path="/context" element={<ContextPage />} />
        </Routes> */}

        <Routes>
          <Route path="/" element={<DashboardPage />} />
          <Route path="/disease" element={<DiseaseDetectionPage />} />
          <Route path="/fertilizer" element={<FertilizerPage />} />
          <Route path="/context" element={<ContextPage />} />
        </Routes>

      </div>
    </Router>
  );
}

export default App;

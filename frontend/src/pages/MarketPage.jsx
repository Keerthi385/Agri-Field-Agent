import React, { useState, useEffect } from "react";
import axios from "axios";

export default function MarketPage() {
  const [prices, setPrices] = useState(null);
  const [geoError, setGeoError] = useState("");

  useEffect(() => {
    const fetchWithCoords = async (lat, lon) => {
      try {
        const res = await axios.get(`http://127.0.0.1:8000/market`, {
          params: lat != null && lon != null ? { lat, lon } : {}
        });
        setPrices(res.data);
      } catch (e) {
        alert("Error fetching market prices");
      }
    };

    if (navigator.geolocation) {
      navigator.geolocation.getCurrentPosition(
        (pos) => {
          const { latitude, longitude } = pos.coords;
          fetchWithCoords(latitude, longitude);
        },
        () => {
          setGeoError("Location permission denied. Showing default prices.");
          fetchWithCoords();
        },
        { enableHighAccuracy: false, timeout: 7000, maximumAge: 300000 }
      );
    } else {
      setGeoError("Geolocation not supported. Showing default prices.");
      fetchWithCoords();
    }
  }, []);

  return (
    <div style={{ textAlign: "center", marginTop: "80px" }}>
      <h1>📈 Market Prices</h1>
      {geoError && <p style={{ color: "#aa0000" }}>{geoError}</p>}
      {!prices ? (
        <p>Loading...</p>
      ) : (
        <table
          style={{
            margin: "20px auto",
            borderCollapse: "collapse",
            width: "60%",
          }}
        >
          <thead>
            <tr style={{ backgroundColor: "#4CAF50", color: "white" }}>
              <th style={th}>Crop</th>
              <th style={th}>Price</th>
              <th style={th}>Market</th>
            </tr>
          </thead>
          <tbody>
            {Object.entries(prices).map(([crop, info]) => (
              <tr key={crop}>
                <td style={td}>{crop}</td>
                <td style={td}>{info.price}</td>
                <td style={td}>{info.market}</td>
              </tr>
            ))}
          </tbody>
        </table>
      )}
    </div>
  );
}

const th = { padding: "10px", border: "1px solid #ddd" };
const td = { padding: "10px", border: "1px solid #ddd" };

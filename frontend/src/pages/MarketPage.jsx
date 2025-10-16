import React, { useState, useEffect } from "react";
import axios from "axios";

export default function MarketPage() {
  const [prices, setPrices] = useState(null);

  useEffect(() => {
    axios.get("http://127.0.0.1:8000/market")
      .then(res => setPrices(res.data))
      .catch(() => alert("Error fetching market prices"));
  }, []);

  return (
    <div style={{ textAlign: "center", marginTop: "80px" }}>
      <h1>📈 Market Prices</h1>
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

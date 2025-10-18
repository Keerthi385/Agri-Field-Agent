import React, { useState, useEffect } from "react";
import axios from "axios";

export default function MarketPage() {
  // State declarations (move all to top)
  const [prices, setPrices] = useState(null);
  const [nearby, setNearby] = useState(null);
  const [loadingNearby, setLoadingNearby] = useState(false);
  const [radius, setRadius] = useState(50); // default to 50km
  const [userLocation, setUserLocation] = useState(null);
  const [selectedState, setSelectedState] = useState("");
  const [selectedCity, setSelectedCity] = useState("");
  const [selectedDistrict, setSelectedDistrict] = useState("");

  // Example data for location selection
  const states = ["Telangana", "Maharashtra", "Andhra Pradesh"];
  const cities = {
    "Telangana": ["Hyderabad"],
    "Maharashtra": ["Pune", "Mumbai"],
    "Andhra Pradesh": ["Vizag", "Vijayawada", "Guntur"]
  };
  const districts = {
    "Hyderabad": ["Hyderabad", "Secunderabad", "Medchal-Malkajgiri", "Ranga Reddy", "Sangareddy"],
    "Pune": ["Pune City", "Haveli", "Mulshi", "Shirur"],
    "Mumbai": ["Mumbai City", "Mumbai Suburban", "Thane", "Navi Mumbai"],
    "Vizag": ["Visakhapatnam", "Anakapalle", "Bheemunipatnam"],
    "Vijayawada": ["Vijayawada Central", "Vijayawada North", "Vijayawada East"],
    "Guntur": ["Guntur East", "Guntur West", "Tenali", "Mangalagiri"]
  };

  // Handle location change
  const handleStateChange = (e) => {
    setSelectedState(e.target.value);
    setSelectedCity("");
    setSelectedDistrict("");
  };
  const handleCityChange = (e) => {
    setSelectedCity(e.target.value);
    setSelectedDistrict("");
  };
  const handleDistrictChange = (e) => {
    setSelectedDistrict(e.target.value);
  };

  // Fetch market prices for selected location
  useEffect(() => {
    if (selectedState && selectedCity && selectedDistrict) {
      setLoadingNearby(true);
      axios.get("http://127.0.0.1:8000/market/nearby", {
        params: {
          state: selectedState,
          city: selectedCity,
          district: selectedDistrict,
          radius_km: radius
        }
      })
        .then(res => setNearby(res.data))
        .catch(() => {
          setNearby(null);
        })
        .finally(() => setLoadingNearby(false));
    }
  }, [selectedState, selectedCity, selectedDistrict, radius]);

  useEffect(() => {
    axios.get("http://127.0.0.1:8000/market")
      .then(res => setPrices(res.data))
      .catch(() => alert("Error fetching market prices"));

    let watchId;
    if (navigator.geolocation) {
      setLoadingNearby(true);
      watchId = navigator.geolocation.watchPosition(
        (pos) => {
          const { latitude, longitude } = pos.coords;
          setUserLocation({ latitude, longitude });
        },
        (err) => {
          setLoadingNearby(false);
          setNearby(null);
        },
        { enableHighAccuracy: false, timeout: 10000, maximumAge: 10000 }
      );
    }
    // Cleanup watchPosition on unmount
    return () => {
      if (navigator.geolocation && watchId) {
        navigator.geolocation.clearWatch(watchId);
      }
    };
  }, []);

  // Fetch nearby mandis whenever userLocation or radius changes
  useEffect(() => {
    if (userLocation) {
      setLoadingNearby(true);
      axios.get("http://127.0.0.1:8000/market/nearby", {
        params: {
          lat: userLocation.latitude,
          lon: userLocation.longitude,
          radius_km: radius
        }
      })
        .then(res => setNearby(res.data))
        .catch(() => {
          setNearby(null);
        })
        .finally(() => setLoadingNearby(false));
    }
  }, [userLocation, radius]);

  // Refetch nearby mandis when radius changes (if location is available)
  const handleRadiusChange = (e) => {
    setRadius(Number(e.target.value));
  };

  return (
    <div style={{ textAlign: "center", marginTop: "80px" }}>
      <h1>📈 Market Prices</h1>
      <div style={{ margin: "20px 0" }}>
        <label htmlFor="radius-select" style={{ fontWeight: "bold" }}>Nearby Radius: </label>
        <select id="radius-select" value={radius} onChange={handleRadiusChange} style={{ marginLeft: 8, padding: 4 }}>
          <option value={50}>50 km</option>
          <option value={100}>100 km</option>
        </select>
      </div>

      <div style={{ margin: "20px 0" }}>
        <label htmlFor="state-select" style={{ fontWeight: "bold" }}>State: </label>
        <select id="state-select" value={selectedState} onChange={handleStateChange} style={{ marginLeft: 8, padding: 4 }}>
          <option value="">Select State</option>
          {states.map((state) => (
            <option key={state} value={state}>{state}</option>
          ))}
        </select>

        {selectedState && (
          <>
            <label htmlFor="city-select" style={{ fontWeight: "bold", marginLeft: 16 }}>City: </label>
            <select id="city-select" value={selectedCity} onChange={handleCityChange} style={{ marginLeft: 8, padding: 4 }}>
              <option value="">Select City</option>
              {cities[selectedState].map((city) => (
                <option key={city} value={city}>{city}</option>
              ))}
            </select>
          </>
        )}

        {selectedCity && (
          <>
            <label htmlFor="district-select" style={{ fontWeight: "bold", marginLeft: 16 }}>District: </label>
            <select id="district-select" value={selectedDistrict} onChange={handleDistrictChange} style={{ marginLeft: 8, padding: 4 }}>
              <option value="">Select District</option>
              {districts[selectedCity].map((district) => (
                <option key={district} value={district}>{district}</option>
              ))}
            </select>
          </>
        )}
      </div>

      {!prices ? (
        <p>Loading...</p>
      ) : (
        <div>
        {loadingNearby ? <p>Fetching market prices for selected location...</p> : null}

        <table
          style={{
            margin: "20px auto",
            borderCollapse: "collapse",
            width: "70%",
          }}
        >
          <thead>
            <tr style={{ backgroundColor: "#1976d2", color: "white" }}>
              <th style={th}>Crop</th>
              <th style={th}>Price</th>
              <th style={th}>Market</th>
            </tr>
          </thead>
          <tbody>
            {nearby && nearby.count > 0
              ? nearby.nearby.flatMap((m) =>
                  Object.entries(m.prices).map(([crop, price]) => (
                    <tr key={m.name + crop}>
                      <td style={td}>{crop}</td>
                      <td style={td}>{`₹${price}`}</td>
                      <td style={td}>{m.name}</td>
                    </tr>
                  ))
                )
              : Object.entries(prices).map(([crop, info]) => (
                  <tr key={crop}>
                    <td style={td}>{crop}</td>
                    <td style={td}>{info.price}</td>
                    <td style={td}>{info.market}</td>
                  </tr>
                ))}
          </tbody>
        </table>
        </div>
      )}
    </div>
  );
}

const th = { padding: "10px", border: "1px solid #ddd" };
const td = { padding: "10px", border: "1px solid #ddd" };
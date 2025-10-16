# backend/fertilizer_advice.py

def get_fertilizer_recommendation(crop: str, soil_type: str):
    crop = crop.strip().lower()
    soil_type = soil_type.strip().lower()

    rules = {
        "rice": {"loamy": "Urea 100kg/ha + DAP 60kg/ha",
                "clay":  "Urea 90kg/ha + Potash 40kg/ha"},
        "wheat": {"sandy": "NPK 20-20-0 100kg/ha",
                  "loamy": "Urea 80kg/ha + SSP 40kg/ha"},
        "potato": {"sandy": "NPK 15-15-15 150kg/ha",
                  "clay":  "Compost 5t/ha + DAP 100kg/ha"},
    }

    return rules.get(crop, {}).get(soil_type, "Use balanced NPK fertilizer (e.g., 20-20-20)")

from fastapi import FastAPI, Query, UploadFile, File
from fastapi.responses import JSONResponse
import torch
from torchvision import models, transforms
from PIL import Image
import io
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import requests
from torchvision import datasets
import json
from gtts import gTTS
from googletrans import Translator
import os
from fastapi.staticfiles import StaticFiles
import numpy as np
import math

app = FastAPI()

# app.mount("/voice_notes", StaticFiles(directory="voice_notes"), name="voice_notes")
os.makedirs("voice_notes", exist_ok=True)
app.mount("/voice_notes", StaticFiles(directory="voice_notes"), name="voice_notes")


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # allow all origins (frontend)
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)



# =====================
# Load Model
# =====================
model_path = "models/plant_disease_model.pth"
data_dir = "data/plantvillage"

transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
])

# Load classes from dataset folder

dataset = datasets.ImageFolder(root=data_dir, transform=transform)
class_names = dataset.classes

# Load model
model = models.mobilenet_v2(weights="IMAGENET1K_V1")
model.classifier[1] = torch.nn.Linear(model.last_channel, len(class_names))
model.load_state_dict(torch.load(model_path, map_location="cpu"))
model.eval()

# =====================
# Predict endpoint
# =====================

import numpy as np

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    image_bytes = await file.read()
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")

    # --- Robust leaf check using Excess Green (ExG) ---
    # Resize for speed, keep aspect ratio
    img_small = img.resize((256, int(256 * img.height / img.width))) if img.width > img.height else img.resize((int(256 * img.width / img.height), 256))
    np_img = np.array(img_small).astype(np.float32)

    R = np_img[..., 0]
    G = np_img[..., 1]
    B = np_img[..., 2]

    # Compute Excess Green index
    ExG = 2 * G - R - B

    # Normalize ExG roughly to 0-255 by clipping
    ExG_clip = np.clip(ExG, -255, 255)
    # threshold: consider pixel vegetation if ExG > exg_thresh
    exg_thresh = 20.0   # you can tune this (lower = more permissive)
    veg_mask = ExG_clip > exg_thresh

    green_ratio = veg_mask.mean()  # fraction of pixels classified as vegetation

    # Also compute overall brightness to reject very dark/blank images
    brightness = np.mean((R + G + B) / 3.0)

    # Determine pass/fail with tolerant thresholds (tune if necessary)
    MIN_GREEN_RATIO = 0.03   # require at least 3% vegetation pixels (low to allow diseased leaves)
    MIN_BRIGHTNESS = 20.0    # avoid almost-black images

    if brightness < MIN_BRIGHTNESS or green_ratio < MIN_GREEN_RATIO:
        # borderline: if green_ratio is slightly below, allow fallback with warning
        if green_ratio >= 0.01 and brightness >= 15.0:
            # fallback: attempt prediction but mark as uncertain
            img_t = transform(img).unsqueeze(0)
            with torch.no_grad():
                outputs = model(img_t)
                _, pred = outputs.max(1)
            result = class_names[pred.item()]

            try:
                with open("disease_library.json", "r") as f:
                    disease_info = json.load(f)
                info = disease_info.get(result, {})
            except Exception:
                info = {}

            return JSONResponse({
                "prediction": result,
                "details": info,
                "warning": "Image had low vegetation signal — result may be unreliable. Please upload a clearer leaf photo if possible."
            })
        # hard reject
        return JSONResponse({
            "prediction": "Not a leaf image",
            "details": {"advice": "Please upload a clear leaf image for disease detection."}
        })

    # Passed the check — continue with normal prediction
    img_t = transform(img).unsqueeze(0)
    with torch.no_grad():
        outputs = model(img_t)
        _, pred = outputs.max(1)

    result = class_names[pred.item()]

    # 🧠 Load disease info
    try:
        with open("disease_library.json", "r") as f:
            disease_info = json.load(f)
        info = disease_info.get(result, {})
    except Exception:
        info = {}

    return JSONResponse({
        "prediction": result,
        "details": info
    })


class FertilizerRequest(BaseModel):
    crop: str
    soil: str
    condition: str

@app.post("/fertilizer")
def recommend_fertilizer(data: FertilizerRequest):
    crop = data.crop.lower()
    soil = data.soil.lower()
    condition = data.condition.lower()

    # simple rule-based logic
    if soil == "black" and crop == "cotton":
        fertilizer = "Urea and DAP"
    elif soil == "sandy" and crop == "wheat":
        fertilizer = "Ammonium Sulphate"
    elif soil == "red" and crop == "rice":
        fertilizer = "Super Phosphate"
    else:
        fertilizer = "NPK Mixture (General Purpose)"

    return {"fertilizer": fertilizer, "advice": f"For {crop} in {soil} soil under {condition} conditions."}


@app.get("/context")
def get_context(lat: float = Query(...), lon: float = Query(...)):
    """Fetch live weather + simple soil and condition info."""
    try:
        # ✅ Fetch real-time weather data
        url = f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}&current_weather=true"
        r = requests.get(url)
        data = r.json()
        weather = data["current_weather"]

        temperature = weather["temperature"]
        windspeed = weather["windspeed"]

        # 🧠 Simple logic for demo
        if temperature < 20:
            condition = "cool"
        elif temperature < 30:
            condition = "normal"
        else:
            condition = "hot"

        # 🌱 Basic soil inference (you can extend later)
        if lat > 20:
            soil = "red"
        else:
            soil = "black"

        return {
            "temperature": temperature,
            "windspeed": windspeed,
            "soil": soil,
            "condition": condition
        }

    except Exception as e:
        return {"error": str(e)}
    

@app.get("/market")
def get_market_prices():
    """Return current crop market prices (mock/demo)."""
    prices = {
        "rice": {"price": "₹1800 / quintal", "market": "Nizamabad"},
        "wheat": {"price": "₹2100 / quintal", "market": "Kurnool"},
        "cotton": {"price": "₹6400 / quintal", "market": "Warangal"},
        "tomato": {"price": "₹2400 / quintal", "market": "Madurai"},
        "groundnut": {"price": "₹5200 / quintal", "market": "Anantapur"}
    }
    return prices


class AdvisoryRequest(BaseModel):
    crop: str
    disease: str
    soil: str
    condition: str

@app.post("/advisory")
def generate_advisory(data: AdvisoryRequest):
    """Combine disease, fertilizer, weather, and price data."""
    crop = data.crop.lower()
    disease = data.disease
    soil = data.soil
    condition = data.condition

    # 🌱 Get fertilizer advice (reuse existing logic)
    if soil == "black" and crop == "cotton":
        fertilizer = "Urea and DAP"
    elif soil == "sandy" and crop == "wheat":
        fertilizer = "Ammonium Sulphate"
    elif soil == "red" and crop == "rice":
        fertilizer = "Super Phosphate"
    else:
        fertilizer = "NPK Mixture (General Purpose)"

    # 📘 Get disease info
    import json
    try:
        with open("disease_library.json", "r") as f:
            disease_data = json.load(f)
        disease_info = disease_data.get(disease, {})
    except Exception:
        disease_info = {}

    # 📈 Market prices (mock for now)
    market_prices = {
        "potato": "₹1800 / quintal",
        "rice": "₹2100 / quintal",
        "cotton": "₹6400 / quintal"
    }
    price = market_prices.get(crop, "₹2000 / quintal")

    # 🧠 Generate unified advisory
    advisory = (
        f"Your {crop} crop is affected by {disease.replace('_', ' ')}. "
        f"{disease_info.get('treatment', '')} "
        f"Maintain {condition} field conditions. "
        f"Recommended fertilizer: {fertilizer}. "
        f"Current market price: {price}."
    )

    return {
        "advisory": advisory,
        "fertilizer": fertilizer,
        "price": price,
        "disease_info": disease_info
    }

@app.post("/voice")
def generate_voice(data: AdvisoryRequest):
    """Generate vernacular (Telugu) voice note for advisory."""
    crop = data.crop.lower()
    disease = data.disease.replace("_", " ")
    condition = data.condition

    # Generate English advisory
    advisory_text = (
        f"Your {crop} crop is affected by {disease}. "
        f"Maintain {condition} field conditions and use proper fertilizer. "
        f"For healthy growth, consult the local agriculture office if needed."
    )

    # 🌐 Translate English → Telugu using Google Translate
    translator = Translator()
    try:
        translated_text = translator.translate(advisory_text, dest="te").text
    except Exception as e:
        translated_text = advisory_text  # fallback if translation fails

    # 🎙 Convert translated Telugu text to speech
    os.makedirs("voice_notes", exist_ok=True)
    file_path = "voice_notes/advice.mp3"
    tts = gTTS(translated_text, lang="te")
    tts.save(file_path)

    return {"audio_url": f"http://127.0.0.1:8000/{file_path}"}

@app.get("/market")
def get_market_prices():
    """Return current crop market prices (mock/demo)."""
    prices = {
        "rice": {"price": "₹1800 / quintal", "market": "Nizamabad"},
        "wheat": {"price": "₹2100 / quintal", "market": "Kurnool"},
        "cotton": {"price": "₹6400 / quintal", "market": "Warangal"},
        "tomato": {"price": "₹2400 / quintal", "market": "Madurai"},
        "groundnut": {"price": "₹5200 / quintal", "market": "Anantapur"}
    }
    return prices


def haversine_distance(lat1, lon1, lat2, lon2):
    # returns distance in kilometers between two lat/lon points
    R = 6371.0
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)

    a = math.sin(dphi/2)**2 + math.cos(phi1)*math.cos(phi2)*math.sin(dlambda/2)**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return R * c

MANDIS = [
    {"name": "Nizamabad Mandi", "lat": 18.6726, "lon": 78.0944, "prices": {"rice": 1800, "wheat": 2100}},
    {"name": "Kurnool Mandi", "lat": 15.8281, "lon": 78.0373, "prices": {"rice": 1750, "wheat": 2100}},
    {"name": "Warangal Mandi", "lat": 17.9789, "lon": 79.5941, "prices": {"cotton": 6400}},
    {"name": "Madurai Mandi", "lat": 9.9252, "lon": 78.1198, "prices": {"tomato": 2400}},
    {"name": "Anantapur Mandi", "lat": 14.6810, "lon": 77.6000, "prices": {"groundnut": 5200}},
]

@app.get("/market/nearby")
def get_nearby_markets(
    lat: float = Query(None),
    lon: float = Query(None),
    radius_km: float = Query(100.0),
    state: str = Query(None),
    city: str = Query(None),
    district: str = Query(None)
):
    """Return mandis within `radius_km` of the provided lat/lon, or by state/city/district if provided."""
    try:
        # If state/city/district are provided, use them to filter mandis
        if state and city and district:
            location_mandis = {
                "Maharashtra": {
                    "Pune": [
                        {"name": "Pune City Mandi", "lat": 18.5204, "lon": 73.8567, "prices": {"rice": 1850, "wheat": 2150}},
                        {"name": "Haveli Mandi", "lat": 18.6050, "lon": 73.8553, "prices": {"cotton": 6500}}
                    ],
                    "Mumbai": [
                        {"name": "Mumbai City Mandi", "lat": 19.0760, "lon": 72.8777, "prices": {"rice": 1900, "wheat": 2200}},
                        {"name": "Thane Mandi", "lat": 19.2183, "lon": 72.9781, "prices": {"tomato": 2500}}
                    ]
                },
                "Andhra Pradesh": {
                    "Vizag": [
                        {"name": "Vizag Mandi", "lat": 17.6868, "lon": 83.2185, "prices": {"rice": 1800, "groundnut": 5300}}
                    ],
                    "Vijayawada": [
                        {"name": "Vijayawada Central Mandi", "lat": 16.5062, "lon": 80.6480, "prices": {"wheat": 2100, "cotton": 6400}}
                    ],
                    "Guntur": [
                        {"name": "Guntur East Mandi", "lat": 16.3067, "lon": 80.4365, "prices": {"rice": 1750, "tomato": 2450}}
                    ]
                },
                "Telangana": {
                    "Hyderabad": [
                        {"name": "Hyderabad Mandi", "lat": 17.3850, "lon": 78.4867, "prices": {"rice": 1800, "wheat": 2100}},
                        {"name": "Secunderabad Mandi", "lat": 17.4399, "lon": 78.4983, "prices": {"cotton": 6400}}
                    ]
                }
            }
            mandis = location_mandis.get(state, {}).get(city, [])
            # Filter by district if needed (for demo, just return all for city)
            results = [
                {
                    "name": m["name"],
                    "lat": m["lat"],
                    "lon": m["lon"],
                    "distance_km": 0,
                    "prices": m["prices"]
                }
                for m in mandis
            ]
            if not results:
                # Suggest some popular locations
                suggestions = [
                    {"state": "Telangana", "city": "Hyderabad", "district": "Hyderabad"},
                    {"state": "Maharashtra", "city": "Pune", "district": "Pune City"},
                    {"state": "Andhra Pradesh", "city": "Vizag", "district": "Visakhapatnam"}
                ]
                return {
                    "count": 0,
                    "nearby": [],
                    "message": "No mandis available in this location.",
                    "suggestions": suggestions
                }
            return {"count": len(results), "nearby": results}

        # Otherwise, use lat/lon logic (default)
        if lat is not None and lon is not None:
            results = []
            for mandi in MANDIS:
                d = haversine_distance(lat, lon, mandi["lat"], mandi["lon"])
                if d <= radius_km:
                    results.append({
                        "name": mandi["name"],
                        "lat": mandi["lat"],
                        "lon": mandi["lon"],
                        "distance_km": round(d, 2),
                        "prices": mandi["prices"]
                    })
            results = sorted(results, key=lambda x: x["distance_km"])
            return {"count": len(results), "nearby": results}
        return {"count": 0, "nearby": []}
    except Exception as e:
        return {"error": str(e)}
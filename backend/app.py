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

app = FastAPI()
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
@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    image_bytes = await file.read()
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    img_t = transform(img).unsqueeze(0)

    with torch.no_grad():
        outputs = model(img_t)
        _, pred = outputs.max(1)

    result = class_names[pred.item()]
    return JSONResponse({"prediction": result})



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

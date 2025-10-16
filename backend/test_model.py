import torch
from torchvision import datasets, models, transforms
from PIL import Image
import os

# =========================
# Paths
# =========================
model_path = "models/plant_disease_model.pth"
data_dir = "data/plantvillage"

# =========================
# Transform (same as training)
# =========================
transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# =========================
# Load model
# =========================
dataset = datasets.ImageFolder(root=data_dir, transform=transform)
class_names = dataset.classes

model = models.mobilenet_v2(weights="IMAGENET1K_V1")
model.classifier[1] = torch.nn.Linear(model.last_channel, len(class_names))
model.load_state_dict(torch.load(model_path, map_location="cpu"))
model.eval()

# =========================
# Test with one image
# =========================
img_path = os.path.join(data_dir, class_names[0], os.listdir(os.path.join(data_dir, class_names[0]))[0])
print("Testing with image:", img_path)

img = Image.open(img_path).convert("RGB")
img_t = transform(img).unsqueeze(0)  # add batch dim

with torch.no_grad():
    outputs = model(img_t)
    _, pred = outputs.max(1)

print("Predicted class:", class_names[pred.item()])

# =========================
# Test with your own image
# =========================
custom_img_path = "my_leaf.jpg"   # <-- put your custom image in backend/ folder
if os.path.exists(custom_img_path):
    print("\nTesting with custom image:", custom_img_path)
    img = Image.open(custom_img_path).convert("RGB")
    img_t = transform(img).unsqueeze(0)

    with torch.no_grad():
        outputs = model(img_t)
        _, pred = outputs.max(1)

    print("Predicted class:", class_names[pred.item()])
else:
    print("\n⚠️ Place a test image named 'my_leaf.jpg' in backend/ folder to try custom testing.")


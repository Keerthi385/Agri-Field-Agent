import torch
from torch import nn, optim
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader
import os

# =========================
# Paths
# =========================
data_dir = "data/plantvillage"   # where your dataset is
model_dir = "models"             # where to save model
os.makedirs(model_dir, exist_ok=True)

# =========================
# Transforms
# =========================
transform = transforms.Compose([
    transforms.Resize((224,224)),   # resize all images to 224x224
    transforms.ToTensor(),          # convert to tensor
    transforms.Normalize([0.485, 0.456, 0.406],   # standard normalization
                         [0.229, 0.224, 0.225])
])

# =========================
# Load Dataset
# =========================
dataset = datasets.ImageFolder(root=data_dir, transform=transform)
class_names = dataset.classes
print("Classes:", class_names)

# split into train/validation (80-20 split)
# train_size = int(0.8 * len(dataset))
# val_size = len(dataset) - train_size
# train_ds, val_ds = torch.utils.data.random_split(dataset, [train_size, val_size])

# Use only 5000 images for quick training (ignore the rest for now)
train_ds, _ = torch.utils.data.random_split(dataset, [5000, len(dataset)-5000])

# You can still create a small validation set if you want
val_size = 1000
val_ds, _ = torch.utils.data.random_split(dataset, [val_size, len(dataset)-val_size])


train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=32)

# =========================
# Load Pretrained Model
# =========================
model = models.mobilenet_v2(pretrained=True)
# Replace classifier layer to match number of classes
model.classifier[1] = nn.Linear(model.last_channel, len(class_names))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# =========================
# Loss & Optimizer
# =========================
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# =========================
# Training Loop
# =========================
for epoch in range(1):   # just 2 epochs for hackathon demo
    model.train()
    running_loss = 0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
    print(f"Epoch {epoch+1}, Loss: {running_loss/len(train_loader):.4f}")

# =========================
# Save Model
# =========================
torch.save(model.state_dict(), os.path.join(model_dir, "plant_disease_model.pth"))
print("✅ Model saved at models/plant_disease_model.pth")

from pathlib import Path
from PIL import Image

dataset_path = Path("data/plantvillage")

# Print all classes
classes = [d.name for d in dataset_path.iterdir() if d.is_dir()]
print("Classes:", classes)

# Display one sample image from each class
for c in classes[:3]:  # just first 3 classes
    img_path = next((dataset_path / c).glob("*.jpg"))
    img = Image.open(img_path)
    img.show()

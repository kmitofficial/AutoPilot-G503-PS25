# evaluate.py
import os
import json
import re
import ast
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from PIL import Image
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np
from model import IntentVLM

# ====== Dataset Class ======
class DrivingDataset(Dataset):
    def __init__(self, data_path, transform=None):
        self.img_dir = os.path.join(data_path, "images")
        self.json_file = os.path.join(data_path, "training_data.jsonl")
        self.transform = transform
        self.img_files = []
        self.labels = []

        with open(self.json_file, "r") as f:
            for line in f:
                entry = json.loads(line.strip())
                self.img_files.append(entry['image'].split('/')[-1])
                match = re.search(r"Low-level commands: (\[.*\])", entry['suffix'])
                if match:
                    commands = ast.literal_eval(match.group(1))
                    self.labels.append(torch.tensor(commands, dtype=torch.float32))
                else:
                    self.labels.append(torch.zeros((6, 2)))

    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_files[idx])
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        label = self.labels[idx]
        return image, label


# ====== Paths ======
data_path = r"C:\Users\info\OneDrive\Desktop\3-1\iot\LightEMMA\drivelm\DriveLM\data\LightEmma"
model_path = r"C:\Users\info\OneDrive\Desktop\3-1\iot\LightEMMA\drivelm\DriveLM\drivelm\intentvlm_best.pth"  # <== NEW best model path


# ====== Config ======
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ====== Load Dataset ======
dataset = DrivingDataset(data_path, transform=transform)

# Split into train/val (same as in train.py)
val_split = 0.15
val_size = int(len(dataset) * val_split)
train_size = len(dataset) - val_size
_, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)

print(f"Validation samples: {len(val_dataset)}")

# ====== Load Model ======
input_dim = 3 * 224 * 224
hidden_dim = 256
output_dim = 12

model = IntentVLM(input_dim, hidden_dim, output_dim)
model.load_state_dict(torch.load(model_path, map_location=device))
model.to(device)
model.eval()

# ====== Evaluate ======
y_true, y_pred = [], []

with torch.no_grad():
    for images, labels in val_loader:
        images = images.to(device).view(images.size(0), -1)
        outputs = model(images).cpu()
        y_true.append(labels.view(1, -1))
        y_pred.append(outputs)

# Convert to numpy arrays
y_true = torch.cat(y_true).numpy()
y_pred = torch.cat(y_pred).numpy()

# ====== Metrics ======
mse = mean_squared_error(y_true, y_pred)
mae = mean_absolute_error(y_true, y_pred)
r2 = r2_score(y_true, y_pred)

# ====== Tolerance-based Accuracy ======
tolerance = 0.1
total = y_true.size
within_tolerance = (abs(y_true - y_pred) <= tolerance).sum()
tolerance_acc = within_tolerance / total * 100

# ====== Print Results ======
print("\n📊 Evaluation Results:")
print(f"Mean Squared Error (MSE): {mse:.4f}")
print(f"Mean Absolute Error (MAE): {mae:.4f}")
print(f"R² Score: {r2:.4f}")
print(f"Tolerance-based Accuracy (±{tolerance}): {tolerance_acc:.2f}%")

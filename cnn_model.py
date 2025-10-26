import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
import numpy as np
from torchvision import models, datasets, transforms
from torch.utils.data import DataLoader
from PIL import Image

from config import MODEL_PATH, DATASET_PATH, CLASS_NAMES

DATASET_PATH = r"C:\Users\HP\OneDrive\Desktop\abdul_fattakh_project\ECG_DATA"
MODEL_PATH = "models/ecg_cnn_model.pth"

class SimpleResNet18(nn.Module):

    def __init__(self):
        super().__init__()
        self.model = models.resnet18(weights=None)
        self.model.fc = nn.Linear(512, len(CLASS_NAMES))

    def forward(self, x):
        return self.model(x)

    def predict(self, image_tensor):
        self.eval()
        with torch.no_grad():
            output = self(image_tensor)
            prob = F.softmax(output, dim=1)
            conf, pred = torch.max(prob, 1)
        return pred.item(), conf.item()


def transform_image(img):
    img = cv2.resize(img, (224, 224))
    img = img.astype(np.float32) / 255.0
    if len(img.shape) == 2:
        img = np.stack([img]*3, axis=0)
    else:
        img = np.transpose(img, (2, 0, 1))
    mean = np.array([0.485, 0.456, 0.406]).reshape(3, 1, 1)
    std = np.array([0.229, 0.224, 0.225]).reshape(3, 1, 1)
    img = (img - mean) / std
    return torch.tensor(img).unsqueeze(0).float()


def preprocess_image(pil_img):
    rgb = np.array(pil_img.convert("RGB"))
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    resized = cv2.resize(gray, (512, 512))
    clahe = cv2.createCLAHE(clipLimit=2.0)
    contrast = clahe.apply(resized)
    _, binary = cv2.threshold(contrast, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    smoothed = cv2.medianBlur(binary, 3)
    return smoothed, rgb


def train_cnn_model():
    if os.path.exists(MODEL_PATH):
        print("Model already exists. Skipping training.")
        return

    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    if not os.path.exists(DATASET_PATH):
        raise FileNotFoundError(f"Dataset not found at {DATASET_PATH}")

    train_dataset = datasets.ImageFolder(DATASET_PATH, transform=train_transform)
    if len(train_dataset) == 0:
        raise ValueError("No images found in dataset.")

    loader = DataLoader(train_dataset, batch_size=100, shuffle=True)
    model = SimpleResNet18()
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    print("Training CNN...")
    for epoch in range(2):
        model.train()
        running_loss = 0.0
        for imgs, labels in loader:
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()
            loss = criterion(model(imgs), labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * imgs.size(0)
        print(f"Epoch {epoch+1} Loss: {running_loss/len(loader.dataset):.4f}")

    torch.save(model.state_dict(), MODEL_PATH)
    print("Model saved.")


def load_cnn():
    model = SimpleResNet18()
    if os.path.exists(MODEL_PATH):
        model.load_state_dict(torch.load(MODEL_PATH, map_location='cpu'))
    model.eval()
    return model
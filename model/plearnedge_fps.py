import os
import time
import torch
from PIL import Image
import torch.nn as nn
from torchinfo import summary
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
from sklearn.metrics import classification_report

# -----------------------------
# 1. LoRA-Injected Linear Layer
# -----------------------------
class LoRALinear(nn.Module):
    def __init__(self, in_features, out_features, r=4, alpha=16):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(out_features, in_features))
        self.bias = nn.Parameter(torch.zeros(out_features))
        self.lora_A = nn.Parameter(torch.randn(r, in_features))
        self.lora_B = nn.Parameter(torch.randn(out_features, r))
        self.scale = alpha / r

    def forward(self, x):
        lora_update = self.scale * torch.matmul(self.lora_B, self.lora_A)
        return F.linear(x, self.weight + lora_update, self.bias)

# -----------------------------
# 2. Lightweight CNN
# -----------------------------
class LightweightCNN(nn.Module):
    def __init__(self, embedding_dim=128):
        super().__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(3, 16, 3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(16, 32, 3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(64, 128, 3, stride=2, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1))
        )
        self.fc = LoRALinear(128, embedding_dim)

    def forward(self, x):
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)
        return F.normalize(self.fc(x), p=2, dim=1)

# -----------------------------
# 3. CLIP-like model
# -----------------------------
class FireClipModel(nn.Module):
    def __init__(self, embedding_dim=128, num_classes=2):
        super().__init__()
        self.image_encoder = LightweightCNN(embedding_dim)
        self.class_embeddings = nn.Parameter(torch.randn(num_classes, embedding_dim))

    def forward(self, x):
        image_emb = self.image_encoder(x)
        class_emb = F.normalize(self.class_embeddings, p=2, dim=1)
        return torch.matmul(image_emb, class_emb.T)

def calculate_fps_for_direct_image(model, device, image_path, class_names, repeat=100):
    # Transform (same as during training)
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    # Load and preprocess image
    image = Image.open(image_path).convert("RGB")
    input_tensor = transform(image).unsqueeze(0).to(device)

    # Move model to device and eval mode
    model.to(device)
    model.eval()

    # Warm-up
    with torch.no_grad():
        _ = model(input_tensor)

    # Time inference
    torch.cuda.synchronize()
    start_time = time.time()
    with torch.no_grad():
        for _ in range(repeat):
            output = model(input_tensor)
    torch.cuda.synchronize()
    end_time = time.time()

    # Compute FPS
    total_time = end_time - start_time
    fps = repeat / total_time

    # Final prediction
    pred_class = torch.argmax(output, dim=1).item()
    pred_label = class_names[pred_class]

    print(f"\n✅ Prediction: {pred_label}")
    print(f"✅ FPS for '{os.path.basename(image_path)}': {fps:.2f} frames/sec ({1000/fps:.2f} ms/frame)")

if __name__ == "__main__":
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load model
    model = FireClipModel()
    model.load_state_dict(torch.load("/home/work/spilab/fire_paper/cnn_clip_lora_kaggle.pth", map_location=device))

    # Define class names in the same order as during training
    class_names = ["fire", "smoke"]

    # Path to image for testing
    custom_image_path = "/home/work/spilab/fire_paper/sorted_dataset/fire/0000000060.jpg"  # <-- change to your image file path

    # Run inference + FPS
    calculate_fps_for_direct_image(model, device, custom_image_path, class_names, repeat=100)

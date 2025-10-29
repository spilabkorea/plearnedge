import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
from sklearn.metrics import classification_report
import os
from torchinfo import summary

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
        self.encoder = LightweightCNN(embedding_dim)
        self.head = nn.Linear(embedding_dim, num_classes)

    def forward(self, x):
        x = self.encoder(x)
        return self.head(x)

# -----------------------------
# 4. Data loading
# -----------------------------
def load_and_split_data(data_dir, batch_size=64, val_ratio=0.2):
    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor()
    ])
    full_dataset = datasets.ImageFolder(data_dir, transform=transform)
    total_len = len(full_dataset)
    val_len = int(val_ratio * total_len)
    train_len = total_len - val_len
    train_set, val_set = random_split(full_dataset, [train_len, val_len])

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
    return train_loader, val_loader, full_dataset.classes

# -----------------------------
# 5. Training
# -----------------------------
def train(model, loader, device):
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    criterion = nn.CrossEntropyLoss()
    model.train()
    for epoch in range(100):
        total_loss = 0
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            logits = model(images)
            loss = criterion(logits, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch + 1}: Loss = {total_loss:.4f}")

# -----------------------------
# 6. Evaluation
# -----------------------------
def evaluate(model, loader, device, class_names):
    model.eval()
    preds, trues = [], []
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            logits = model(images)
            predicted = torch.argmax(logits, dim=1).cpu().numpy()
            preds.extend(predicted)
            trues.extend(labels.numpy())

    report = classification_report(trues, preds, target_names=class_names)
    print(report)

    with open("classification_report_cnn_clip_lora_lch.txt", "w") as f:
        f.write(report)
    print("✅ Classification report saved.")

# -----------------------------
# 7. Main
# -----------------------------
if __name__ == "__main__":
    dataset_path = "/home/work/spilab/fire_paper/sorted_dataset"
    train_loader, val_loader, class_names = load_and_split_data(dataset_path)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = FireClipModel(num_classes=len(class_names))
    summary(model, input_size=(64, 3, 128, 128))
    train(model, train_loader, device)
    evaluate(model, val_loader, device, class_names)

    torch.save(model.state_dict(), "cnn_clip_lora_kaggle_lch.pth")
    print("✅ LoRA-injected model saved as cnn_clip_lora_kaggle.pth")

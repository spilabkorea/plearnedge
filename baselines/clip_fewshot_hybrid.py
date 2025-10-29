import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from sklearn.metrics import classification_report
from collections import defaultdict
from torchinfo import summary
import os

# -----------------------------
# 1. LoRA-Injected Conv2d Layer
# -----------------------------
class LoRAConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, r=4, alpha=16):
        super().__init__()
        self.base_conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.lora_A = nn.Conv2d(in_channels, r, kernel_size=1, stride=stride, padding=0, bias=False)
        self.lora_B = nn.Conv2d(r, out_channels, kernel_size=1, stride=1, padding=0, bias=False)
        self.scale = alpha / r

    def forward(self, x):
        base_out = self.base_conv(x)
        lora_out = self.lora_B(self.lora_A(x))
        return base_out + self.scale * lora_out

# -----------------------------
# 2. Lightweight CNN with LoRA
# -----------------------------
class LightweightCNNWithLoRA(nn.Module):
    def __init__(self, embedding_dim=128):
        super().__init__()
        self.conv_layers = nn.Sequential(
            LoRAConv2d(3, 16, 3, stride=2, padding=1), nn.ReLU(),
            LoRAConv2d(16, 32, 3, stride=2, padding=1), nn.ReLU(),
            LoRAConv2d(32, 64, 3, stride=2, padding=1), nn.ReLU(),
            LoRAConv2d(64, 128, 3, stride=2, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1))
        )
        self.fc = nn.Linear(128, embedding_dim)

    def forward(self, x):
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)
        return F.normalize(self.fc(x), p=2, dim=1)

# -----------------------------
# 3. SoftPrompt Module
# -----------------------------
class SoftPrompt(nn.Module):
    def __init__(self, prompt_len=5, embedding_dim=128):
        super().__init__()
        self.prompt_embeddings = nn.Parameter(torch.randn(prompt_len, embedding_dim))

    def forward(self, class_embeddings):
        B = class_embeddings.size(0)
        prompt = self.prompt_embeddings.unsqueeze(0).expand(B, -1, -1)
        class_embeddings = class_embeddings.unsqueeze(1)
        enriched = torch.cat([prompt, class_embeddings], dim=1)
        return enriched.mean(dim=1)

# -----------------------------
# 4. Hybrid FireClip Model
# -----------------------------
class FireClipHybrid(nn.Module):
    def __init__(self, embedding_dim=128, num_classes=2, prompts_per_class=1, prompt_len=5):
        super().__init__()
        self.image_encoder = LightweightCNNWithLoRA(embedding_dim)
        self.prompts_per_class = prompts_per_class
        self.class_embeddings = nn.Parameter(torch.randn(num_classes * prompts_per_class, embedding_dim))
        self.soft_prompt = SoftPrompt(prompt_len, embedding_dim)

    def forward(self, x):
        image_emb = self.image_encoder(x)
        text_emb = F.normalize(self.class_embeddings, p=2, dim=1)
        if self.prompts_per_class > 1:
            text_emb = text_emb.view(-1, self.prompts_per_class, text_emb.size(1)).mean(dim=1)
        text_emb = self.soft_prompt(text_emb)
        return torch.matmul(image_emb, text_emb.T)

# -----------------------------
# 5. Few-Shot Data Loader
# -----------------------------
def load_few_shot_data(data_dir, shots_per_class=5, batch_size=32):
    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor()
    ])
    full_dataset = datasets.ImageFolder(data_dir, transform=transform)
    class_indices = defaultdict(list)

    for idx, (_, label) in enumerate(full_dataset.samples):
        class_indices[label].append(idx)

    few_shot_indices = []
    for label, indices in class_indices.items():
        few_shot_indices.extend(indices[:shots_per_class])

    few_shot_subset = Subset(full_dataset, few_shot_indices)
    val_indices = list(set(range(len(full_dataset))) - set(few_shot_indices))
    val_subset = Subset(full_dataset, val_indices)

    train_loader = DataLoader(few_shot_subset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader, full_dataset.classes

# -----------------------------
# 6. Freeze all except LoRA + SoftPrompt
# -----------------------------
def freeze_except_lora_and_softprompt(model):
    for name, param in model.named_parameters():
        if "lora_" in name or "soft_prompt" in name:
            param.requires_grad = True
        else:
            param.requires_grad = False

# -----------------------------
# 7. Training Loop
# -----------------------------
def train(model, loader, device):
    model.to(device)
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=0.0005)
    criterion = nn.CrossEntropyLoss()
    model.train()
    for epoch in range(30):
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
# 8. Evaluation
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
    with open("classification_report_hybrid.txt", "w") as f:
        f.write(report)
    print("✅ Evaluation complete.")

# -----------------------------
# 9. Main
# -----------------------------
if __name__ == "__main__":
    dataset_path = "/home/work/spilab/fire_paper/sorted_dataset"
    train_loader, val_loader, class_names = load_few_shot_data(dataset_path, shots_per_class=5)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = FireClipHybrid(num_classes=len(class_names), prompts_per_class=3, prompt_len=5)

    freeze_except_lora_and_softprompt(model)  # Hybrid tuning

    summary(model, input_size=(64, 3, 128, 128))
    train(model, train_loader, device)
    evaluate(model, val_loader, device, class_names)

    torch.save(model.state_dict(), "fire_clip_hybrid_fewshot.pth")
    print("✅ Saved Few-Shot Hybrid model (SoftPrompt + LoRA).")

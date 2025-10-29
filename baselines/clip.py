import os
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import CLIPProcessor, CLIPModel
from PIL import Image
from tqdm import tqdm
from sklearn.metrics import classification_report

# === Config ===
DATA_DIR = "/home/work/spilab/fire_paper/3d_scalogram_eo_ir"
BATCH_SIZE = 64
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# === Prompt Templates ===
class_prompts = {
    0: "fire in image",
    1: "nonfire in image"
}
prompt_texts = list(class_prompts.values())

# === Dataset (Image only, no training label needed) ===
class FireDataset(Dataset):
    def __init__(self, root_dir):
        self.samples = []
        self.labels = {"fire_scene": 0, "smoke_area": 1}
        for label_name in ["fire_scene", "smoke_area"]:
            folder_path = os.path.join(root_dir, label_name)
            for file in os.listdir(folder_path):
                if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                    self.samples.append((os.path.join(folder_path, file), self.labels[label_name]))
        self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        image_path, label = self.samples[idx]
        image = Image.open(image_path).convert("RGB")
        return image, label

# === Custom Collate Function for PIL Images ===
def collate_fn(batch):
    images, labels = zip(*batch)  # unzip list of (image, label)
    return list(images), torch.tensor(labels)

# === Load Dataset and Dataloader ===
dataset = FireDataset(DATA_DIR)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)

# === Load CLIP Model and Processor ===
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(DEVICE)
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
model.eval()

# === Encode Text Prompts Once ===
text_inputs = processor(text=prompt_texts, return_tensors="pt", padding=True).to(DEVICE)
with torch.no_grad():
    text_features = model.get_text_features(**text_inputs)
    text_features = text_features / text_features.norm(dim=-1, keepdim=True)

# === Inference Loop ===
all_preds = []
all_labels = []

with torch.no_grad():
    for images, labels in tqdm(dataloader, desc="Zero-shot inference"):
        inputs = processor(images=images, return_tensors="pt", padding=True).to(DEVICE)
        image_features = model.get_image_features(pixel_values=inputs["pixel_values"])
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)

        # Cosine similarity between image and text embeddings
        logits = image_features @ text_features.T
        preds = logits.argmax(dim=1).cpu().tolist()

        all_preds.extend(preds)
        all_labels.extend(labels.tolist())

# === Classification Report ===
print("\n📊 Zero-Shot Classification Report:")
print(classification_report(all_labels, all_preds, target_names=["fire", "non-fire"]))

import torch
import open_clip
from PIL import Image
from torchvision import datasets
from torchvision.datasets.folder import default_loader
from torchvision import transforms
import os
from tqdm import tqdm
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ✅ Load model and tokenizer
model, _, preprocess = open_clip.create_model_and_transforms(
    model_name="coca_ViT-B-32",
    pretrained="laion2B-s13B-b90k"
)
tokenizer = open_clip.get_tokenizer("coca_ViT-B-32")
model = model.to(device).eval()

# ✅ Define prompts and class names
class_names = ["fire", "smoke"]
candidate_prompts = [f"a photo of {label}" for label in class_names]
text_inputs = tokenizer(candidate_prompts).to(device)

# ✅ Encode prompts once
with torch.no_grad():
    text_features = model.encode_text(text_inputs)
    text_features /= text_features.norm(dim=-1, keepdim=True)

# ✅ Dataset root
root_dir = "/home/work/spilab/fire_paper/sorted_dataset"  # has subfolders 'fire/', 'smoke/'

# ✅ Gather all image paths and labels
image_paths = []
labels = []

for label_idx, class_name in enumerate(class_names):
    class_dir = os.path.join(root_dir, class_name)
    for fname in os.listdir(class_dir):
        if fname.lower().endswith(('.jpg', '.png', '.jpeg', '.bmp')):
            image_paths.append(os.path.join(class_dir, fname))
            labels.append(label_idx)

# ✅ Predict and collect
preds = []
for img_path in tqdm(image_paths, desc="Predicting"):
    image = Image.open(img_path).convert("RGB")
    image_tensor = preprocess(image).unsqueeze(0).to(device)

    with torch.no_grad():
        image_features = model.encode_image(image_tensor)
        image_features /= image_features.norm(dim=-1, keepdim=True)

        logits = (100.0 * image_features @ text_features.T)
        probs = logits.softmax(dim=-1).squeeze()
        pred_class = probs.argmax().item()

    preds.append(pred_class)

# ✅ Evaluate
accuracy = accuracy_score(labels, preds)
print(f"\n✅ Accuracy: {accuracy * 100:.2f}%\n")

# ✅ Optional: Detailed metrics
print("Classification Report:")
print(classification_report(labels, preds, target_names=class_names))

print("Confusion Matrix:")
print(confusion_matrix(labels, preds))

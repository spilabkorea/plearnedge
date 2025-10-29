import os
import torch
from PIL import Image
from tqdm import tqdm
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from transformers import AutoProcessor, AutoModelForCausalLM

# ✅ Load GIT model and processor
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
processor = AutoProcessor.from_pretrained("microsoft/git-base")
model = AutoModelForCausalLM.from_pretrained("microsoft/git-base").to(device)

# ✅ Define dataset path and class names
root_dir = "/home/work/spilab/fire_paper/sorted_dataset"
class_names = ["fire", "smoke"]

# ✅ Collect image paths and true labels
image_paths = []
true_labels = []

for idx, cls in enumerate(class_names):
    class_dir = os.path.join(root_dir, cls)
    for fname in os.listdir(class_dir):
        if fname.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
            image_paths.append(os.path.join(class_dir, fname))
            true_labels.append(idx)

# ✅ Captioning and classification
predicted_labels = []
captions = []

for img_path in tqdm(image_paths, desc="Captioning"):
    image = Image.open(img_path).convert("RGB")
    inputs = processor(images=image, return_tensors="pt").to(device)

    with torch.no_grad():
        generated_ids = model.generate(
            pixel_values=inputs["pixel_values"],
            max_new_tokens=20,
            num_beams=5,
            early_stopping=True
        )

    caption = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
    captions.append(caption)

    # Simple keyword-based classification from caption
    caption_lower = caption.lower()
    if "fire" in caption_lower:
        predicted_labels.append(0)
    elif "smoke" in caption_lower:
        predicted_labels.append(1)
    else:
        predicted_labels.append(-1)  # unknown / reject

# ✅ Filter out unknown predictions
valid_indices = [i for i, pred in enumerate(predicted_labels) if pred != -1]
filtered_preds = [predicted_labels[i] for i in valid_indices]
filtered_labels = [true_labels[i] for i in valid_indices]
filtered_captions = [captions[i] for i in valid_indices]

# ✅ Evaluation
accuracy = accuracy_score(filtered_labels, filtered_preds)
print(f"\n✅ Caption-based Classification Accuracy: {accuracy * 100:.2f}% ({len(filtered_preds)}/{len(true_labels)} images used)\n")
print("Classification Report:")
print(classification_report(filtered_labels, filtered_preds, target_names=class_names))
print("Confusion Matrix:")
print(confusion_matrix(filtered_labels, filtered_preds))

# ✅ Optional: Save captions and predictions
with open("git_fire_smoke_captions.txt", "w") as f:
    for path, label, pred, cap in zip(image_paths, true_labels, predicted_labels, captions):
        f.write(f"{os.path.basename(path)}\tGT:{class_names[label]}\tPred:{class_names[pred] if pred!=-1 else 'Unknown'}\tCaption:{cap}\n")

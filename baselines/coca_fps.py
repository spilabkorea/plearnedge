import time
import torch
import open_clip
from PIL import Image

# ✅ Setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ✅ Load model and tokenizer
model, _, preprocess = open_clip.create_model_and_transforms(
    model_name="coca_ViT-B-32",
    pretrained="laion2B-s13B-b90k"
)
tokenizer = open_clip.get_tokenizer("coca_ViT-B-32")
model = model.to(device).eval()

# ✅ Define prompts and image path
class_names = ["fire", "smoke"]
prompts = [f"a photo of {label}" for label in class_names]
image_path = "/home/work/spilab/fire_paper/sorted_dataset/fire/0000000060.jpg"  # 🔁 Replace with your image

# ✅ Preprocess image and text
image = Image.open(image_path).convert("RGB")
image_tensor = preprocess(image).unsqueeze(0).to(device)
text_inputs = tokenizer(prompts).to(device)

# ✅ Encode text once
with torch.no_grad():
    text_features = model.encode_text(text_inputs)
    text_features = text_features / text_features.norm(dim=-1, keepdim=True)

# ✅ Timed inference
torch.cuda.synchronize() if device.type == "cuda" else None
start_time = time.time()

with torch.no_grad():
    image_features = model.encode_image(image_tensor)
    image_features = image_features / image_features.norm(dim=-1, keepdim=True)
    logits = (100.0 * image_features @ text_features.T)
    probs = logits.softmax(dim=-1).squeeze()

torch.cuda.synchronize() if device.type == "cuda" else None
end_time = time.time()

# ✅ Results
pred_idx = torch.argmax(probs).item()
pred_label = class_names[pred_idx]
inference_time = end_time - start_time
fps = 1.0 / inference_time

# ✅ Output
print(f"\n🖼️ Image: {image_path}")
print(f"📝 Predicted Label: {pred_label}")
print(f"📊 Probabilities: {dict(zip(class_names, probs.tolist()))}")
print(f"⏱️ Inference Time: {inference_time * 1000:.2f} ms")
print(f"🚀 FPS: {fps:.2f} frames/sec")

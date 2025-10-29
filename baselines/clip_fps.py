import os
import time
import torch
from PIL import Image
from torchvision import transforms
from transformers import CLIPProcessor, CLIPModel

# === CLIP Inference + FPS Measurement ===
def calculate_fps_clip(model, processor, device, image_path, prompt_texts, repeat=100):
    image = Image.open(image_path).convert("RGB")

    # Move model to device before any inference
    model.to(device).eval()

    # Process and move text inputs to device
    text_inputs = processor(text=prompt_texts, return_tensors="pt", padding=True)
    text_inputs = {k: v.to(device) for k, v in text_inputs.items()}

    with torch.inference_mode():
        text_features = model.get_text_features(**text_inputs)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

    # Image preprocessing
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.4815, 0.4578, 0.4082], [0.2686, 0.2613, 0.2758])
    ])
    input_tensor = transform(image).unsqueeze(0).to(device)

    # Warm-up
    with torch.inference_mode():
        _ = model.get_image_features(pixel_values=input_tensor)

    torch.cuda.synchronize() if device.type == "cuda" else None
    start_time = time.time()

    with torch.inference_mode():
        for _ in range(repeat):
            image_features = model.get_image_features(pixel_values=input_tensor)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            logits = image_features @ text_features.T

    torch.cuda.synchronize() if device.type == "cuda" else None
    end_time = time.time()

    total_time = end_time - start_time
    fps = repeat / total_time

    pred_class = torch.argmax(logits, dim=1).item()
    pred_label = prompt_texts[pred_class]

    print(f"\n✅ Prediction: {pred_label}")
    print(f"✅ FPS: {fps:.2f} frames/sec ({1000/fps:.2f} ms/frame)")

# === MAIN ===
if __name__ == "__main__":
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    prompt_texts = ["a picture of fire", "a picture of smoke"]
    image_path = "/home/work/spilab/fire_paper/sorted_dataset/fire/0000000060.jpg"  # Replace with your image path

    calculate_fps_clip(model, processor, device, image_path, prompt_texts, repeat=100)

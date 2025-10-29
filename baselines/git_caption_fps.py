import time
import torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForCausalLM

def calculate_fps_git(model, processor, device, image_path, repeat=100):
    # Load and preprocess image
    image = Image.open(image_path).convert("RGB")
    inputs = processor(images=image, return_tensors="pt").to(device)

    # Warm-up (especially important for GPU)
    with torch.no_grad():
        _ = model.generate(pixel_values=inputs["pixel_values"], max_new_tokens=20)

    # Timed inference
    torch.cuda.synchronize() if device.type == "cuda" else None
    start_time = time.time()
    with torch.no_grad():
        for _ in range(repeat):
            generated_ids = model.generate(
                pixel_values=inputs["pixel_values"],
                max_new_tokens=20,
                num_beams=5,
                early_stopping=True
            )
    torch.cuda.synchronize() if device.type == "cuda" else None
    end_time = time.time()

    # Decode caption from last iteration
    caption = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

    # FPS and latency
    total_time = end_time - start_time
    fps = repeat / total_time
    latency = 1000 / fps

    print(f"📝Generated Caption: {caption}")
    print(f"🚀Inference FPS: {fps:.2f} frames/sec")
    print(f"⏱️Average Latency: {latency:.2f} ms/frame")

# === MAIN ===
if __name__ == "__main__":
    model = AutoModelForCausalLM.from_pretrained("microsoft/git-base")
    processor = AutoProcessor.from_pretrained("microsoft/git-base")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device).eval()

    image_path = "/home/work/spilab/fire_paper/sorted_dataset/fire/0000000060.jpg" 
    calculate_fps_git(model, processor, device, image_path, repeat=100)

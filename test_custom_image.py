import sys
import os
from PIL import Image
import torch
from torchvision import transforms

# --- CRITICAL FIX FOR MAC & CPU ---
# This tells PyTorch to use the same engine you trained with.
# Without this, it doesn't know how to read the compressed layers.
torch.backends.quantized.engine = 'qnnpack'

MODEL_PATH = "Team_30_She+2/model/model.pth"

def debug_predict(image_path):
    print(f"🔍 Debugging file: {image_path}")
    
    if not os.path.exists(image_path):
        print("❌ ERROR: File not found.")
        return

    try:
        img = Image.open(image_path).convert('RGB')
        print(f"✅ Image loaded! Size: {img.size}")
    except Exception as e:
        print(f"❌ ERROR opening image: {e}")
        return

    try:
        # Load model structure
        device = torch.device("cpu")
        # weights_only=False is required for full model loading
        model = torch.load(MODEL_PATH, map_location=device, weights_only=False)
        model.eval()
        
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        
        input_tensor = transform(img).unsqueeze(0)
        
        with torch.no_grad():
            outputs = model(input_tensor)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            confidence, predicted_idx = torch.max(probabilities, 1)
            
            raw_label = predicted_idx.item()
            
            # Label Swap Logic
            final_label = 1 if raw_label == 0 else 0
            gender = "FEMALE 👩" if final_label == 1 else "MALE 👨"
            
            print("-" * 30)
            print(f"🎯 RESULT: {gender}")
            print(f"📊 CONFIDENCE: {confidence.item():.2%}")
            print("-" * 30)

    except Exception as e:
        print(f"❌ MODEL CRASH: {e}")

if __name__ == "__main__":
    path = input("Drag image here: ").strip()
    path = path.replace("'", "").replace('"', "").replace("\\ ", " ").strip()
    debug_predict(path)
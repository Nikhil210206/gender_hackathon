import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import os
import warnings

# 1. Suppress Warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# 2. CRITICAL: Set the Quantization Engine
# This ensures the model runs on Linux, Windows, AND Mac without crashing.
torch.backends.quantized.engine = 'qnnpack'

class GenderPredictor:
    def __init__(self):
        self.model_path = os.path.join(os.path.dirname(__file__), 'model/model.pth')
        self.device = torch.device("cpu")
        
        try:
            # Load Model
            self.model = torch.load(self.model_path, map_location=self.device, weights_only=False)
            self.model.eval()
        except Exception as e:
            print(f"CRITICAL ERROR loading model: {e}")
            self.model = None
            
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

    def predict_image(self, image_path):
        if self.model is None:
            return 0, 0.0

        try:
            image = Image.open(image_path).convert('RGB')
            input_tensor = self.transform(image).unsqueeze(0)
            
            with torch.no_grad():
                outputs = self.model(input_tensor)
                probabilities = torch.nn.functional.softmax(outputs, dim=1)
                confidence, predicted_idx = torch.max(probabilities, 1)
                
                raw_label = predicted_idx.item()
                conf_score = confidence.item()

                # LOGIC SWAP (0->1, 1->0)
                final_label = 1 if raw_label == 0 else 0
                
                return final_label, conf_score
        except Exception as e:
            return 0, 0.0

predictor = GenderPredictor()

def predict(image_path):
    """
    Returns:
    label: int (0 = Male, 1 = Female)
    confidence: float (0–1)
    """
    return predictor.predict_image(image_path)
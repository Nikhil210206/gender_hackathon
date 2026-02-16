# Team She+2 (Team_30) - Gender Classification

## Model Overview
We utilize a quantized **MobileNetV3-Large** architecture. This model was chosen to maximize the trade-off between accuracy and inference speed/size, specifically optimized for CPU-only environments.

## Setup
1. Install requirements: `pip install -r requirements.txt`
2. Run inference using `inference.py`.

## Technical Details
- **Architecture**: MobileNetV3-Large (Pretrained on ImageNet)
- **Optimization**: Dynamic Quantization (Int8)
- **Input Size**: 224x224
- **Labels**: 0 -> Male, 1 -> Female
import sys
import os

# Add the team folder to python path so we can import inference
sys.path.append("Team_30_She+2")

try:
    from inference import predict
    print("✅ inference.py imported successfully!")
except ImportError as e:
    print(f"❌ Failed to import inference.py: {e}")
    exit()

# Test on a real image (Change this path to a real image from your dataset)
test_image = "dataset/val/male/063477.jpg.jpg"  # Pick any image filename you see in your folder

if os.path.exists(test_image):
    label, conf = predict(test_image)
    print(f"Testing on {test_image}")
    print(f"Prediction: {label} (Should be 0 for Male, 1 for Female)")
    print(f"Confidence: {conf:.4f}")
    
    if label == 0: 
        print("🎉 SUCCESS! It predicted Male correctly!")
    else:
        print("🤔 It predicted Female. (If image is Male, check label swap logic)")
else:
    print("⚠️ Could not find test image. Please update the 'test_image' path in this script to verify.")
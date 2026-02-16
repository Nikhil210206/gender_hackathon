import os
import zipfile
import shutil
import random
from pathlib import Path

# --- Configuration ---
ZIP_FILE_NAME = "archive.zip"  # Rename your downloaded file to this, or change this line
DATA_DIR = "dataset"
RAW_DIR = "raw_data_temp"

def setup_dataset():
    # 1. Check if zip exists
    if not os.path.exists(ZIP_FILE_NAME):
        print(f"❌ Error: {ZIP_FILE_NAME} not found!")
        print("Please download the dataset from Kaggle and place it in this folder.")
        return

    print(f"📂 Unzipping {ZIP_FILE_NAME}...")
    with zipfile.ZipFile(ZIP_FILE_NAME, 'r') as zip_ref:
        zip_ref.extractall(RAW_DIR)

    # 2. Define source paths (Kaggle datasets usually have 'Training' and 'Validation' or just 'male'/'female')
    # We need to find where the images are inside the extracted folder.
    print("🔍 Searching for image folders...")
    
    male_src = []
    female_src = []

    for root, dirs, files in os.walk(RAW_DIR):
        for file in files:
            if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                path = os.path.join(root, file)
                if "male" in root.lower() and "female" not in root.lower():
                    male_src.append(path)
                elif "female" in root.lower():
                    female_src.append(path)

    print(f"✅ Found {len(male_src)} Male images and {len(female_src)} Female images.")

    # 3. Create Clean Destination Structure
    for split in ['train', 'val']:
        for cls in ['male', 'female']:
            os.makedirs(os.path.join(DATA_DIR, split, cls), exist_ok=True)

    # 4. Shuffle and Split (80% Train, 20% Val)
    random.shuffle(male_src)
    random.shuffle(female_src)

    def move_files(files, label):
        split_idx = int(len(files) * 0.8)  # 80/20 split
        train_files = files[:split_idx]
        val_files = files[split_idx:]

        print(f"   Moving {len(train_files)} to Train and {len(val_files)} to Val for '{label}'...")

        for f in train_files:
            shutil.copy(f, os.path.join(DATA_DIR, 'train', label, os.path.basename(f)))
        for f in val_files:
            shutil.copy(f, os.path.join(DATA_DIR, 'val', label, os.path.basename(f)))

    print("🚀 Organizing Dataset...")
    move_files(male_src, 'male')
    move_files(female_src, 'female')

    # 5. Cleanup
    print("🧹 Cleaning up temp files...")
    shutil.rmtree(RAW_DIR)
    print("✨ Data setup complete! You can now run train.py.")

if __name__ == "__main__":
    setup_dataset()
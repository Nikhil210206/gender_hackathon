import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, models, transforms
import os
import copy
from torch.quantization import quantize_dynamic
from tqdm import tqdm

def train_model():
    # --- CONFIGURATION ---
    DATA_DIR = 'dataset'
    os.makedirs('Team_30_She+2/model', exist_ok=True)
    MODEL_SAVE_PATH = 'Team_30_She+2/model/model.pth' 
    NUM_EPOCHS = 5
    BATCH_SIZE = 32
    
    # --- DETECT MAC GPU ---
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print("🚀 MAC GPU ACTIVATED (MPS) - Training will be fast!")
    else:
        device = torch.device("cpu")
        print("⚠️ MPS not found. Using CPU.")

    # --- DATA SETUP ---
    data_transforms = {
        'train': transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'val': transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }

    image_datasets = {x: datasets.ImageFolder(os.path.join(DATA_DIR, x), data_transforms[x]) for x in ['train', 'val']}
    dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=BATCH_SIZE, shuffle=True, num_workers=2) for x in ['train', 'val']}
    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
    class_names = image_datasets['train'].classes
    print(f"Classes: {class_names}")

    # --- MODEL SETUP ---
    print("Loading MobileNetV3...")
    model = models.mobilenet_v3_large(weights=models.MobileNet_V3_Large_Weights.IMAGENET1K_V1)
    
    for param in model.parameters():
        param.requires_grad = False
    for param in model.classifier.parameters():
        param.requires_grad = True
    
    num_ftrs = model.classifier[3].in_features
    model.classifier[3] = nn.Linear(num_ftrs, 2)
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr=0.001)

    # --- TRAINING LOOP ---
    best_acc = 0.0
    best_model_wts = copy.deepcopy(model.state_dict())

    for epoch in range(NUM_EPOCHS):
        print(f'\nEpoch {epoch+1}/{NUM_EPOCHS}')
        print('-' * 10)

        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_corrects = 0
            loop = tqdm(dataloaders[phase], desc=f"{phase} Phase", leave=False)

            for inputs, labels in loop:
                inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
                loop.set_postfix(loss=loss.item())

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.float() / dataset_sizes[phase]

            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

    print(f'\nBest Val Acc: {best_acc:4f}')

    # --- SAVING & QUANTIZATION ---
    print("Preparing model for submission (CPU Conversion)...")
    model.load_state_dict(best_model_wts)
    model.to('cpu')
    model.eval()
    
    # *** THE FIX: Explicitly set the engine for Mac/ARM ***
    torch.backends.quantized.engine = 'qnnpack'
    
    print("Quantizing (Shrinking size)...")
    quantized_model = quantize_dynamic(
        model, {nn.Linear}, dtype=torch.qint8
    )
    
    torch.save(quantized_model, MODEL_SAVE_PATH)
    print(f"✅ SUCCESS! Model saved to {MODEL_SAVE_PATH}")

if __name__ == '__main__':
    train_model()
import os
import sys
import random
import json
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix

def set_seed(seed=42):
    """
    Set random seeds for reproducibility.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

class OwnerDataset(Dataset):
    """
    Custom Dataset to load owner/other images from paths list.
    """
    def __init__(self, file_paths, labels, transform=None):
        self.file_paths = file_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        path = self.file_paths[idx]
        # Open image and convert to RGB
        img = Image.open(path).convert("RGB")
        label = self.labels[idx]
        
        if self.transform:
            img = self.transform(img)
            
        return img, label

def check_dataset_and_get_paths(owner_dir, others_dir):
    """
    Check if dataset folders exist and contain images. Return lists of paths and labels.
    """
    if not os.path.exists(owner_dir) or not os.path.exists(others_dir):
        print("\n[ERROR] Dataset directories do not exist.")
        print("Please structure your dataset directories under:")
        print(f"  - {owner_dir}")
        print(f"  - {others_dir}")
        sys.exit(1)

    valid_extensions = ('.png', '.jpg', '.jpeg')
    
    owner_paths = [
        os.path.join(owner_dir, f) for f in os.listdir(owner_dir)
        if f.lower().endswith(valid_extensions)
    ]
    others_paths = [
        os.path.join(others_dir, f) for f in os.listdir(others_dir)
        if f.lower().endswith(valid_extensions)
    ]

    print(f"\nScanning dataset folders...")
    print(f"  - Owner images found: {len(owner_paths)}")
    print(f"  - Other/non-owner images found: {len(others_paths)}")

    if len(owner_paths) == 0 or len(others_paths) == 0:
        print("\n[ERROR] Empty dataset detected.")
        print("You must collect and store image crops for both classes before training.")
        print(f"Place owner crops in: {owner_dir}")
        print(f"Place other crops in: {others_dir}")
        print("Exiting training pipeline gracefully...")
        sys.exit(0)

    # Class 0: OTHER, Class 1: OWNER
    file_paths = others_paths + owner_paths
    labels = [0] * len(others_paths) + [1] * len(owner_paths)
    
    return file_paths, labels

def main():
    # Set random seed for reproducibility
    set_seed(42)
    
    # Paths setup
    owner_dir = os.path.join("data", "owner")
    others_dir = os.path.join("data", "others")
    results_dir = os.path.join("experiments", "results")
    model_save_path = os.path.join("models", "owner_classifier.pth")
    
    # Ensure results and models folders exist
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)

    # Scans directories and returns paths (exits gracefully if empty)
    file_paths, labels = check_dataset_and_get_paths(owner_dir, others_dir)
    
    # Split into train and validation sets (80% train, 20% validation, stratified)
    train_paths, val_paths, train_labels, val_labels = train_test_split(
        file_paths, labels, test_size=0.2, random_state=42, stratify=labels
    )
    
    print(f"\nDataset split complete:")
    print(f"  - Training samples: {len(train_paths)}")
    print(f"  - Validation samples: {len(val_paths)}")
    
    # Image preprocessing and data augmentation
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Datasets and Loaders
    train_dataset = OwnerDataset(train_paths, train_labels, transform=train_transform)
    val_dataset = OwnerDataset(val_paths, val_labels, transform=val_transform)

    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False, num_workers=0)

    # Set up Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nTraining on device: {device}")

    # Load pretrained MobileNetV3 Small
    print("Loading pretrained MobileNetV3-Small backbone...")
    weights = models.MobileNet_V3_Small_Weights.DEFAULT
    model = models.mobilenet_v3_small(weights=weights)

    # Freeze pretrained backbone parameters
    for param in model.parameters():
        param.requires_grad = False

    # Replace the final linear classification layer in the head
    # MobileNetV3 small classifier has structure:
    # classifier[0]: Linear, classifier[1]: Hardswish, classifier[2]: Dropout, classifier[3]: Linear (1000 outputs)
    num_features = model.classifier[3].in_features
    model.classifier[3] = nn.Linear(num_features, 2)  # Binary classification: OTHER vs OWNER

    # Unfreeze the classifier head so it learns
    for param in model.classifier.parameters():
        param.requires_grad = True

    model.to(device)

    # Define optimizer and loss function
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.classifier.parameters(), lr=0.001)

    # Tracking metrics
    num_epochs = 10
    history = {
        "train_loss": [],
        "val_loss": [],
        "train_acc": [],
        "val_acc": []
    }

    print("\nStarting Training Pipeline...")
    for epoch in range(num_epochs):
        # --- Training Phase ---
        model.train()
        running_loss = 0.0
        correct_train = 0
        total_train = 0
        
        for images, labels_batch in train_loader:
            images = images.to(device)
            labels_batch = labels_batch.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels_batch)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs, 1)
            total_train += labels_batch.size(0)
            correct_train += (predicted == labels_batch).sum().item()

        epoch_train_loss = running_loss / total_train
        epoch_train_acc = correct_train / total_train

        # --- Validation Phase ---
        model.eval()
        running_val_loss = 0.0
        correct_val = 0
        total_val = 0
        
        with torch.no_grad():
            for images, labels_batch in val_loader:
                images = images.to(device)
                labels_batch = labels_batch.to(device)
                
                outputs = model(images)
                loss = criterion(outputs, labels_batch)
                
                running_val_loss += loss.item() * images.size(0)
                _, predicted = torch.max(outputs, 1)
                total_val += labels_batch.size(0)
                correct_val += (predicted == labels_batch).sum().item()

        epoch_val_loss = running_val_loss / total_val
        epoch_val_acc = correct_val / total_val

        # Record metrics
        history["train_loss"].append(epoch_train_loss)
        history["val_loss"].append(epoch_val_loss)
        history["train_acc"].append(epoch_train_acc)
        history["val_acc"].append(epoch_val_acc)

        print(f"Epoch [{epoch+1}/{num_epochs}] -> "
              f"Train Loss: {epoch_train_loss:.4f} | Train Acc: {epoch_train_acc:.2%} | "
              f"Val Loss: {epoch_val_loss:.4f} | Val Acc: {epoch_val_acc:.2%}")

    # --- Save Model ---
    print(f"\nSaving model to: {model_save_path}...")
    torch.save(model.state_dict(), model_save_path)
    print("Model saved successfully.")

    # --- Detailed Validation Evaluation ---
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels_batch in val_loader:
            images = images.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels_batch.numpy())

    # Calculate metrics
    accuracy = accuracy_score(all_labels, all_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average='binary', pos_label=1)
    conf_mat = confusion_matrix(all_labels, all_preds)

    # Save metrics in JSON
    metrics = {
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1_score": float(f1),
        "confusion_matrix": conf_mat.tolist()
    }
    
    metrics_path = os.path.join(results_dir, "metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=4)
    print(f"Saved evaluation metrics to: {metrics_path}")

    # Save confusion matrix text representation
    conf_mat_path = os.path.join(results_dir, "confusion_matrix.txt")
    with open(conf_mat_path, "w") as f:
        f.write("Confusion Matrix:\n")
        f.write("                    Predicted Other  Predicted Owner\n")
        f.write(f"Actual Other (0):   {conf_mat[0][0]:<15} {conf_mat[0][1]:<15}\n")
        f.write(f"Actual Owner (1):   {conf_mat[1][0]:<15} {conf_mat[1][1]:<15}\n")
    print(f"Saved confusion matrix visualization to: {conf_mat_path}")

    # Plot and save training/validation curve charts
    plt.figure(figsize=(12, 5))
    
    # Loss Plot
    plt.subplot(1, 2, 1)
    plt.plot(range(1, num_epochs+1), history["train_loss"], label="Train Loss", marker='o')
    plt.plot(range(1, num_epochs+1), history["val_loss"], label="Val Loss", marker='x')
    plt.title("Training and Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.grid(True)
    plt.legend()
    
    # Accuracy Plot
    plt.subplot(1, 2, 2)
    plt.plot(range(1, num_epochs+1), history["train_acc"], label="Train Acc", marker='o')
    plt.plot(range(1, num_epochs+1), history["val_acc"], label="Val Acc", marker='x')
    plt.title("Training and Validation Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.grid(True)
    plt.legend()
    
    plt.tight_layout()
    plot_path = os.path.join(results_dir, "loss_accuracy_curves.png")
    plt.savefig(plot_path)
    plt.close()
    print(f"Saved metric curves plot to: {plot_path}")
    print("\nTraining execution complete.")

if __name__ == "__main__":
    main()

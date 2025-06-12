# Configuration pour Google Colab
# Copiez ce code dans une cellule de Colab

# 1. Installation des dépendances
!pip install torch torchvision

# 2. Vérification de l'installation
import torch
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA disponible: {torch.cuda.is_available()}")

# 3. Création des fichiers nécessaires
%%writefile model.py
import torch.nn as nn

class SimpleCNN(nn.Module):
    def __init__(self, num_classes):
        super(SimpleCNN, self).__init__()

        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 224 -> 112

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 112 -> 56

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 56 -> 28
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 28 * 28, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

def create_model(num_classes):
    return SimpleCNN(num_classes)

%%writefile dataset.py
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import os

def get_transforms(img_size=(224, 224)):
    return transforms.Compose([
        transforms.Resize(img_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],  # valeurs standard ImageNet
                             std=[0.229, 0.224, 0.225])
    ])

def get_dataloaders(data_dir="data/data/popular_street_foods", batch_size=32, img_size=(224, 224)):
    transform = get_transforms(img_size)

    train_path = os.path.join(data_dir, "train")
    val_path   = os.path.join(data_dir, "val")
    test_path  = os.path.join(data_dir, "test")

    train_dataset = datasets.ImageFolder(root=train_path, transform=transform)
    val_dataset   = datasets.ImageFolder(root=val_path, transform=transform)
    test_dataset  = datasets.ImageFolder(root=test_path, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(val_dataset, batch_size=batch_size)
    test_loader  = DataLoader(test_dataset, batch_size=batch_size)

    return train_loader, val_loader, test_loader, train_dataset.classes

%%writefile train.py
import torch
import torch.nn as nn
import torch.optim as optim

from dataset import get_dataloaders
from model import create_model

#sert à definir si les tenseurs seront traites sur le GPU ou le CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Utilisation de: {device}")

# Charger les DataLoaders
train_loader, val_loader, test_loader, class_names = get_dataloaders()

# Initialiser le modèle
num_classes = len(class_names)
model = create_model(num_classes)
model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
epochs = 15

for epoch in range(epochs):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

    train_acc = 100. * correct / total
    print(f"Epoch [{epoch+1}/{epochs}] - Loss: {running_loss/len(train_loader):.4f}, Accuracy: {train_acc:.2f}%")

    # Validation
    model.eval()
    val_correct = 0
    val_total = 0

    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = outputs.max(1)
            val_total += labels.size(0)
            val_correct += predicted.eq(labels).sum().item()

    val_acc = 100. * val_correct / val_total
    print(f"Validation Accuracy: {val_acc:.2f}%\n")

# Sauvegarder le modèle
torch.save(model.state_dict(), "model.pth")
print("Modèle sauvegardé avec succès!") 
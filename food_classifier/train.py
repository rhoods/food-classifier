import torch
import torch.nn as nn
import torch.optim as optim

from dataset import get_dataloaders
from model import create_model

#sert à definir si les tenseurs seront traites sur le GPU ou le CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Charger les DataLoaders
train_loader, val_loader, test_loader, class_names = get_dataloaders()

# Initialiser le modèle
num_classes = len(class_names)
model = create_model(num_classes)
model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
epochs = 30

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

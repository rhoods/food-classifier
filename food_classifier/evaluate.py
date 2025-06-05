import torch
from dataset import get_dataloaders
from model import create_model

#sert à definir si les tenseurs seront traites sur le GPU ou le CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Charger les DataLoaders (on récupère le test_loader)
_, _, test_loader, class_names = get_dataloaders()

# Initialiser le modèle (même architecture que lors de l'entraînement)
num_classes = len(class_names)
model = create_model(num_classes)
model.to(device)

# Charger les poids entraînés
model.load_state_dict(torch.load("model.pth"))
model.eval()

correct = 0
total = 0

with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

print('correct : ', correct)
print('total : ', total)

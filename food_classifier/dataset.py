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

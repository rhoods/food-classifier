import torch.nn as nn
import torch

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

        # Calcul dynamique de la taille des features
        # Avec 3 MaxPool2d(2), l'image 224x224 devient 28x28
        # Donc la taille finale est 128 * 28 * 28 = 100352
        self._calculate_feature_size()

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self.feature_size, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )

    def _calculate_feature_size(self):
        """Calcule dynamiquement la taille des features après les couches convolutives"""
        # Créer un tenseur factice pour calculer la taille
        x = torch.randn(1, 3, 224, 224)
        with torch.no_grad():
            x = self.features(x)
            self.feature_size = x.view(1, -1).size(1)

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

def create_model(num_classes):
    return SimpleCNN(num_classes)






class ConvBlock(nn.Module):
    """Bloc convolutif réutilisable avec BatchNorm"""
    def __init__(self, in_channels, out_channels, num_convs=2):
        super(ConvBlock, self).__init__()
        layers = []
        
        for i in range(num_convs):
            layers.extend([
                nn.Conv2d(in_channels if i == 0 else out_channels, 
                         out_channels, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)
            ])
        
        self.conv_block = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.conv_block(x)

class ModernCNN(nn.Module):
    def __init__(self, num_classes):
        super(ModernCNN, self).__init__()

        self.features = nn.Sequential(
            # Bloc 1: 3 -> 64
            ConvBlock(3, 64, num_convs=2),
            nn.MaxPool2d(2),  # 224 -> 112

            # Bloc 2: 64 -> 128
            ConvBlock(64, 128, num_convs=2),
            nn.MaxPool2d(2),  # 112 -> 56

            # Bloc 3: 128 -> 256
            ConvBlock(128, 256, num_convs=3),  # 3 couches dans ce bloc
            nn.MaxPool2d(2),  # 56 -> 28

            # Bloc 4: 256 -> 512
            ConvBlock(256, 512, num_convs=3),
            nn.MaxPool2d(2),  # 28 -> 14

            # Bloc 5: 512 -> 512
            ConvBlock(512, 512, num_convs=3),
        )

        # Global Average Pooling au lieu de Flatten
        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.global_avg_pool(x)
        x = x.view(x.size(0), -1)  # Flatten
        x = self.classifier(x)
        return x
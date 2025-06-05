import os
import shutil
import random
from pathlib import Path

def split_dataset(source_dir, dest_dir, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15):
    assert train_ratio + val_ratio + test_ratio == 1.0, "Les ratios doivent faire 100%"

    categories = [d for d in os.listdir(source_dir) if os.path.isdir(os.path.join(source_dir, d))]

    for category in categories:
        images = list(Path(source_dir, category).glob("*.jpg"))
        random.shuffle(images)

        train_count = int(len(images) * train_ratio)
        val_count = int(len(images) * val_ratio)

        splits = {
            "train": images[:train_count],
            "val": images[train_count:train_count + val_count],
            "test": images[train_count + val_count:]
        }

        for split_name, split_images in splits.items():
            split_path = Path(dest_dir, split_name, category)
            split_path.mkdir(parents=True, exist_ok=True)

            for img in split_images:
                shutil.copy(img, split_path / img.name)

    print("✅ Dataset divisé avec succès.")

if __name__ == "__main__":
    src = Path("data/data/popular_street_foods/dataset")   # Dossier d'origine où sont toutes les images classées
    dst = Path("data/data/popular_street_foods/")       # Racine où créer les dossiers train/val/test
    split_dataset(src, dst)

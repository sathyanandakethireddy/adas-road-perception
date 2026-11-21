
import random
import shutil
from pathlib import Path

# Original dataset paths
BASE_DIR = Path("data/road_damage")
IMAGE_DIR = BASE_DIR / "data" / "images"
LABEL_DIR = BASE_DIR / "data" / "labels-YOLO"

# Destination paths
OUTPUT_DIR = BASE_DIR / "dataset"

# Reproducible random split
random.seed(42)

# Supported image formats
image_extensions = {".jpg", ".jpeg", ".png"}

# Collect image files
images = [
    path for path in IMAGE_DIR.iterdir()
    if path.is_file() and path.suffix.lower() in image_extensions
]

# Keep only images that have matching annotation files
pairs = [
    image for image in images
    if (LABEL_DIR / f"{image.stem}.txt").exists()
]

if not pairs:
    raise RuntimeError("No matching image and annotation pairs found.")

# Shuffle and split
random.shuffle(pairs)
split_index = int(len(pairs) * 0.8)

train_images = pairs[:split_index]
val_images = pairs[split_index:]

def copy_pairs(image_list, split):
    for image in image_list:
        label = LABEL_DIR / f"{image.stem}.txt"

        shutil.copy2(
            image,
            OUTPUT_DIR / "images" / split / image.name
        )

        shutil.copy2(
            label,
            OUTPUT_DIR / "labels" / split / label.name
        )

copy_pairs(train_images, "train")
copy_pairs(val_images, "val")

print(f"Total matched pairs: {len(pairs)}")
print(f"Training pairs: {len(train_images)}")
print(f"Validation pairs: {len(val_images)}")
print("Dataset splitting completed.")

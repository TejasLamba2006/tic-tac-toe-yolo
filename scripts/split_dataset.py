from pathlib import Path
import random
import shutil

random.seed(42)

dataset = Path("dataset")

images_dir = dataset / "images"
labels_dir = dataset / "labels"

train_images = dataset / "train" / "images"
train_labels = dataset / "train" / "labels"

val_images = dataset / "val" / "images"
val_labels = dataset / "val" / "labels"

test_images = dataset / "test" / "images"
test_labels = dataset / "test" / "labels"

for p in [
    train_images, train_labels,
    val_images, val_labels,
    test_images, test_labels
]:
    p.mkdir(parents=True, exist_ok=True)

images = list(images_dir.glob("*.jpg"))

random.shuffle(images)

train_count = int(len(images) * 0.8)
val_count = int(len(images) * 0.1)

train_set = images[:train_count]
val_set = images[train_count:train_count + val_count]
test_set = images[train_count + val_count:]


def copy_set(files, img_dst, lbl_dst):
    for img in files:
        lbl = labels_dir / f"{img.stem}.txt"

        shutil.copy2(img, img_dst / img.name)

        if lbl.exists():
            shutil.copy2(lbl, lbl_dst / lbl.name)


copy_set(train_set, train_images, train_labels)
copy_set(val_set, val_images, val_labels)
copy_set(test_set, test_images, test_labels)

print(f"Train: {len(train_set)}")
print(f"Val: {len(val_set)}")
print(f"Test: {len(test_set)}")

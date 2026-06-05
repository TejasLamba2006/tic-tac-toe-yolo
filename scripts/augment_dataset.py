import cv2
import numpy as np
from pathlib import Path
import random

INPUT_DIR = "dataset3"
OUTPUT_DIR = "dataset_augmented"

Path(OUTPUT_DIR).mkdir(exist_ok=True)


def change_brightness(img, factor):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    hsv = hsv.astype(np.float32)

    hsv[:, :, 2] *= factor
    hsv[:, :, 2] = np.clip(hsv[:, :, 2], 0, 255)

    return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)


def change_contrast(img, alpha):
    return cv2.convertScaleAbs(img, alpha=alpha, beta=0)


def gaussian_blur(img):
    return cv2.GaussianBlur(img, (5, 5), 0)


def add_noise(img):
    noise = np.random.normal(0, 10, img.shape).astype(np.int16)

    noisy = img.astype(np.int16) + noise

    return np.clip(noisy, 0, 255).astype(np.uint8)


def rotate_small(img):
    h, w = img.shape[:2]

    angle = random.uniform(-5, 5)

    M = cv2.getRotationMatrix2D(
        (w // 2, h // 2),
        angle,
        1.0
    )

    return cv2.warpAffine(
        img,
        M,
        (w, h),
        borderMode=cv2.BORDER_REPLICATE
    )


def perspective_shift(img):
    h, w = img.shape[:2]

    shift = 15

    src = np.float32([
        [0, 0],
        [w, 0],
        [0, h],
        [w, h]
    ])

    dst = np.float32([
        [random.randint(0, shift), random.randint(0, shift)],
        [w-random.randint(0, shift), random.randint(0, shift)],
        [random.randint(0, shift), h-random.randint(0, shift)],
        [w-random.randint(0, shift), h-random.randint(0, shift)]
    ])

    M = cv2.getPerspectiveTransform(src, dst)

    return cv2.warpPerspective(
        img,
        M,
        (w, h),
        borderMode=cv2.BORDER_REPLICATE
    )


augmentations = [
    ("dark", lambda x: change_brightness(x, 0.6)),
    ("bright", lambda x: change_brightness(x, 1.4)),
    ("contrast_low", lambda x: change_contrast(x, 0.7)),
    ("contrast_high", lambda x: change_contrast(x, 1.4)),
    ("blur", gaussian_blur),
    ("noise", add_noise),
    ("rotate", rotate_small),
    ("perspective", perspective_shift),
]

images = list(Path(INPUT_DIR).glob("*.jpg"))

print(f"Found {len(images)} images")

for image_path in images:

    img = cv2.imread(str(image_path))

    stem = image_path.stem

    cv2.imwrite(
        str(Path(OUTPUT_DIR) / f"{stem}_orig.jpg"),
        img
    )

    for name, aug in augmentations:

        augmented = aug(img)

        cv2.imwrite(
            str(Path(OUTPUT_DIR) / f"{stem}_{name}.jpg"),
            augmented
        )

print("Done")

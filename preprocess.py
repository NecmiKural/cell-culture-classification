"""
Preprocessing script for deep learning assignment.

Pipeline (per image):
  1. Resize to TARGET_SIZE
  2. CLAHE contrast enhancement
  3. (Train only) Augment minority classes to MIN_SAMPLES
"""

import os
import random
import cv2
import numpy as np

# ── Config ────────────────────────────────────────────────────────────────────
INPUT_DIR   = "/Users/necmikural/Documents/assignments/deep_learning_1"
OUTPUT_DIR  = "/Users/necmikural/Documents/assignments/deep_learning_1/processed"
TARGET_SIZE = 224
MIN_SAMPLES = 200
USE_CLAHE   = True
SEED        = 42
# ─────────────────────────────────────────────────────────────────────────────

random.seed(SEED)
np.random.seed(SEED)


# ── Core preprocessing ───────────────────────────────────────────────────────

def apply_clahe(img):
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l = clahe.apply(l)
    return cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2BGR)


def load_and_preprocess(path):
    img = cv2.imread(path)
    if img is None:
        raise ValueError(f"Could not read: {path}")

    # Arka plan temizleme (threshold mask)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray, 15, 255, cv2.THRESH_BINARY)
    img = cv2.bitwise_and(img, img, mask=mask)

    img = cv2.resize(img, (TARGET_SIZE, TARGET_SIZE), interpolation=cv2.INTER_AREA)

    if USE_CLAHE:
        img = apply_clahe(img)
    return img


# ── Augmentation ─────────────────────────────────────────────────────────────

def augment(img):
    h, w = img.shape[:2]

    # Random rotation
    angle = random.uniform(-20, 20)
    M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    img = cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_REFLECT_101)

    # Random flip (horizontal and/or vertical)
    if random.random() < 0.5:
        img = cv2.flip(img, 1)
    if random.random() < 0.5:
        img = cv2.flip(img, 0)

    # Brightness / contrast
    alpha = random.uniform(0.8, 1.2)
    beta = random.randint(-20, 20)
    img = cv2.convertScaleAbs(img, alpha=alpha, beta=beta)

    # Random zoom
    zoom = random.uniform(0.85, 1.0)
    cy, cx = h // 2, w // 2
    half = int(min(h, w) * zoom / 2)
    img = img[cy - half: cy + half, cx - half: cx + half]
    img = cv2.resize(img, (TARGET_SIZE, TARGET_SIZE), interpolation=cv2.INTER_AREA)

    # Occasional blur
    if random.random() < 0.15:
        img = cv2.GaussianBlur(img, (3, 3), 0)

    return img


# ── Save / process ───────────────────────────────────────────────────────────

def save(img, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    cv2.imwrite(path, img)


def process_train():
    train_in = os.path.join(INPUT_DIR, "train")
    train_out = os.path.join(OUTPUT_DIR, "train")

    for cls in sorted(os.listdir(train_in)):
        cls_in = os.path.join(train_in, cls)
        cls_out = os.path.join(train_out, cls)
        if not os.path.isdir(cls_in):
            continue

        files = [f for f in os.listdir(cls_in)
                 if f.lower().endswith((".jpg", ".jpeg", ".png"))]
        n = len(files)
        print(f"  Class '{cls}': {n} images", end="")

        processed = []
        for fname in files:
            img = load_and_preprocess(os.path.join(cls_in, fname))
            save(img, os.path.join(cls_out, fname))
            processed.append(img)

        if 0 < n < MIN_SAMPLES:
            needed = MIN_SAMPLES - n
            print(f"  -> +{needed} aug", end="")
            for i in range(needed):
                aug = augment(processed[i % n].copy())
                save(aug, os.path.join(cls_out, f"aug_{i:05d}.jpg"))
            print(f" ({MIN_SAMPLES} total)")
        else:
            print()

    print(f"\nTrain -> {train_out}")


def process_test():
    test_in = os.path.join(INPUT_DIR, "test")
    test_out = os.path.join(OUTPUT_DIR, "test")
    files = [f for f in os.listdir(test_in)
             if f.lower().endswith((".jpg", ".jpeg", ".png"))]
    print(f"\nProcessing {len(files)} test images ...")
    for fname in files:
        img = load_and_preprocess(os.path.join(test_in, fname))
        save(img, os.path.join(test_out, fname))
    print(f"Test -> {test_out}")


if __name__ == "__main__":
    print(f"Config: size={TARGET_SIZE}, min_samples={MIN_SAMPLES}, clahe={USE_CLAHE}\n")
    process_train()
    process_test()
    print("Done.")

#!/usr/bin/env python3

import os
import cv2
import numpy as np


DATASET_DIR = os.path.expanduser(
    "~/ME5413_Final_Project/src/me5413_team_solution/config/digit_dataset"
)

MODEL_DIR = os.path.expanduser(
    "~/ME5413_Final_Project/src/me5413_team_solution/config/digit_model"
)

MODEL_PATH = os.path.join(MODEL_DIR, "digit_knn_train_data.npz")


def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def normalize_digit_image(img_bin):
    coords = cv2.findNonZero(img_bin)
    if coords is None:
        return np.zeros((96, 64), dtype=np.uint8)

    x, y, w, h = cv2.boundingRect(coords)
    digit = img_bin[y:y + h, x:x + w]

    canvas = np.zeros((96, 64), dtype=np.uint8)

    scale = min(64 / float(w), 96 / float(h))
    new_w = max(1, int(w * scale))
    new_h = max(1, int(h * scale))

    digit_resized = cv2.resize(
        digit, (new_w, new_h), interpolation=cv2.INTER_NEAREST
    )

    x_offset = (64 - new_w) // 2
    y_offset = (96 - new_h) // 2
    canvas[y_offset:y_offset + new_h, x_offset:x_offset + new_w] = digit_resized

    return canvas


def compute_hog(img_bin):
    hog = cv2.HOGDescriptor(
        _winSize=(64, 96),
        _blockSize=(16, 16),
        _blockStride=(8, 8),
        _cellSize=(8, 8),
        _nbins=9
    )
    feat = hog.compute(img_bin)
    return feat.flatten().astype(np.float32)


def preprocess_image(img_gray):
    _, img_bin = cv2.threshold(img_gray, 80, 255, cv2.THRESH_BINARY_INV)
    img_norm = normalize_digit_image(img_bin)
    feat = compute_hog(img_norm)
    return feat


def load_dataset(dataset_dir):
    samples = []
    labels = []

    for digit in range(1, 10):
        digit_dir = os.path.join(dataset_dir, str(digit))
        if not os.path.isdir(digit_dir):
            print(f"[WARN] Missing folder: {digit_dir}")
            continue

        file_list = sorted([
            f for f in os.listdir(digit_dir)
            if f.lower().endswith((".png", ".jpg", ".jpeg", ".bmp"))
        ])

        print(f"[INFO] Loading digit {digit}: {len(file_list)} images")

        for fname in file_list:
            path = os.path.join(digit_dir, fname)
            img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue

            feat = preprocess_image(img)
            samples.append(feat)
            labels.append(digit)

    if len(samples) == 0:
        raise RuntimeError("No valid training samples found.")

    samples = np.array(samples, dtype=np.float32)
    labels = np.array(labels, dtype=np.int32)
    return samples, labels


def main():
    ensure_dir(MODEL_DIR)

    print("[INFO] Loading dataset...")
    samples, labels = load_dataset(DATASET_DIR)

    print(f"[INFO] Total samples: {len(samples)}")
    print(f"[INFO] Feature dim   : {samples.shape[1]}")

    np.savez(MODEL_PATH, samples=samples, labels=labels)

    print(f"[DONE] Saved training data to: {MODEL_PATH}")


if __name__ == "__main__":
    main()

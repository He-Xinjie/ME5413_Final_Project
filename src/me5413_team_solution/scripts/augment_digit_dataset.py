#!/usr/bin/env python3

import os
import cv2
import numpy as np


RAW_DATASET_DIR = os.path.expanduser(
    "~/ME5413_Final_Project/src/me5413_team_solution/config/digit_dataset_raw"
)
OUT_DATASET_DIR = os.path.expanduser(
    "~/ME5413_Final_Project/src/me5413_team_solution/config/digit_dataset"
)


def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def normalize_background(img):
    """
    输入灰度图，输出统一到较稳定的灰度范围
    """
    if len(img.shape) != 2:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.GaussianBlur(img, (3, 3), 0)
    return img


def rotate_image(img, angle_deg):
    h, w = img.shape[:2]
    center = (w / 2.0, h / 2.0)
    M = cv2.getRotationMatrix2D(center, angle_deg, 1.0)
    rotated = cv2.warpAffine(
        img,
        M,
        (w, h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REPLICATE,
    )
    return rotated


def scale_image_keep_canvas(img, scale):
    h, w = img.shape[:2]
    new_w = max(1, int(w * scale))
    new_h = max(1, int(h * scale))

    resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    canvas = np.full((h, w), int(np.median(img)), dtype=np.uint8)

    if new_w <= w and new_h <= h:
        x0 = (w - new_w) // 2
        y0 = (h - new_h) // 2
        canvas[y0:y0 + new_h, x0:x0 + new_w] = resized
        return canvas

    # 如果放大超出画布，中心裁剪回原尺寸
    x0 = (new_w - w) // 2
    y0 = (new_h - h) // 2
    cropped = resized[y0:y0 + h, x0:x0 + w]
    return cropped


def shift_image(img, dx, dy):
    h, w = img.shape[:2]
    M = np.float32([[1, 0, dx], [0, 1, dy]])
    shifted = cv2.warpAffine(
        img,
        M,
        (w, h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REPLICATE,
    )
    return shifted


def adjust_brightness_contrast(img, alpha=1.0, beta=0):
    """
    alpha: contrast
    beta: brightness shift
    """
    out = cv2.convertScaleAbs(img, alpha=alpha, beta=beta)
    return out


def add_gaussian_noise(img, sigma=4.0):
    noise = np.random.normal(0, sigma, img.shape).astype(np.float32)
    out = img.astype(np.float32) + noise
    out = np.clip(out, 0, 255).astype(np.uint8)
    return out


def maybe_blur(img, ksize):
    if ksize <= 1:
        return img
    if ksize % 2 == 0:
        ksize += 1
    return cv2.GaussianBlur(img, (ksize, ksize), 0)


def augment_one_image(img):
    """
    返回若干增强版本
    """
    variants = []

    rotation_list = [0, -12, -8, -5, 5, 8, 12]
    scale_list = [0.92, 1.00, 1.08]
    shift_list = [(0, 0), (-4, 0), (4, 0), (0, -4), (0, 4)]
    brightness_contrast_list = [
        (1.00, 0),
        (0.95, -8),
        (1.05, 8),
        (1.10, 0),
        (0.90, 0),
    ]
    blur_list = [1, 3]

    for angle in rotation_list:
        rotated = rotate_image(img, angle)

        for scale in scale_list:
            scaled = scale_image_keep_canvas(rotated, scale)

            for dx, dy in shift_list:
                shifted = shift_image(scaled, dx, dy)

                for alpha, beta in brightness_contrast_list:
                    bc = adjust_brightness_contrast(shifted, alpha=alpha, beta=beta)

                    for blur_k in blur_list:
                        blurred = maybe_blur(bc, blur_k)

                        noisy = add_gaussian_noise(blurred, sigma=3.0)
                        variants.append(noisy)

    return variants


def save_image(path, img):
    cv2.imwrite(path, img)


def process_digit_folder(digit_dir_raw, digit_dir_out):
    ensure_dir(digit_dir_out)

    files = sorted([
        f for f in os.listdir(digit_dir_raw)
        if f.lower().endswith((".png", ".jpg", ".jpeg", ".bmp"))
    ])

    total_saved = 0

    for idx, fname in enumerate(files):
        path = os.path.join(digit_dir_raw, fname)
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue

        img = normalize_background(img)

        # 先保存原图
        base_name = os.path.splitext(fname)[0]
        save_image(os.path.join(digit_dir_out, f"{base_name}_orig.png"), img)
        total_saved += 1

        variants = augment_one_image(img)

        for j, aug in enumerate(variants):
            out_name = f"{base_name}_aug_{j:03d}.png"
            save_image(os.path.join(digit_dir_out, out_name), aug)
            total_saved += 1

    return total_saved


def main():
    ensure_dir(OUT_DATASET_DIR)

    grand_total = 0

    for digit in map(str, range(1, 10)):
        raw_dir = os.path.join(RAW_DATASET_DIR, digit)
        out_dir = os.path.join(OUT_DATASET_DIR, digit)

        if not os.path.isdir(raw_dir):
            print(f"[WARN] Missing raw folder: {raw_dir}")
            continue

        saved = process_digit_folder(raw_dir, out_dir)
        grand_total += saved
        print(f"[INFO] Digit {digit}: saved {saved} images")

    print(f"[DONE] Total saved images: {grand_total}")


if __name__ == "__main__":
    main()

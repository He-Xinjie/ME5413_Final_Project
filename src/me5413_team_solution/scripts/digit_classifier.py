#!/usr/bin/env python3

import os
import cv2
import numpy as np


class DigitClassifier:
    def __init__(self, model_path=None, k=3):
        if model_path is None:
            model_path = os.path.expanduser(
                "~/ME5413_Final_Project/src/me5413_team_solution/config/digit_model/digit_knn_train_data.npz"
            )

        self.model_path = model_path
        self.k = k
        self.knn = cv2.ml.KNearest_create()
        self.ready = False

        self._load_model()

    def _load_model(self):
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Model file not found: {self.model_path}")

        data = np.load(self.model_path)
        samples = data["samples"].astype(np.float32)
        labels = data["labels"].astype(np.int32)

        self.knn.train(samples, cv2.ml.ROW_SAMPLE, labels)
        self.ready = True

    def normalize_digit_image(self, img_bin):
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

    def compute_hog(self, img_bin):
        hog = cv2.HOGDescriptor(
            _winSize=(64, 96),
            _blockSize=(16, 16),
            _blockStride=(8, 8),
            _cellSize=(8, 8),
            _nbins=9
        )
        feat = hog.compute(img_bin)
        return feat.flatten().astype(np.float32)

    def preprocess_digit_roi(self, roi_gray):
        _, roi_bin = cv2.threshold(roi_gray, 80, 255, cv2.THRESH_BINARY_INV)
        roi_norm = self.normalize_digit_image(roi_bin)
        feat = self.compute_hog(roi_norm)
        return roi_norm, feat

    def classify(self, roi_gray):
        if not self.ready:
            return None, 0.0

        _, feat = self.preprocess_digit_roi(roi_gray)

        ret, result, neighbours, dist = self.knn.findNearest(
            feat.reshape(1, -1), k=self.k
        )

        pred = int(result[0][0])
        mean_dist = float(np.mean(dist))
        score = 1.0 / (1.0 + mean_dist / 1000.0)

        return pred, score

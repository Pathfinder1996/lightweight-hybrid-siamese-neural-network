import os

import cv2
import numpy as np

def load_image(folder):
    images = []
    for filename in sorted(os.listdir(folder)):
        path = os.path.join(folder, filename)
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if img is not None:
            images.append(img)
    return images

def create_labels(images1, images2):
    pairs_1 = []
    pairs_2 = []
    labels = []

    # generate positive samples
    for img1, img2 in zip(images1, images2):
        pairs_1.append(img1)
        pairs_2.append(img2)
        labels.append(0)

    # generate negative samples
    for i, img1 in enumerate(images1):
        exclude_indices = set(range(max(0, i - 11), min(len(images2), i + 11)))
        available_indices = [x for x in range(len(images2)) if x not in exclude_indices]
        idx = np.random.choice(available_indices)
        img2 = images2[idx]
        pairs_1.append(img1)
        pairs_2.append(img2)
        labels.append(1)

    return np.array(pairs_1), np.array(pairs_2), np.array(labels)

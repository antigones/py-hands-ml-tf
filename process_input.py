import csv
import glob
import os

from collections import Counter
from math import floor
from pathlib import Path

import numpy as np
import pandas as pd

from PIL import Image


INPUT_CSV_TRAIN = "./dataset_csv/sign_mnist_train/sign_mnist_train.csv"
INPUT_CSV_TEST = "./dataset_csv/sign_mnist_test/sign_mnist_test.csv"


BASE_TRAIN_DIR = Path("./dataset/train/")
BASE_VAL_DIR = Path("./dataset/val/")
BASE_TEST_DIR = Path("./dataset/test/")
IMAGES_PER_CLASS = 1000
NUM_CLASSES = 9
IMAGES_10P = floor(IMAGES_PER_CLASS * 0.10)
IMAGES_70P = floor(IMAGES_PER_CLASS * 0.70)
IMAGES_90P = floor(IMAGES_PER_CLASS * 0.90)

# Split: 70 / 10 / 20

for dir in [BASE_TRAIN_DIR, BASE_VAL_DIR, BASE_TEST_DIR]:
    for file in dir.glob('[0-9]/*'):
        print(file)
        file.unlink()

data = pd.read_csv(INPUT_CSV_TRAIN, dtype="uint8")
pixels = np.array(data, dtype="uint8")


print("Extracting train/val dataset")
counters_train_val = Counter()

# reshape the array into 28 x 28 array (as images are 28x28)
for p_serie in pixels:
    label = p_serie[0]
    pxs = p_serie[1:]
    p_serie = pxs.reshape((28, 28))
    if label in range(NUM_CLASSES):
        image = Image.fromarray(p_serie, "L")
        if counters_train_val[label] < IMAGES_70P:
            image.save(BASE_TRAIN_DIR / str(label) / f'{counters_train_val[label]}.jpg')
        if counters_train_val[label] > IMAGES_70P and counters_train_val[label] <= IMAGES_90P:
            image.save(BASE_VAL_DIR / str(label) / f'{counters_train_val[label]-IMAGES_70P}.jpg')

        counters_train_val[label] += 1

print("Extracting test dataset")
data = pd.read_csv(INPUT_CSV_TEST, dtype="uint8")
pixels = np.array(data, dtype="uint8")

counters_test = Counter()

for p_serie in pixels:
    label = p_serie[0]
    pxs = p_serie[1:]
    p_serie = pxs.reshape((28, 28))
    if label in range(NUM_CLASSES):
        image = Image.fromarray(p_serie, "L")
        if counters_test[label] < IMAGES_10P:
            image.save(BASE_TEST_DIR / str(label) / f'{counters_test[label]}.jpg')
        counters_test[label] += 1

import pandas as pd 
import numpy as np
import csv
from PIL import Image 
from collections import defaultdict

import os
import glob

BASE_TRAIN_DIR = './dataset/train/'
BASE_TEST_DIR = './dataset/test/'

for i in range(0,9):
    files = glob.glob(BASE_TRAIN_DIR+str(i)+'/*')
    for f in files:
        os.remove(f)

    files = glob.glob(BASE_TEST_DIR+str(i)+'/*')
    for f in files:
        os.remove(f)

data = pd.read_csv("./dataset_csv/sign_mnist_train/sign_mnist_train.csv",dtype='uint8')
pixels = np.array(data, dtype='uint8')
def def_value():
    return 0
counters = defaultdict(def_value)

#Reshape the array into 28 x 28 array (As images are 28x28)
for p_serie in pixels:
    label = p_serie[0]
    pxs = p_serie[1:]
    p_serie = pxs.reshape((28, 28))
    if label in range(0,9):
        image = Image.fromarray(p_serie,'L')
        if counters[label] <= 1000:
            image.save(BASE_TRAIN_DIR+str(label)+'/'+str(counters[label])+'.jpg')
        counters[label] = counters[label]+1

data = pd.read_csv("./dataset_csv/sign_mnist_test/sign_mnist_test.csv",dtype='uint8')
pixels = np.array(data, dtype='uint8')

counters_test = defaultdict(def_value)

#Reshape the array into 28 x 28 array (As images are 28x28)
for p_serie in pixels:
    label = p_serie[0]
    pxs = p_serie[1:]
    p_serie = pxs.reshape((28, 28))
    if label in range(0,9):
        image = Image.fromarray(p_serie,'L')
        if counters_test[label] <= 1000:
            image.save(BASE_TEST_DIR+str(label)+'/'+str(counters_test[label])+'.jpg')
        counters_test[label] = counters_test[label]+1
import os
import numpy as np
import glob
import shutil
import matplotlib.pyplot as plt
import time

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import tensorflow_hub as hub

# base_dir ='./dataset'

# classes = ['0','1','2','3','4','5','6','7','8']
classes = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I']

train_dir = "./dataset/train"
val_dir = "./dataset/val"
test_dir = "./dataset/test"
exported_model_name = "./model.h5"
batch_size = 32
IMG_SHAPE = 28

image_gen_train = ImageDataGenerator(
    rescale=1.0 / 255,
    rotation_range=45,
    width_shift_range=0.15,
    height_shift_range=0.15,
    horizontal_flip=True,
    zoom_range=0.5,
)

train_data_gen = image_gen_train.flow_from_directory(
    batch_size=batch_size,
    directory=train_dir,
    shuffle=True,
    color_mode="grayscale",
    target_size=(IMG_SHAPE, IMG_SHAPE),
    class_mode="sparse",
)
image_gen_val = ImageDataGenerator(rescale=1.0 / 255)
val_data_gen = image_gen_val.flow_from_directory(
    batch_size=batch_size,
    directory=val_dir,
    color_mode="grayscale",
    target_size=(IMG_SHAPE, IMG_SHAPE),
    class_mode="sparse",
)

image_gen_test = ImageDataGenerator(rescale=1.0 / 255)
test_data_gen = image_gen_test.flow_from_directory(
    batch_size=batch_size,
    directory=test_dir,
    color_mode="grayscale",
    target_size=(IMG_SHAPE, IMG_SHAPE),
    class_mode="sparse",
)

model = tf.keras.Sequential(
    [
        tf.keras.layers.Conv2D(32,(3, 3), padding="same", activation=tf.nn.relu, input_shape=(IMG_SHAPE, IMG_SHAPE, 1),),
        tf.keras.layers.MaxPooling2D((2, 2), strides=2),
        tf.keras.layers.Dropout(0.4),
        tf.keras.layers.Conv2D(64, (3, 3), padding="same", activation=tf.nn.relu),
        tf.keras.layers.MaxPooling2D((2, 2), strides=2),
        tf.keras.layers.Dropout(0.4),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation=tf.nn.relu),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(9, activation=tf.nn.softmax),
    ]
)

model.compile(
    optimizer="adam",
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=["accuracy"],
)

epochs = 80


print(model.summary())


history = model.fit(
    train_data_gen,
    steps_per_epoch=int(np.ceil(train_data_gen.n / float(batch_size))),
    epochs=epochs,
    validation_data=val_data_gen,
    validation_steps=int(np.ceil(val_data_gen.n / float(batch_size))),
)
# save model
model.save(exported_model_name)

model = tf.keras.models.load_model(
    exported_model_name,
    # `custom_objects` tells keras how to load a `hub.KerasLayer`
    custom_objects={"KerasLayer": hub.KerasLayer},
)

model.summary()
print(model.summary())


print("Evaluate on test data")
results = model.evaluate(test_data_gen, batch_size=batch_size)
print("test loss, test acc:", results)

image_list = []
batch_for_prediction = next(test_data_gen) 
image_list.append(batch_for_prediction[0][0])
image_list = np.array(image_list)
print(image_list.shape)  # (1,28,28,1), one image 28*28, 1 channel (grayscale)

predicted_batch = model.predict(image_list)
predicted_batch = tf.squeeze(predicted_batch).numpy()
predicted_ids = np.argmax(predicted_batch, axis=-1)
predicted_class_names = classes[predicted_ids]
print(predicted_class_names)

imgplot = plt.imshow(image_list[0])
plt.savefig("test.png")


acc = history.history["accuracy"]
val_acc = history.history["val_accuracy"]

loss = history.history["loss"]
val_loss = history.history["val_loss"]

epochs_range = range(epochs)

plt.figure(figsize=(8, 8))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label="Training Accuracy")
plt.plot(epochs_range, val_acc, label="Validation Accuracy")
plt.legend(loc="lower right")
plt.title("Training and Validation Accuracy")

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label="Training Loss")
plt.plot(epochs_range, val_loss, label="Validation Loss")
plt.legend(loc="upper right")
plt.title("Training and Validation Loss")
plt.savefig("history.png")

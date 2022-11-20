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
import tarfile
from azureml.core import Dataset
from azureml.core import Workspace
import mlflow
import mlflow.tensorflow

mlflow.start_run()
#mlflow.keras.autolog()
mlflow.tensorflow.autolog()

# load training files from dataset
ws = Workspace.from_config()
def_blob_store = ws.get_default_datastore()

# Get a dataset by name
py_hands_dataset = Dataset.get_by_name(workspace=ws, name='py-hands-dataset')
py_hands_dataset.download('/tmp/py-hands')

# untar files

file = tarfile.open('/tmp/py-hands/train.tar')
# extracting train files
file.extractall('./dataset/')
file.close()

file = tarfile.open('/tmp/py-hands/val.tar')
# extracting val files
file.extractall('./dataset/')
file.close()

file = tarfile.open('/tmp/py-hands/test.tar')
# extracting test files
file.extractall('./dataset/')
file.close()

print(tf. __version__)
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
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
    metrics=["accuracy"],
)

epochs = 2


print(model.summary())


history = model.fit(
    train_data_gen,
    steps_per_epoch=int(np.ceil(train_data_gen.n / float(batch_size))),
    epochs=epochs,
    validation_data=val_data_gen,
    validation_steps=int(np.ceil(val_data_gen.n / float(batch_size))),
)
# save model

#model.save(exported_model_name)

print("Registering the model via MLFlow")

# saving the model to a file
#mlflow.keras.save_model(
#    keras_model=model,
#    path=os.path.join("t_model", "trained_model"),
#)

#mlflow.tensorflow.save_model(
#    mlflow_model=model,
#    path=os.path.join("t_model", "trained_model"),
#)


"""
model = tf.keras.models.load_model(
    exported_model_name,
    # `custom_objects` tells keras how to load a `hub.KerasLayer`
    custom_objects={"KerasLayer": hub.KerasLayer},
)

model.summary()
print(model.summary())
"""

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

# imgplot = plt.imshow(image_list[0])
#plt.show()
mlflow.log_image(image_list[0], "figure.png")
mlflow.end_run()
"""
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']

epochs_range = range(epochs)

plt.figure(figsize=(8, 8))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.show()
"""
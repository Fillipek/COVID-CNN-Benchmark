import os
os.add_dll_directory("C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v11.3/bin")
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow.keras import datasets, layers, models, preprocessing, applications
import matplotlib.pyplot as plt

DATA_DIR = "data/train"

IMG_SIZE = 240
IMG_SHAPE = (IMG_SIZE, IMG_SIZE, 3)
BATCH_SIZE = 48

datagen = tf.keras.preprocessing.image.ImageDataGenerator(
    rotation_range=180,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.1,
    zoom_range=0.1,
    fill_mode='nearest'
)

dataset_flow = datagen.flow_from_directory(
    DATA_DIR,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    shuffle=True
)

base_model = applications.ResNet50(
    input_shape=IMG_SHAPE,
    include_top=False
)
base_model.trainable = False

global_average_layer = tf.keras.layers.GlobalMaxPooling2D()
prediction_layer = layers.Dense(3, activation="softmax")

model = tf.keras.Sequential([
  base_model,
  layers.Conv2D(64, (7,7), strides=2),
  global_average_layer,
  prediction_layer
])

# model.summary()

base_learning_rate = 0.0001
model.compile(
    optimizer=tf.keras.optimizers.Adam(),
    loss=tf.keras.losses.CategoricalCrossentropy(),
    metrics=['accuracy', 'mse']
)

EPOCHS = 4

history = model.fit(
    dataset_flow,
    epochs=EPOCHS
)

acc = history.history['accuracy']
print(acc)

model.save(f"{IMG_SIZE}_{BATCH_SIZE}_ResNet50.h5")
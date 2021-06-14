import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Comment or change the path below if you with t toggle on/off GPU computing 
os.add_dll_directory("C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v11.3/bin")


import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import datasets, layers, models, preprocessing, applications
import matplotlib.pyplot as plt

### Configurable parameters

TRAIN_DIR = "data/train"
IMG_SIZE = 64
BATCH_SIZE = 48
BASE_NETWORK = applications.resnet50.ResNet50
EPOCHS = 4

### Data generator preparation

datagen = tf.keras.preprocessing.image.ImageDataGenerator(
    rotation_range=15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.1,
    zoom_range=0.1,
    fill_mode='nearest'
)

dataset_flow = datagen.flow_from_directory(
    TRAIN_DIR,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    shuffle=True
)

### Trained model

base_model = BASE_NETWORK(
    input_shape=(IMG_SIZE, IMG_SIZE, 3),
    include_top=False
)
base_model.trainable = False

global_average_layer = tf.keras.layers.GlobalAveragePooling2D()
prediction_layer = layers.Dense(3, activation="softmax")

model = tf.keras.Sequential([
  base_model,
  global_average_layer,
  prediction_layer
])

### Training

model.summary()

base_learning_rate = 0.0001
model.compile(
    optimizer=tf.keras.optimizers.Adam(),
    loss=tf.keras.losses.CategoricalCrossentropy(),
    metrics=['accuracy', 'mse']
)

history = model.fit(
    dataset_flow,
    epochs=EPOCHS
)

acc = history.history['accuracy']
print(acc)

model.save(f"{IMG_SIZE}_{BATCH_SIZE}_{BASE_NETWORK.__name__}.h5")
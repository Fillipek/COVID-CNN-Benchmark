import os
os.add_dll_directory("C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v11.3/bin")
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt

def plot_confusion_matrix(matrix, labels):
    fig,ax = plt.subplots()
    im = ax.imshow(matrix, cmap=plt.cm.nipy_spectral)
    n = len(labels)

    ax.set_xticks(np.arange(n))
    ax.set_xticklabels(labels)
    ax.set_yticks(np.arange(n))
    ax.set_yticklabels(labels)

    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    for i in range(n):
        for j in range(n):
            text = ax.text(j, i, matrix[i, j], ha="center", va="center", color="w")

    ax.set_title("Confusion matrix")
    fig.tight_layout()
    plt.show()

test_data_path = 'data/test'
model_name = '71_16_Xception.h5'
batch_size = 16
img_size = 71

target_names = ['normal', 'pneumonia', 'COVID-19']

test_datagen = ImageDataGenerator()

validation_generator = test_datagen.flow_from_directory(
    test_data_path,
    classes=target_names,
    target_size=(img_size, img_size),
    batch_size=batch_size,
    class_mode='categorical'
)
print(validation_generator.classes)
model = load_model(model_name)

# Confution Matrix and Classification Report
y_pred = model.predict(validation_generator)
eval_data = model.evaluate(validation_generator)
print(np.argmax(y_pred, axis=1))
y_pred_classes = np.argmax(y_pred, axis=1)

print('Confusion Matrix')
cm = confusion_matrix(validation_generator.classes, y_pred_classes)
plot_confusion_matrix(cm, target_names)

print('Classification Report')
print(classification_report(validation_generator.classes, y_pred_classes, target_names=target_names))




from habana_frameworks.tensorflow import load_habana_module
# tensorflow.compact.v1.disable_eager_execution()
load_habana_module()

import itertools
import os
import random
from xml.etree.ElementInclude import include

import cv2
import matplotlib.pyplot as plt
import numpy as np
import PIL
import tensorflow

from IPython.display import Image, display
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from tensorflow import keras
# from tensorflow.keras import optimizers, Dense, Flatten, layers
# from tensorflow.keras import optimizers
from tensorflow.keras.layers import (Activation, BatchNormalization, Conv2D,
                                     Dense, Dropout, Flatten, MaxPooling2D,
                                     SeparableConv2D)
# from tensorflow.keras.metrics import categorical_crossentropy
# from tensorflow.keras.metrics import sparse_categorical_crossentropy
from tensorflow.keras.models import Sequential
# from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.applications.vgg16 import preprocess_input


# listing local directories
real = "Dataset/real_and_fake_face/training_real"
fake = "Dataset/real_and_fake_face/training_fake"
datadir = "Dataset/real_and_fake_face"

real_path = os.listdir(real)
fake_path = os.listdir(fake)

training_data = []
IMG_SIZE = 224

def load_img(path):
    image = cv2.imread(path)
    image = cv2.resize(image, (224, 224))
    return image[...,::-1]


def prepare(image):
    IMG_SIZE = 224
    new_array = cv2.resize(image, (IMG_SIZE, IMG_SIZE)) 
    return new_array.reshape(-1, IMG_SIZE,IMG_SIZE,3)


categories = ["training_real" , "training_fake"]
# 0 ——> Real (Original) Images
# 1 ——> Fake (Photoshopped/Morphed) Images

def create_training_data():
    for category in categories:
        path = os.path.join(datadir, category)
        class_num = categories.index(category)
        for img in os.listdir(path):
            try:
                img_array = cv2.imread(os.path.join(path,img), cv2.IMREAD_UNCHANGED)
                new_array = cv2.resize(img_array,(IMG_SIZE,IMG_SIZE))
                training_data.append([new_array,class_num])
            except: pass


create_training_data()

training_data = np.array(training_data)
print(training_data.shape)
np.random.shuffle(training_data)

X, y = [], []

for features,label in training_data:
    X.append(features)
    y.append(label)

X = np.array(X).reshape(-1, IMG_SIZE, IMG_SIZE, 3)
y = np.array(y)

print(X.shape)
print(y.shape)
print(np.unique(y, return_counts = True))
# Expected Output: (array([0, 1]), array([1081,  960])) 

print(y[1:10])

X = X/255.0
# Performing Normalization

# Dataset split into training and testing groups
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.1, random_state=42)

print("Shape of test_x: ", X_train.shape)
print("Shape of train_y: ", y_train.shape)
print("Shape of test_x: ", X_test.shape)
print("Shape of test_y: ", y_test.shape)

print(y_test[1:10])
print(np.unique(y_train, return_counts = True))
print(np.unique(y_test, return_counts = True))

train_x = tensorflow.keras.utils.normalize(X_train, axis=1)
test_x = tensorflow.keras.utils.normalize(X_test, axis=1)

# First Sequential Model
model = tensorflow.keras.models.Sequential([
            tensorflow.keras.layers.Conv2D(filters=64, kernel_size=(3,3), 
                            padding="same", activation = 'relu', input_shape= X.shape[1:]),
            tensorflow.keras.layers.Conv2D(filters=64, kernel_size=(3,3), 
                            padding="same", activation = 'relu'),
            tensorflow.keras.layers.MaxPooling2D(pool_size=(2,2)),
            tensorflow.keras.layers.Conv2D(filters=64, kernel_size=(3,3), 
                            padding="same", activation = 'relu'),
            tensorflow.keras.layers.Conv2D(filters=64, kernel_size=(3,3), 
                            padding="same", activation = 'relu'),
            tensorflow.keras.layers.MaxPooling2D(pool_size=(2,2)),
            tensorflow.keras.layers.Dropout(0.25),
            tensorflow.keras.layers.Flatten(data_format=None),
            tensorflow.keras.layers.Dense(units=128, activation=tensorflow.nn.relu),
            tensorflow.keras.layers.Dense(units=2, activation=tensorflow.nn.softmax)
])
# Alternative for last layer try: model.add(Dense(units=1, activation='sigmoid'))

# Customizing SGD Optimizer
sgd = tensorflow.keras.optimizers.SGD(learning_rate=0.01, decay=1e-6, momentum=0.9, nesterov=True)

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
# Alternative for optimizer=sgd

hist = model.fit(X_train,y_train, batch_size=20, epochs=7, validation_split=0.1)

model.save('first_model.h5')

val_loss, val_acc = model.evaluate(X_test, y_test)
print(val_loss)
print(val_acc)

# Importing VGG-16 Model 
vgg16_model= tensorflow.keras.applications.vgg16.VGG16(
    include_top=True, 
    weights='imagenet', 
    input_tensor=None,
    input_shape=None, 
    pooling=None, 
    classes=1000,
    classifier_activation='softmax'
)

vgg16_model.summary()
print(type(vgg16_model))

# VGG—16 is trained for classification of 1000 different classes, but that is not required.
# Hence, we replace the last layer with one for a binary classifier.

# Customizing the model
model = Sequential()

for layer in vgg16_model.layers[:-1]: model.add(layer)
# Replicating the vgg16_model, excluding the output layer, to a new Sequential model.

for layer in model.layers: layer.trainable = False
# Iterating over each layer in our new Sequential model and setting them to be 
# non-trainable. This freezes the weights and other trainable parameters in each 
# layer so that they won't be updated when the fake and real face images are passed.

model.add(Dense(units=2, activation='softmax'))
# Alternative try: model.add(Dense(units=1, activation='sigmoid'))

# Customizing SGD Optimizer
sgd = tensorflow.keras.optimizers.SGD(learning_rate=0.01, decay=1e-6, momentum=0.9, nesterov=True)

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
# Alternative for optimizer=sgd

hist = model.fit(X_train,y_train, batch_size=20, epochs = 50, validation_split=0.1)

model.save('final_model.h5')
# Download this model

val_loss, val_acc = model.evaluate(X_test, y_test)
print(val_loss)
print(val_acc)
# Final accuracy and loss

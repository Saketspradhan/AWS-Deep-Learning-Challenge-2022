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
# from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import (Activation, BatchNormalization, Conv2D,
                                     Dense, Dropout, Flatten, MaxPooling2D,
                                     SeparableConv2D)
# from tensorflow.keras.metrics import (categorical_crossentropy, 
#                                       sparse_categorical_crossentropy)
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.resnet50 import ResNet50
# from keras.applications.resnet50 import ResNet50
# from keras.applications.resnet import ResNet50
from tensorflow.keras.applications.resnet import preprocess_input


# listing all local directories
real = "Dataset/real_and_fake_face/training_real"
fake = "Dataset/real_and_fake_face/training_fake"
datadir = "Dataset/real_and_fake_face"

real_path = os.listdir(real)
fake_path = os.listdir(fake)

training_data = []
IMG_SIZE = 224

# Preprocessing for testing images
def load_img(path):
    image = cv2.imread(path)
    image = cv2.resize(image, (224, 224))
    return image[...,::-1]


def prepare(image):
    IMG_SIZE = 224
    new_array = cv2.resize(image, (IMG_SIZE, IMG_SIZE)) 
    return new_array.reshape(-1, IMG_SIZE,IMG_SIZE,3)


categories = ["training_real", "training_fake"]

# Correspondance: 
# 0 ——> Real (Original) images
# 1 ——> Fake (Photoshopped/Morphed) images

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

# Randomizing the dataset
np.random.shuffle(training_data)

X, y = [], []

for features,label in training_data:
    X.append(features)
    y.append(label)

X = np.array(X).reshape(-1, IMG_SIZE, IMG_SIZE, 3)
y = np.array(y)

print(X.shape)
print(y.shape)
print(np.unique(y, return_counts=True))
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
print(np.unique(y_train, return_counts=True))
print(np.unique(y_test, return_counts=True))

train_x = tensorflow.keras.utils.normalize(X_train, axis=1)
test_x = tensorflow.keras.utils.normalize(X_test, axis=1)

# Initializing a new sequential Model
resnet_model = Sequential()

# Importing ResNet50 Model
pretrained_model=tensorflow.keras.applications.resnet50.ResNet50(
    include_top=False,
    weights="imagenet",
    input_tensor=None,
    input_shape=None,
    pooling="avg",
    classes=1000,
)

for layer in pretrained_model.layers: layer.trainable=False

resnet_model.add(pretrained_model)
resnet_model.add(Flatten(data_format=None))
resnet_model.add(Dense(units=128, activation='relu'))
resnet_model.add(Dense(units=2, activation='softmax'))
# Alternative try: resnet_model.add(Dense(units=1, activation='sigmoid'))

resnet_model.summary()

sgd = tensorflow.keras.optimizers.SGD(learning_rate=0.001, decay=1e-6, momentum=0.9, nesterov=True)

resnet_model.compile(optimizer=sgd,
            loss='categorical_crossentropy',
            metrics=['accuracy'])
# Alternative for optimizer=Adam(learning_rate=0.001)

history = resnet_model.fit(X_train, validation_data=y_train, epochs=10)

resnet_model.save('resnet_model_final.h5')
# Download this model

val_loss, val_acc = resnet_model.evaluate(X_test, y_test)
print(val_loss)
print(val_acc)
# Final accuracy and loss

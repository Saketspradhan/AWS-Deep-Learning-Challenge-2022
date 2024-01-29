import itertools
import os
import random
from keras.models import Sequential

import cv2
import keras
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from IPython.display import Image, display
from keras.layers import Activation
from keras.layers.convolutional import Conv2D
from keras.layers.core import Dense, Flatten
from keras.metrics import categorical_crossentropy
from keras.models import Sequential
from keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import (Activation, BatchNormalization, Conv2D,
                                     Dense, Dropout, Flatten, MaxPooling2D,
                                     SeparableConv2D)
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam

from habana_frameworks.tensorflow import load_habana_module
tf.compact.v1.disable_eager_execution()
load_habana_module()

real = "Dataset/real_and_fake_face/training_fake"
fake = "Dataset/real_and_fake_face/training_fake"
datadir = "Dataset/real_and_fake_face"

real_path = os.listdir(real)
fake_path = os.listdir(fake)

def load_img(path):
    image = cv2.imread(path)
    image = cv2.resize(image, (224, 224))
    return image[...,::-1]


categories = ["training_real" , "training_fake"]

for category in categories:
    path = os.path.join(datadir, category)
    for img in os.listdir(path):
        img_array = cv2.imread(os.path.join(path,img), cv2.IMREAD_GRAYSCALE)
        break
    break

training_data = []
IMG_SIZE = 224
categories = ["training_real" , "training_fake"] # 0 -> Real and 1 -> Fake 

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
np.random.shuffle(training_data)

for sample in training_data[:10]:
    print(sample[1])

X = []
y = []

for features,label in training_data:
    X.append(features)
    y.append(label)

X = np.array(X).reshape(-1, IMG_SIZE, IMG_SIZE, 3)
y = np.array(y)

print(X.shape)
print(y.shape)
print(np.unique(y, return_counts = True))

print(y[1:10]) # Expected -> (array([0, 1]), array([1081,  960])) 

# Normalization 
X = X/255.0

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("Shape of test_x: ",X_train.shape)
print("Shape of train_y: ",y_train.shape)
print("Shape of test_x: ",X_test.shape)
print("Shape of test_y: ",y_test.shape)

print(y_test[1:10])

print(np.unique(y_train, return_counts = True))
print(np.unique(y_test, return_counts = True))

train_x = tf.keras.utils.normalize(X_train,axis=1)
test_x = tf.keras.utils.normalize(X_test, axis=1)

model = tf.keras.models.Sequential([
            tf.keras.layers.Conv2D(64,(3,3),activation = 'relu',
                            input_shape= X.shape[1:]),
            tf.keras.layers.Conv2D(64,(3,3),activation = 'relu'),
            tf.keras.layers.MaxPooling2D(2,2),
            tf.keras.layers.Conv2D(64,(3,3),activation = 'relu'),
            tf.keras.layers.Conv2D(64,(3,3),activation = 'relu'),
            tf.keras.layers.MaxPooling2D(2,2),
            tf.keras.layers.Dropout(0.25),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(128, activation=tf.nn.relu),
            tf.keras.layers.Dense(2, activation=tf.nn.softmax)

])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])


hist = model.fit(X_train,y_train, batch_size=20, epochs = 5, validation_split=0.1)

epochs = 5
train_loss = hist.history['loss']
val_loss = hist.history['val_loss']
train_acc = hist.history['accuracy']
val_acc = hist.history['val_accuracy']
xc = range(epochs)

# plt.figure(1,figsize=(7,5))
# plt.plot(xc,train_loss)
# plt.plot(xc,val_loss)
# plt.xlabel('num of Epochs')
# plt.ylabel('loss')
# plt.title('train_loss vs val_loss')
# plt.grid(True)
# plt.legend(['train','val'])
# print plt.style.available # use bmh, classic,ggplot for big pictures
# plt.style.use(['classic'])

# plt.figure(2,figsize=(7,5))
# plt.plot(xc,train_acc)
# plt.plot(xc,val_acc)
# plt.xlabel('num of Epochs')
# plt.ylabel('accuracy')
# plt.title('train_acc vs val_acc')
# plt.grid(True)
# plt.legend(['train','val'],loc=4)
# print plt.style.available # use bmh, classic,ggplot for big pictures
# plt.style.use(['classic'])

# val_loss, val_acc = model.evaluate(X_test, y_test)
# print(val_loss)
# print(val_acc)

predictions = model.predict(X_test)
predictions

rounded_predictions=model.predict(x = X_test, batch_size=10, verbose=0) 
classes_x=np.argmax(rounded_predictions,axis=1)

for i in rounded_predictions[:10]: print(i)

# %matplotlib inline

# cm = confusion_matrix(y_test,rounded_predictions)

def plot_confusion_matrix(cm, classes,
    normalize=False,
    title='Confusion matrix',
    cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
      
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else: print('Confusion matrix, without normalization')
    print(cm)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="black" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

cm_plot_labels = ['Real', 'Fake']
plot_confusion_matrix(cm=cm, classes=cm_plot_labels, title='Confusion Matrix')

"""- This means our model predicted every single time and got the 54% accuracy.

## VGG 16

- Checkout [this](https://neurohive.io/en/popular-networks/vgg16/) article for better Explanation.
"""

# %matplotlib inline

## I don't know why but without running this cell the below code is showing errors. 
## Running all these imports again solved it.
## Will figure out soon.

vgg16_model = keras.applications.vgg16.VGG16()
vgg16_model.summary()
print('VGG16 done')

"""- **predictions (Dense)          (None, 1000)              4097000**

- Vgg-16 is trained for classification of 1000 different classes, but we do not need that.
- So we will remove that last layer and add one of our own.
"""

# type(vgg16_model)
## This is not a sequential model.

"""### Customizing our model"""

model = Sequential()
for layer in vgg16_model.layers[:-1]: model.add(layer)
    
# Now, we have replicated the entire vgg16_model
# (excluding the output layer) to a new Sequential model, which we've just given the name model

for layer in model.layers: layer.trainable = False
    
# Next, we’ll iterate over each of the layers in our new Sequential model and set them to
# be non-trainable. This freezes the weights and other trainable parameters 
# in each layer so that they will not be updated when we pass in our images of fake and real faces.

model.add(Dense(2, activation='softmax'))
model.summary()

model.save('final_model')
print('Model saved')

"""**Model is Ready, time to Compile it**

"""

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

hist = model.fit(X_train,y_train, batch_size=20, epochs = 2, validation_split=0.1) # isme originally 50 epoch the

epochs = 2 # 50
train_loss = hist.history['loss']
val_loss = hist.history['val_loss']
train_acc = hist.history['accuracy']
val_acc = hist.history['val_accuracy']
xc = range(epochs)

plt.figure(1,figsize=(7,5))
plt.plot(xc,train_loss)
plt.plot(xc,val_loss)
plt.xlabel('num of Epochs')
plt.ylabel('loss')
plt.title('train_loss vs val_loss')
plt.grid(True)
plt.legend(['train','val'])
plt.style.use(['classic'])

plt.figure(2,figsize=(7,5))
plt.plot(xc,train_acc)
plt.plot(xc,val_acc)
plt.xlabel('num of Epochs')
plt.ylabel('accuracy')
plt.title('train_acc vs val_acc')
plt.grid(True)
plt.legend(['train','val'],loc=4)
#print plt.style.available # use bmh, classic,ggplot for big pictures
plt.style.use(['classic'])

val_loss, val_acc = model.evaluate(X_test, y_test)
print(val_loss)
print(val_acc)

predictions = model.predict(X_test)

rounded_predictions=model.predict(x = X_test, batch_size=10, verbose=0) 
classes_x=np.argmax(rounded_predictions,axis=1)
for i in rounded_predictions[:10]: print(i)

print(y_test[1:10])
print(np.unique(y_test, return_counts = True))

rounded_prediction = np.array(rounded_prediction)
print(np.unique(rounded_prediction, return_counts = True))

"""### Performance Analysis:

- **Actual Test Data**:  Real Faces[209], Fake Faces[200]

- **Predicted Data** :   Real Faces[263], Fake Faces[146]
"""

cm = confusion_matrix(y_test,rounded_prediction)
cm_plot_labels = ['Real', 'Fake']
plot_confusion_matrix(cm=cm, classes=cm_plot_labels, title='Confusion Matrix')

"""## Explanation of Matrix:

**True Positive**

- Correct Real Faces Predictions: 157 

**False Positive**

- Incorrect Real Faces Predictions: 106

**True Negative**

- Correct Fake Faces Predictions: 94

**False Negative**

- Incorrect Fake Faces Predictions: 52

## Predictions vs Actual.
"""

## For Image Display.
def load_img(path):
    image = cv2.resize(path, (224, 224))
    return image[...,::-1]

## For Predicting result.
def prepare(image):
    IMG_SIZE = 224
    new_array = cv2.resize(image, (IMG_SIZE, IMG_SIZE)) 
    return new_array.reshape(-1, IMG_SIZE,IMG_SIZE,3)


print("It's over!")

"""## 0 is Real Face
## 1 is Fake Face

"""

from keras.models import load_model
model = load_model('/content/gdrive/MyDrive/complete_model.h5')

# Change the value of n for other images. I have chosen these images randomly.
# Test 1

n = 171
prediction = model.predict(prepare(X_test[n]))
print("Probabilities: ",prediction)

x = ["Real-Face" if y_test[n]== 0 else "Fake-Face"]
print("Actual: ",x[0])

rounded_prediction=model.predict(x = prepare(X_test[n]), batch_size=10, verbose=0) 
classes_x=np.argmax(rounded_prediction,axis=1)

plt.imshow(load_img(X_test[n]), cmap='gray')
plt.show()

model.save('complete_model.h5')

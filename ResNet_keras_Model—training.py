from habana_frameworks.tensorflow import load_habana_module
# tensorflow.compact.v1.disable_eager_execution()
load_habana_module()

import itertools
import os
import random
from xml.etree.ElementInclude import include
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from absl import flags

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

"""
    Importing the ResNet keras model from 
    HabanaAI/Model-References/TensorFlow/computer_vision/Resnets/resnet_keras/ 

    ResNet keras model is a modified version of the original TensorFlow model 
    garden model. It uses a custom training loop, supports 50 layers and can 
    work both with SGD and LARS optimizers.
"""

# FLAGS = flags.FLAGS
# flags.DEFINE_float(
#     'weight_decay',
#     default=1e-4,
#     help=('Weight decay coefficiant for l2 regularization.'))

weight_decay=1e-4
layers = tensorflow.keras.layers

def _gen_l2_regularizer(use_l2_regularizer=True):
  return tensorflow.keras.regularizers.L2(
      weight_decay) if use_l2_regularizer else None
#FLAGS.weight_decay


def identity_block(input_tensor,
                    kernel_size,
                    filters,
                    stage,
                    block,
                    use_l2_regularizer=True,
                    batch_norm_decay=0.9,
                    batch_norm_epsilon=1e-5):

    """The identity block is the block that has no conv layer at shortcut.
    Args:
        input_tensor: input tensor
        kernel_size: default 3, the kernel size of middle conv layer at main path
        filters: list of integers, the filters of 3 conv layer at main path
        stage: integer, current stage label, used for generating layer names
        block: 'a','b'..., current block label, used for generating layer names
        use_l2_regularizer: whether to use L2 regularizer on Conv layer.
        batch_norm_decay: Moment of batch norm layers.
        batch_norm_epsilon: Epsilon of batch borm layers.

    Returns:
        Output tensor for the block.
    """

    filters1, filters2, filters3 = filters
    if tensorflow.keras.backend.image_data_format() == 'channels_last': bn_axis = 3
    else: bn_axis = 1
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    x = layers.Conv2D(
        filters1, (1, 1),
        use_bias=False,
        kernel_initializer='he_normal',
        kernel_regularizer=_gen_l2_regularizer(use_l2_regularizer),
        name=conv_name_base + '2a')(
            input_tensor)
    x = layers.BatchNormalization(
        axis=bn_axis,
        momentum=batch_norm_decay,
        epsilon=batch_norm_epsilon,
        name=bn_name_base + '2a')(
            x)
    x = layers.Activation('relu')(x)

    x = layers.Conv2D(
        filters2,
        kernel_size,
        padding='same',
        use_bias=False,
        kernel_initializer='he_normal',
        kernel_regularizer=_gen_l2_regularizer(use_l2_regularizer),
        name=conv_name_base + '2b')(
            x)
    x = layers.BatchNormalization(
        axis=bn_axis,
        momentum=batch_norm_decay,
        epsilon=batch_norm_epsilon,
        name=bn_name_base + '2b')(
            x)
    x = layers.Activation('relu')(x)

    x = layers.Conv2D(
        filters3, (1, 1),
        use_bias=False,
        kernel_initializer='he_normal',
        kernel_regularizer=_gen_l2_regularizer(use_l2_regularizer),
        name=conv_name_base + '2c')(
            x)
    x = layers.BatchNormalization(
        axis=bn_axis,
        momentum=batch_norm_decay,
        epsilon=batch_norm_epsilon,
        name=bn_name_base + '2c')(
            x)

    x = layers.add([x, input_tensor])
    x = layers.Activation('relu')(x)
    return x


def conv_block(input_tensor,
                kernel_size,
                filters,
                stage,
                block,
                strides=(2, 2),
                use_l2_regularizer=True,
                batch_norm_decay=0.9,
                batch_norm_epsilon=1e-5):

    """A block that has a conv layer at shortcut.
    Note that from stage 3,
    the second conv layer at main path is with strides=(2, 2)
    And the shortcut should have strides=(2, 2) as well
    Args:
        input_tensor: input tensor
        kernel_size: default 3, the kernel size of middle conv layer at main path
        filters: list of integers, the filters of 3 conv layer at main path
        stage: integer, current stage label, used for generating layer names
        block: 'a','b'..., current block label, used for generating layer names
        strides: Strides for the second conv layer in the block.
        use_l2_regularizer: whether to use L2 regularizer on Conv layer.
        batch_norm_decay: Moment of batch norm layers.
        batch_norm_epsilon: Epsilon of batch borm layers.

    Returns:
        Output tensor for the block.
    """

    filters1, filters2, filters3 = filters
    if tensorflow.keras.backend.image_data_format() == 'channels_last': bn_axis = 3
    else: bn_axis = 1
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    x = layers.Conv2D(
        filters1, (1, 1),
        use_bias=False,
        kernel_initializer='he_normal',
        kernel_regularizer=_gen_l2_regularizer(use_l2_regularizer),
        name=conv_name_base + '2a')(
            input_tensor)
    x = layers.BatchNormalization(
        axis=bn_axis,
        momentum=batch_norm_decay,
        epsilon=batch_norm_epsilon,
        name=bn_name_base + '2a')(
            x)
    x = layers.Activation('relu')(x)

    x = layers.Conv2D(
        filters2,
        kernel_size,
        strides=strides,
        padding='same',
        use_bias=False,
        kernel_initializer='he_normal',
        kernel_regularizer=_gen_l2_regularizer(use_l2_regularizer),
        name=conv_name_base + '2b')(
            x)
    x = layers.BatchNormalization(
        axis=bn_axis,
        momentum=batch_norm_decay,
        epsilon=batch_norm_epsilon,
        name=bn_name_base + '2b')(
            x)
    x = layers.Activation('relu')(x)

    x = layers.Conv2D(
        filters3, (1, 1),
        use_bias=False,
        kernel_initializer='he_normal',
        kernel_regularizer=_gen_l2_regularizer(use_l2_regularizer),
        name=conv_name_base + '2c')(
            x)
    x = layers.BatchNormalization(
        axis=bn_axis,
        momentum=batch_norm_decay,
        epsilon=batch_norm_epsilon,
        name=bn_name_base + '2c')(
            x)

    shortcut = layers.Conv2D(
        filters3, (1, 1),
        strides=strides,
        use_bias=False,
        kernel_initializer='he_normal',
        kernel_regularizer=_gen_l2_regularizer(use_l2_regularizer),
        name=conv_name_base + '1')(
            input_tensor)
    shortcut = layers.BatchNormalization(
        axis=bn_axis,
        momentum=batch_norm_decay,
        epsilon=batch_norm_epsilon,
        name=bn_name_base + '1')(
            shortcut)

    x = layers.add([x, shortcut])
    x = layers.Activation('relu')(x)
    return x


def resnet50(batch_size=None,
            use_l2_regularizer=True,
            rescale_inputs=False,
            batch_norm_decay=0.9,
            batch_norm_epsilon=1e-5):
  
    """Instantiates the ResNet50 architecture.
    Args:
        num_classes: `int` number of classes for image classification.
        batch_size: Size of the batches for each step.
        use_l2_regularizer: whether to use L2 regularizer on Conv/Dense layer.
        rescale_inputs: whether to rescale inputs from 0 to 1.
        batch_norm_decay: Moment of batch norm layers.
        batch_norm_epsilon: Epsilon of batch borm layers.
    
    Returns:
        A Keras model instance.
    """

    _R_MEAN = 123.68
    _G_MEAN = 116.78
    _B_MEAN = 103.94
    CHANNEL_MEANS = [_R_MEAN, _G_MEAN, _B_MEAN]

    input_shape = (224, 224, 3)
    img_input = layers.Input(shape=input_shape, batch_size=batch_size)
    
    if rescale_inputs:
        # Hub image modules expect inputs in the range [0, 1]. This rescales these
        # inputs to the range expected by the trained model.
        x = layers.Lambda( # pylint: disable=g-long-lambda
            lambda x: x * 255.0 - tensorflow.keras.backend.constant(    
                CHANNEL_MEANS,
                shape=[1, 1, 3],
                dtype=x.dtype),
            name='rescale')(
                img_input)
    else: x = img_input

    if tensorflow.keras.backend.image_data_format() == 'channels_first':
        x = layers.Permute((3, 1, 2))(x)
        bn_axis = 1
    else: bn_axis = 3 # channels_last

    block_config = dict(
        use_l2_regularizer=use_l2_regularizer,
        batch_norm_decay=batch_norm_decay,
        batch_norm_epsilon=batch_norm_epsilon)
    x = layers.ZeroPadding2D(padding=(3, 3), name='conv1_pad')(x)
    x = layers.Conv2D(
        64, (7, 7),
        strides=(2, 2),
        padding='valid',
        use_bias=False,
        kernel_initializer='he_normal',
        kernel_regularizer=_gen_l2_regularizer(use_l2_regularizer),
        name='conv1')(
            x)
    x = layers.BatchNormalization(
        axis=bn_axis,
        momentum=batch_norm_decay,
        epsilon=batch_norm_epsilon,
        name='bn_conv1')(
            x)
    x = layers.Activation('relu')(x)
    x = layers.MaxPooling2D((3, 3), strides=(2, 2), padding='same')(x)

    x = conv_block(
        x, 3, [64, 64, 256], stage=2, block='a', strides=(1, 1), **block_config)
    x = identity_block(x, 3, [64, 64, 256], stage=2, block='b', **block_config)
    x = identity_block(x, 3, [64, 64, 256], stage=2, block='c', **block_config)

    x = conv_block(x, 3, [128, 128, 512], stage=3, block='a', **block_config)
    x = identity_block(x, 3, [128, 128, 512], stage=3, block='b', **block_config)
    x = identity_block(x, 3, [128, 128, 512], stage=3, block='c', **block_config)
    x = identity_block(x, 3, [128, 128, 512], stage=3, block='d', **block_config)

    x = conv_block(x, 3, [256, 256, 1024], stage=4, block='a', **block_config)
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='b', **block_config)
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='c', **block_config)
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='d', **block_config)
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='e', **block_config)
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='f', **block_config)

    x = conv_block(x, 3, [512, 512, 2048], stage=5, block='a', **block_config)
    x = identity_block(x, 3, [512, 512, 2048], stage=5, block='b', **block_config)
    x = identity_block(x, 3, [512, 512, 2048], stage=5, block='c', **block_config)

    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(
        1,
        kernel_initializer=tensorflow.compat.v1.keras.initializers.random_normal(
            stddev=0.01),
        kernel_regularizer=_gen_l2_regularizer(use_l2_regularizer),
        bias_regularizer=_gen_l2_regularizer(use_l2_regularizer),
        name='fc1000')(
            x)

    # A sigmoid that is followed by the model loss must be done cannot be done
    # in float16 due to numeric issues. So we pass dtype=float32.
    x = layers.Activation('sigmoid', dtype='float32')(x)

    # Create model.
    return tensorflow.keras.Model(img_input, x, name='resnet50')


model = resnet50()

model.summary()

sgd = tensorflow.keras.optimizers.SGD(learning_rate=0.001, decay=1e-6, momentum=0.9, nesterov=True)

# ResNet keras model can work with both SGD and LARS optimizers.
# LARS not recommended for batch size less than 2000.
model.compile(optimizer=sgd,
              loss='categorical_crossentropy',
              metrics=['accuracy'])

hist = model.fit(X_train, y_train, batch_size=16, epochs=20, validation_data=(X_test, y_test))

model.save('final_model.h5')

# val_loss, val_acc = model.evaluate(X_test, y_test)
# print(val_loss)
# print(val_acc)
# Final accuracy and loss
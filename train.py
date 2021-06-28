
from tensorflow.python.util.nest import flatten
from tensorflow.python.ops.gen_array_ops import shape
from tensorflow.python.keras.engine import input_layer
from tensorflow.python.ops.gradients_util import _Inputs
import tensorflow as tf
# import tensorflow_hub as hub
import time
import string

import numpy as np
import h5py
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from PIL import Image
from os import listdir

plt.rcParams["figure.figsize"] = (13.0, 8.0)


chars = " 0123456789" + string.ascii_uppercase + string.ascii_lowercase
print(chars)


def read_img(path):
    img = Image.open(path).convert('LA')

    #img = img.resize((32, 32))
    return np.asarray(img) / 255


def load_data():
    imgFolder = list(filter(lambda x: not x.startswith("."), listdir("mnt/ramdisk/max/90kDICT32px")))
    imgFolder.sort()

    images = {}
    for path in imgFolder[0:1]:
        path = '1'
        imgPaths = listdir("mnt/ramdisk/max/90kDICT32px/{}".format(path))
        imgPaths.sort()
        for i in imgPaths:
            imageFolder = listdir("mnt/ramdisk/max/90kDICT32px/{}/{}".format(path, i))
            imageFolder.sort()
            for image in imageFolder:
                print(image)
                label = image.split('_')[1]
                label = label + ''.join([' ' for _ in range(32 - len(label))])
                print(label, '.')
                currentImage = read_img("mnt/ramdisk/max/90kDICT32px/{}/{}/{}".format(path, i, image))
                print(np.shape(currentImage))
                currentImage = np.pad(currentImage, ((0,0),(0,512 - len(currentImage[0])),(0,0)), mode='constant')
                print(len(currentImage), len(currentImage[0]), len(currentImage[0][0]))
                if not (label in images):
                    images[label] = []
                else:
                    print(images.keys())
                print(type(images[label])) 
                images[label].append(currentImage)

    return images

""" def load_data():
    imgFolder = list(filter(lambda x: not x.startswith("."), listdir("Images")))
    imgFolder.sort()

    images = {}
    i = 0
    for path in imgFolder:
        imgPaths = listdir("Images/{}".format(path))
        imgPaths.sort()
        images[chars[i]] = [read_img("Images/{}/{}".format(path, x)) for x in imgPaths]
        i = i + 1

    return images
 """

def split_data(labels, imgs):
    border = round(len(labels) * 0.8)
    training = ( labels[:border],imgs[:border])
    test = ( labels[border:],imgs[border:])
    return training, test


def flatten_dictionary(data):
    labels = []
    imgs = []
    for (key, imgArray) in data.items():
        idx = [chars.find(c) for c in key]
        for img in imgArray:
            labels.append(idx)
            imgs.append(img)

    return np.array(labels, dtype="int"), np.array(imgs)


def train(data):
    labels, imgs = data[0], data[1]
    input_layer = tf.keras.layers.Conv2D(
                kernel_size=5,
                filters=32,
                strides=1,
                activation="relu",
                kernel_initializer="variance_scaling",
            )
    pool1 =tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(input_layer)
    conv2 = tf.keras.layers.Conv2D(inputs=pool1,
                kernel_size=5,
                filters=64,
                strides=1,
                activation="relu",
                kernel_initializer="variance_scaling",
            )
    pool2 = tf.keras.layers.MaxPooling2D(inputs=conv2,pool_size=(2, 2), strides=(2, 2))
    conv3 = tf.keras.layers.Conv2D(inputs=pool2,
                kernel_size=3,
                filters=128,
                strides=1,
                activation="relu",
                kernel_initializer="variance_scaling",
            )
            # tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2)),
            # tf.keras.layers.Conv2D(
            #     kernel_size=3,
            #     filters=256,
            #     strides=1,
            #     activation="relu",
            #     kernel_initializer="variance_scaling",
            # ),
            # tf.keras.layers.MaxPooling2D(pool_size=(1, 2), strides=(1, 2)),
            # tf.keras.layers.Conv2D(
            #     kernel_size=3,
            #     filters=512,
            #     strides=1,
            #     activation="relu",
            #     kernel_initializer="variance_scaling",
            # ),
            # tf.keras.layers.MaxPooling2D(pool_size=(1, 2), strides=(1, 2)),
            # tf.keras.layers.Conv2D(
            #     kernel_size=3,
            #     filters=512,
            #     strides=1,
            #     activation="relu",
            #     kernel_initializer="variance_scaling",
            # ),
            # tf.keras.layers.MaxPooling2D(pool_size=(1, 2), strides=(1, 2)),
    flatten1 = tf.keras.layers.Flatten(inputs=conv3)
            # tf.keras.layers.Dense(
            #     units=400,
            #     activation="relu",
            # ),
            # tf.keras.layers.Dense(
            #     units=400,
            #     activation="relu",
            # ),
            # tf.keras.layers.Dense(
            #     units=400,
            #     activation="relu",
            # ),
            #tf.keras.layers.Dense(
            #    units=32 * len(chars),
            #    kernel_initializer="variance_scaling",
            #    # activation="softmax",
            #),

    model = tf.keras.Model(inputs=tf.keras.Input(shape=(len(imgs),31,512,1)), outputs=[tf.keras.layers.Dense(inputs=flatten,
                units=len(chars),
                kernel_initializer="variance_scaling",
                activation="softmax",
            ) for _ in range(32)])

    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    model.compile(optimizer="adam", loss=loss_fn, metrics=["accuracy"])

    print(imgs.shape, labels.shape)
    model.fit(imgs.reshape(len(imgs), 31, 512, 1), labels, epochs=10)
    return model


def evaluate(data, model):
    labels, imgs = data[0], data[1]

    print(model.evaluate(imgs.reshape(len(imgs), 31, 512, 1), labels, verbose=2))


data = load_data()
labels, imgs = flatten_dictionary(data)
training, test = split_data(labels, imgs)

# for line in training["A"][0]:
#     print(line)

# plt.imshow(training["A"][0])
# plt.show()

model = train(training)

evaluate(test, model)

model.save("model")

print("HIER")
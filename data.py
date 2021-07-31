from tensorflow.python.ops.gen_math_ops import maximum
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
    img = Image.open(path).convert("L")

    # img = img.resize((32, 32))
    return np.asarray(img) / 255

# assumes that the data is in the same folder
def load_data():
    imgFolder = list(
        filter(
            lambda x: not x.startswith("."), listdir("./mnt/ramdisk/max/90kDICT32px")
        )
    )
    imgFolder.sort()

    images = {}
    for path in range(300,400):
        print(path)
        path = str(path)
        imgPaths = listdir("mnt/ramdisk/max/90kDICT32px/{}".format(path))
        imgPaths.sort()
        for i in imgPaths:
            imageFolder = listdir("mnt/ramdisk/max/90kDICT32px/{}/{}".format(path, i))
            imageFolder.sort()
            for image in imageFolder:
                label = image.split("_")[1]
                label = label + "".join([" " for _ in range(32 - len(label))])
                currentImage = read_img(
                    "mnt/ramdisk/max/90kDICT32px/{}/{}/{}".format(path, i, image)
                )
                if len(currentImage) >= 10 and len(currentImage[0]) <= 512: 
                    currentImage = np.pad(
                        currentImage,
                        ((0, 32 - len(currentImage)), (0, 512 - len(currentImage[0]))),
                        mode="constant",
                    )
                    if not (label in images):
                        images[label] = []
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
    training = (labels[:border], imgs[:border])
    test = (labels[border:], imgs[border:])
    return training, test


def flatten_dictionary(data):
    labels = []
    imgs = []
    max = 0
    for (key, imgArray) in data.items():
        idx = np.array([chars.find(c) for c in key])
        for img in imgArray:
            labels.append(idx)
            max = maximum(max, len(img))
            imgs.append(img)
    print("Here:", np.shape(imgs))
    return np.array(labels, dtype="int"), np.array(imgs)

data = load_data()
labels, imgs = flatten_dictionary(data)
print(imgs.shape, labels.shape)


with open('collected_data.npy', 'wb') as f:
    np.savez_compressed(f, labels=labels, imgs=imgs)

#with open('test.npy', 'rb') as f:
#    a = np.load(f)
#    b = np.load(f)
print('done')
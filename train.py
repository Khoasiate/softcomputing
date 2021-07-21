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


chars = "0123456789" + string.ascii_uppercase + string.ascii_lowercase
print(chars)


def read_img(path):
    img = Image.open(path)

    img = img.resize((32, 32))
    return np.asarray(img) / 255


def load_data():
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


def split_data(data):
    border = round(len(data["0"]) * 0.8)
    training = {label: (imgs[:border]) for (label, imgs) in data.items()}
    test = {label: (imgs[border:]) for (label, imgs) in data.items()}
    return training, test


def flatten_dictionary(data):
    labels = []
    imgs = []
    for (key, imgArray) in data.items():
        idx = chars.find(key)
        for img in imgArray:
            labels.append(idx)
            imgs.append(img)

    return np.array(labels, dtype="int"), np.array(imgs)


def train(data):
    labels, imgs = flatten_dictionary(data)

    model = tf.keras.models.Sequential(
        [
            tf.keras.layers.Flatten(
                input_shape=(32, 32, 1),
            ),
            tf.keras.layers.Dense(
                units=32 * 32,
                activation="relu",
            ),
            tf.keras.layers.Dense(
                units=32 * 32,
                activation="relu",
            ),
            tf.keras.layers.Dense(
                units=32 * 32,
                activation="relu",
            ),
            tf.keras.layers.Dense(
                units=32 * 32,
                activation="relu",
            ),
            tf.keras.layers.Dense(
                units=32 * 32,
                activation="relu",
            ),
            tf.keras.layers.Dense(
                units=32 * 32,
                activation="relu",
            ),
            tf.keras.layers.Dense(
                units=32 * 32,
                activation="relu",
            ),
            tf.keras.layers.Dense(
                units=len(chars),
                kernel_initializer="variance_scaling",
                activation="softmax",
            ),
        ]
    )

    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    model.compile(optimizer="adam", loss=loss_fn, metrics=["accuracy"])

    print(imgs.shape, labels.shape)
    model.fit(imgs.reshape(len(imgs), 32, 32, 1), labels, epochs=5)
    return model


def evaluate(data, model):
    labels, imgs = flatten_dictionary(data)

    print(model.evaluate(imgs.reshape(len(imgs), 32, 32, 1), labels, verbose=2))


data = load_data()
training, test = split_data(data)

# for line in training["A"][0]:
#     print(line)

# plt.imshow(training["A"][0])
# plt.show()

model = train(training)

print(model.summary())

evaluate(test, model)

model.save("model")

print("HIER")


# def test_data():
#     imgPaths = list(filter(lambda x: not x.startswith("."), listdir("Images")))
#     imgPaths.sort()
#     images = []
#     for i in range(62):
#         imgPathsPaths = listdir("Images/{}".format(imgPaths[i]))
#         imgPathsPaths.sort()
#         img = read_img("Images/{}/{}".format(imgPaths[i], imgPathsPaths[0]))
#         images.append(img)

#     for i in range(len(images)):
#         img = np.asarray([images[i]])
#         prediction = model.predict(img)
#         print(prediction)
#         print("is:", chars[np.argmax(prediction)], "should be:", chars[i])


# test_data()

# probability_model = tf.keras.Sequential([model, tf.keras.layers.Softmax()])
# print(np.argmax(probability_model.predict(test["A"][0])))

import tensorflow as tf
import numpy as np
import string
from PIL import Image
from os import listdir

model = tf.keras.models.load_model("model")
chars = "0123456789" + string.ascii_uppercase + string.ascii_lowercase

print(model.summary())


def read_img(path):
    img = Image.open(path)

    img = img.resize((32, 32))
    return np.asarray(img) / 255


def test_data():
    imgPaths = list(filter(lambda x: not x.startswith("."), listdir("Images")))
    imgPaths.sort()
    images = []
    for i in range(62):
        imgPathsPaths = listdir("Images/{}".format(imgPaths[i]))
        imgPathsPaths.sort()
        img = read_img("Images/{}/{}".format(imgPaths[i], imgPathsPaths[0]))
        images.append(img)

    for i in range(len(images)):
        img = np.asarray([images[i]]).reshape(1, 32, 32, 1)
        prediction = model.predict(img)
        # print(prediction)
        print("is:", chars[np.argmax(prediction)], "should be:", chars[i])


test_data()

# probability_model = tf.keras.Sequential([model, tf.keras.layers.Softmax()])
# print(np.argmax(probability_model.predict(test["A"][0])))

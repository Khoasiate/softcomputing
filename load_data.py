
import numpy as np


""" with open('test.npy', 'rb') as f:
    data = np.load(f)
    labels = data['labels']
    imgs = data['imgs']

print(labels.shape)
print(imgs.shape)

print(imgs[34]) """
#print('done')
#print(data['labels'].shape)
#print()


class load_data:
    def load(file):
        with open('test.npy', 'rb') as f:
            data = np.load(f)
            labels = data['labels']
            imgs = data['imgs']
        return labels, imgs



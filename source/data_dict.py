from keras.datasets import *

from keras.utils import to_categorical
from numpy.random import shuffle
from numpy import arange
import numpy as np

cifar10_dict = {'name': 'cifar10',
                'classes': 10,
                'data': cifar10.load_data()
                }

fashion_mnist_dict = {'name': 'fashion_mnist',
                      'classes': 10,
                      'data': fashion_mnist.load_data()
                      }

datasets_dict = {'cifar10': cifar10_dict, 'fashion_mnist': fashion_mnist_dict}
datasets_dict = {'cifar10': cifar10_dict}

def load_data(name, image=True, small_data=False):
    data_dict = datasets_dict[name]
    classes = data_dict['classes']

    (x_train, y_train), (x_test, y_test) = data_dict['data']
    if len(x_train.shape) < 4:
        x_train = x_train[:, :, :, np.newaxis]
        x_test = x_test[:, :, :, np.newaxis]

    x_train, x_test = x_train.astype('float32'), x_test.astype('float32')
    if image:
        x_train /= 255
        x_test /= 255
    else:
        pass

    y_train, y_test = to_categorical(y_train, classes), to_categorical(y_test, classes)

    if small_data:
        num_train, num_test = x_train.shape[0], x_test.shape[0]
        train_indices, test_indices = arange(num_train), arange(num_test)
        shuffle(train_indices), shuffle(test_indices)
        subsets_train = train_indices[: num_train // 10]
        subsets_test = test_indices[: num_test // 10]

        return [(x_train[subsets_train], y_train[subsets_train]),
                (x_test[subsets_test], y_test[subsets_test]), classes]
    else:
        return [(x_train, y_train), (x_test, y_test), classes]

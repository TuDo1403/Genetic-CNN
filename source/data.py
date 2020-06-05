from data_dict import *
datasets_dict = { 'cifar10' : cifar10_dict}
from keras.utils import to_categorical

def load_data(name, normalize_input=True):
    data_dict = datasets_dict[name]
    classes = data_dict['classes']

    (x_train, y_train), (x_test, y_test) = data_dict['data']
    x_train, x_test = x_train.astype('float32'), x_test.astype('float32')
    if normalize_input:
        x_train /= 255
        x_test /= 255

    y_train, y_test = to_categorical(y_train, classes), to_categorical(y_test, classes)

    return (x_train, y_train), (x_test, y_test), classes
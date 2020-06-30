import numpy as np


def code_length_of(nodes):
    return (nodes * (nodes - 1)) // 2


from  keras.layers import  Conv2D, BatchNormalization, Activation, Add, Flatten, Dense, Dropout, MaxPool2D, AveragePooling2D
import keras.models


def conv_layer(X, num_f, k_size):
    node = X
    node = Conv2D(filters=num_f, kernel_size=k_size,
                  strides=(1, 1), padding='same')(node)
    node = BatchNormalization(axis=3)(node)
    node = Activation('relu')(node)
    return node


def do_something(a0, code, num_nodes, num_f, k_size):
    nodes = {'a1': conv_layer(a0, num_f, k_size)}

    for node in range(2, num_nodes + 1):
        nodes['a' + str(node)] = conv_layer(nodes['a' + str(node - 1)],
                                            num_f, k_size)

        end_idx = code_length_of(node)
        start_idx = end_idx - (node - 1)
        prev_nodes = code[start_idx: end_idx]
        connected_nodes = np.where(prev_nodes == 1)[0] + 1  # increment index number
        for prev_node in connected_nodes:
            nodes['a' + str(node)] = Add()([nodes['a' + str(node)],
                                            nodes['a' + str(prev_node)]])

    # Get node last
    node_L = conv_layer(nodes['a' + str(num_nodes)], num_f, k_size)
    return node_L


from keras.layers import Input


def genetic_model(architecture, hyper_params,
                  input_shape=(32, 32, 3), classes=1):
    Pool = MaxPool2D if hyper_params['pooling'] == 'max' else AveragePooling2D

    X_input = Input(input_shape)
    X = X_input

    stages = architecture['stages']
    _, nodes = stages[0]
    end_idx = code_length_of(nodes)
    code = architecture['code'][:end_idx]
    X = do_something(X, code, nodes, hyper_params['filters'],
                     hyper_params['kernel size'])
    X = Pool(pool_size=hyper_params['pool size'],
                 strides=hyper_params['strides'], padding='valid')(X)
    for i in range(1, len(stages)):
        _, nodes = stages[i]
        start_idx = code_length_of(stages[i - 1][1])
        end_idx = start_idx + code_length_of(nodes)
        code = architecture['code'][start_idx: end_idx]
        X = do_something(X, code, nodes, hyper_params['filters'] * (i + 1),
                         hyper_params['kernel size'])
        X = Pool(pool_size=hyper_params['pool size'],
                 strides=hyper_params['strides'], padding='valid')(X)

    # Output layer
    X = Flatten()(X)
    X = Dense(hyper_params['fc units'], activation='relu')(X)
    X = Dropout(hyper_params['drop out'])(X)
    X = Dense(classes, activation='softmax')(X)

    # Create model
    model = keras.models.Model(inputs=X_input, outputs=X)
    model.compile(loss='categorical_crossentropy',
                  optimizer=hyper_params['optimizer'], metrics=['accuracy'])
    return model

import timeit
from keras.datasets import cifar10
from keras.utils import to_categorical
architecture = { 'stages' : [('Stage1', 3), ('Stage2', 5)], 
                 'code' : np.array([0, 0, 1, 1, 1, 0, 1, 0, 0, 1, 1, 0, 1]),
                }
hyper_params = { 'optimizer' : 'adam',
                 'drop out' : 0.5,
                 'epochs' : 20,
                 'kernel size' : (5, 5),
                 'pool size' : 2,
                 'strides' : 2,
                 'filters' : 20,
                 'fc units': 500,
                 'pooling' : 'max'}

(x_train, y_train), (x_test, y_test) = cifar10.load_data()
# Convert class vectors to binary class matrices.
num_classes = 10
y_train = to_categorical(y_train, num_classes)
y_test = to_categorical(y_test, num_classes)

# Data normalization
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255


model = genetic_model(architecture, hyper_params,
                     input_shape=x_train.shape[1:], classes=num_classes)
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()
#model.save('./models/sga')
start = timeit.default_timer()
model.fit(x=x_train, y=y_train, epochs=20, validation_data=(x_test, y_test))
end = timeit.default_timer()
print('Elapsed time: {}'.format(end - start))
import numpy as np
import keras
from keras.datasets import cifar10

from keras.layers import Conv2D, BatchNormalization, Activation, Add, MaxPool2D, Flatten, Dense
from keras.models import Model

KERNEL_SIZE = 5

def code_length_of(nodes):
    return (nodes * (nodes - 1)) // 2

def conv_layer(X, num_f):
    node = X
    node = Conv2D(filters=num_f, kernel_size=(5, 5),
                               strides=(1, 1), padding='same')(node)
    node = BatchNormalization(axis=3)(node)
    node = Activation('relu')(node)
    return node

def do_something(a0, code, num_nodes, num_f):
    nodes = {}

    nodes['a1'] = conv_layer(a0, num_f)
    for node in range(2, num_nodes + 1):
        nodes['a' + str(node)] = conv_layer(nodes['a' + str(node-1)], num_f)
        
        end_idx = code_length_of(node)
        start_idx = end_idx - (node - 1)
        prev_nodes = code[start_idx : end_idx]
        connected_nodes = np.where(prev_nodes == 1)[0] + 1  # increment index number
        for prev_node in connected_nodes:
            nodes['a' + str(node)] = Add()([nodes['a' + str(node)], 
                                            nodes['a' + str(prev_node)]])

    
    node_L = conv_layer(nodes['a' + str(num_nodes)], num_f)
    return node_L

def genetic_model(architecture, input_shape=(32, 32, 3), classes=1):
    X_input = keras.layers.Input(input_shape)

    states = architecture['states']

    _, nodes = states[0]
    end_idx = code_length_of(nodes)
    code = architecture['code'][:end_idx]
    X = do_something(X_input, code, nodes, 32)
    for i in range(1, len(states)):
        _, nodes = states[i]
        start_idx = code_length_of(states[i-1][1])
        end_idx = start_idx + code_length_of(nodes)
        code = architecture['code'][start_idx : end_idx]
        X = do_something(X_input, code, nodes, 64)
        X = MaxPool2D((2, 2), strides=(2, 2), padding='valid')(X)

    # # Stage 1
    # X = do_something(X_input, architecture, 32)
    # X = keras.layers.MaxPool2D((2, 2), strides=(2, 2), padding='valid')(X)
    # # Stage 2
    # X = do_something(X, architecture, 64)
    # X = keras.layers.MaxPool2D((2, 2), strides=(2, 2), padding='valid')(X)

    # Output layer
    X = Flatten()(X)
    X = Dense(classes, activation='softmax')(X)

    # Create model
    model = Model(inputs=X_input, outputs=X, 
                name='model')

    return model

def train(model, data, epochs, batch_size):
    (x_train, y_train), test_sets = data
    model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size,
            validation_data=(test_sets))




# states = ['S1', 'S2', 'S3']
# nodes = [3, 5, 4]
# architecture = {}
# architecture['states'] = list(zip(states, nodes))
# architecture['code'] = np.random.randint(low=0, high=2, size=(19,))
# # architecture['num nodes'] = 4
# # architecture['code'] = np.array([1, 0, 0, 1, 1, 1])
# # architecture['num nodes'] = 5
# # architecture['code'] = np.array([0, 0, 0, 1, 0, 0, 0, 1, 0, 1])
# (x_train, y_train), (x_test, y_test) = cifar10.load_data()
# y_train = keras.utils.to_categorical(y_train, 10)
# y_test = keras.utils.to_categorical(y_test, 10)

# x_train = x_train.astype('float32')
# x_test = x_test.astype('float32')
# x_train /= 255
# x_test /= 255

# x_train = x_train[:25000]
# y_train = y_train[:25000]


# model_test = genetic_model(architecture, x_train.shape[1:], classes=10)
# model_test.compile(loss='categorical_crossentropy', optimizer='adam',
#                     metrics=["accuracy"])
# train(model_test, x_train, y_train, epochs=10, batch_size=64)
# print(evaluate_model(model_test, x_test, y_test))
# print('im here')


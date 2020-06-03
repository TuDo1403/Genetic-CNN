import numpy as np
import keras
from keras.datasets import cifar10

NUM_NODES = 3
KERNEL_SIZE = 5
STRIDE = 1


def convolutional_layer(X, num_f):
    node = X
    node = keras.layers.Conv2D(filters=num_f, kernel_size=(5, 5),
                               strides=(1, 1), padding='same')(node)
    node = keras.layers.BatchNormalization(axis=3)(node)
    node = keras.layers.Activation('relu')(node)
    return node

def do_something(a0, architecture, num_f):
    num_nodes = architecture['num nodes']
    code = architecture['code']
    nodes = {}

    nodes['a1'] = convolutional_layer(a0, num_f)
    for node in range(2, num_nodes + 1):
        nodes['a' + str(node)] = convolutional_layer(nodes['a' + str(node-1)], num_f)
        
        end_idx = (node * (node - 1)) // 2
        start_idx = end_idx - (node - 1)
        prev_nodes = code[start_idx : end_idx]
        connected_nodes = np.where(prev_nodes == 1)[0] + 1  # increment index number
        for prev_node in connected_nodes:
            nodes['a' + str(node)] = keras.layers.Add()([nodes['a' + str(node)], nodes['a' + str(prev_node)]])

    
    node_L = convolutional_layer(nodes['a' + str(num_nodes)], num_f)
    return node_L

def model(architecture, input_shape=(32, 32, 3), classes=1):
    X_input = keras.layers.Input(input_shape)

    # states = architecture['stages']
    # for state in states:
    #     pass
    # Stage 1
    X = do_something(X_input, architecture, 32)
    X = keras.layers.MaxPool2D((2, 2), strides=(2, 2), padding='valid')(X)
    # Stage 2
    X = do_something(X, architecture, 64)
    X = keras.layers.MaxPool2D((2, 2), strides=(2, 2), padding='valid')(X)

    # Output layer
    X = keras.layers.Flatten()(X)
    X = keras.layers.Dense(classes, activation='softmax')(X)

    # Create model
    model = keras.models.Model(inputs=X_input, outputs=X, 
                               name='model')

    return model

def train(model, X_train, Y_train, epochs, batch_size):
    model.fit(X_train, Y_train, epochs=epochs, batch_size=batch_size)
    pass

def evaluate_model(model, X_test, Y_test):
    evals = model.evaluate(X_test, Y_test, batch_size=128)
    return evals


def evaluate_architecture(architecture):
    pass

def decode():
    pass

architecture = {}
architecture['num nodes'] = 4
architecture['code'] = np.array([1, 0, 0, 1, 1, 1])
# architecture['num nodes'] = 5
# architecture['code'] = np.array([0, 0, 0, 1, 0, 0, 0, 1, 0, 1])
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
y_train = keras.utils.to_categorical(y_train, 10)
y_test = keras.utils.to_categorical(y_test, 10)

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255

# x_train = x_train[:25000]
# y_train = y_train[:25000]


model_test = model(architecture, x_train.shape[1:], classes=10)
model_test.compile(loss='categorical_crossentropy', optimizer='adam',
                    metrics=["accuracy"])
train(model_test, x_train, y_train, epochs=10, batch_size=32)
print(evaluate_model(model_test, x_test, y_test))
print('im here')


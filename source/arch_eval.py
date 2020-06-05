import numpy as np

from CNN import *
from data import load_data

def evaluate_model(code, arch, epochs, batch_size, data):
    arch['code'] = code
    (x_train, y_train), (x_test, y_test), classes = data
    model = genetic_model(arch, x_train.shape[1:], classes)
    model.compile(loss='categorical_crossentropy', 
                optimizer='adam', metrics=['arccuracy'])
    model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size,
            validation_data=(x_test, y_test))
    [_, fitness] = model.evaluate(x_test, y_test, batch_size=128)
    return fitness

def evaluate_table(ind):
    pass

def evaluate_architecture(nodes, epochs, batch_size, dataset):
    state_names = ['S' + str(i) for i in range(1, len(nodes)+1)]
    architecture = {}
    architecture['states'] = list(zip(state_names, nodes))
    data = load_data(dataset, normalize_input=True)
    return lambda code : evaluate_model(code, architecture, 
                                        epochs, batch_size, data)
    pass

def evaluate_model():
    pass
from genetic_CNN import genetic_model


def evaluate_model(code, arch, nn_dict, data):
    arch['code'] = code
    [(x_train, y_train), (x_test, y_test), classes] = data

    model = genetic_model(arch, nn_dict, x_train.shape[1:], classes)
    model.fit(x=x_train, y=y_train, epochs=nn_dict['epochs'],
              validation_data=(x_test, y_test))
    [_, fitness] = model.evaluate(x_test, y_test)
    return fitness


def evaluate_table(ind):
    pass


def evaluate_architecture(nodes, nn_dict, data):
    stage_names = ['S' + str(i) for i in range(1, len(nodes) + 1)]
    architecture = {'stages': list(zip(stage_names, nodes))}
    return lambda code: evaluate_model(code, architecture, nn_dict, data)

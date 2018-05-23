
import sys
import numpy as np
from copy import deepcopy
from collections import OrderedDict
from models.nn import get_model
from models.metrics import strict_accuracy_N


def grid_search_params(base_args, cv_samples, X, y, padding_y, num_tags, max_slen, num_words, wemb_dim, wemb_matrix, max_wlen, num_chars, cemb_dim, cemb_matrix):
    args = deepcopy(base_args)
    # define model parameters possible values
    grid_params = OrderedDict()
    #grid_params['epochs'] = [20, 30, 40]
    #grid_params['batch_size'] = [512, 1024, 2048]
    #grid_params['optimizer'] = ['rmsprop', 'adam', 'nadam']
    #grid_params['dropout'] = [0.1, 0.2, 0.3]
    #grid_params['model_size'] = [200, 300, 400]
    #grid_params['num_layers'] = [1, 2, 3]

    grid_params['epochs'] = [2, 3]
    grid_params['batch_size'] = [512]
    grid_params['optimizer'] = ['adam']
    grid_params['dropout'] = [0.1]
    grid_params['model_size'] = [200]
    grid_params['num_layers'] = [1]

    # define parameter combinations
    grid_space = np.array(np.meshgrid(grid_params['epochs'],
                                      grid_params['batch_size'],
                                      grid_params['optimizer'],
                                      grid_params['dropout'],
                                      grid_params['model_size'],
                                      grid_params['num_layers'])).T.reshape(-1,len(grid_params))

    print([grid_params[x] for x in grid_params.keys()])
    print(grid_space)

    # perform 3-fold cross validation for each parameter combinations
    print('[INFO] Grid-search will optimize the following hyper-parameters:', list(grid_params.keys()))
    cv_samples = 3

    X_cv_train = X
    if args.use_words and args.use_chars:
        block_size = len(X[0]) // cv_samples
        X_cv_dev = [[],[]]
    else:
        block_size = len(X) // cv_samples
        X_cv_dev = []
    y_cv_dev = []
    y_cv_train = y

    best_acc = 0
    best_params = None

    # test each parameter combination
    for i in range(grid_space.shape[0]):

        current_acc = 0
        cell = grid_space[i]
        args.optimizer = cell[2]
        args.dropout = float(cell[3])
        args.model_size = int(cell[4])
        args.num_layers = int(cell[5])

        model = get_model(args, num_tags,
                          max_slen, num_words, wemb_dim, wemb_matrix,
                          max_wlen, num_chars, cemb_dim, cemb_matrix)

        print('[INFO] ' + str(cv_samples) + '-fold cross-validation using the hyper-parameter set', cell)
        for _ in range(cv_samples):
            # rotate the block used for testing
            if args.use_words and args.use_chars:
                if len(X_cv_dev[0]):
                    X_cv_train = [np.append(X_cv_train[0], X_cv_dev[0], axis=0), np.append(X_cv_train[1], X_cv_dev[1], axis=0)]
                X_cv_dev = [X_cv_train[0][:block_size], X_cv_train[1][:block_size]]
                X_cv_train = [X_cv_train[0][block_size:], X_cv_train[1][block_size:]]

            else:
                if len(X_cv_dev):
                    X_cv_train = np.append(X_cv_train, X_cv_dev, axis=0)
                X_cv_dev = X_cv_train[:block_size]
                X_cv_train = X_cv_train[block_size:]

            if len(y_cv_dev):
                y_cv_train = np.append(y_cv_train, y_cv_dev, axis=0)
            y_cv_dev = y_cv_train[:block_size]
            y_cv_train = y_cv_train[block_size:]

            # fit the model
            history = model.fit(X_cv_train, np.array(y_cv_train), batch_size = int(cell[1]), epochs=int(cell[0]), validation_split=0.0, verbose=0)

            # obtain accuracy on validation

            # predictions on the test set
            p_cv_dev = model.predict(X_cv_dev, verbose=0)
            p_cv_dev = np.argmax(p_cv_dev, axis=-1) + 1
            true_cv_dev = np.argmax(y_cv_dev, axis=-1) + 1
            current_acc += strict_accuracy_N(true_cv_dev, p_cv_dev, 1)

        # average
        current_acc = current_acc / cv_samples
        print('[INFO] The accuracy with the given hyper-parameters is', current_acc)
        if current_acc > best_acc:
            best_acc = current_acc
            best_params = cell

    print('[INFO] The best set of hyper-parameters found is', best_params)

    # plug in args
    args.epochs = int(best_params[0])
    args.batch_size = int(best_params[1])
    args.optimizer = best_params[2]
    args.dropout = float(best_params[3])
    args.model_size = int(best_params[4])
    args.num_layers = int(best_params[5])
    return args


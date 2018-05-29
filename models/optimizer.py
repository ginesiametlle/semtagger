#!/usr/bin/python3
# this script provides procedures for optimizing hyper-parameters

import sys

import numpy as np

from copy import deepcopy
from collections import OrderedDict

from models.nn import get_model
from models.metrics import strict_accuracy_N


def grid_search_params(base_args, cv_samples, X, y, ignore_y, num_tags, max_slen, num_words, wemb_dim, wemb_matrix, max_wlen, num_chars, cemb_dim, cemb_matrix):
    """
    Performs grid-search and determines the best hyper-parameters for a given model
        Inputs:
            - base_args: parsed command line arguments
            - cv_samples: number of cross-validation folds
            - X: input data
            - y: output data
            - ignore_y: numerical output to ignore
            - num_tags: number of output tags
            - max_slen: maximum number of words in a sentence
            - num_words: size of the word embedding vocabulary
            - wemb_dim: dimensionality of the word embedding vectors
            - wemb_matrix: word embedding matrix
            - max_wlen: maximum number of characters in a word
            - num_chars: size of the character embedding vocabulary
            - cemb_dim: dimensionality of the character embedding vectors
            - cemb_matrix: character embedding matrix
        Returns:
            the best-performing hyper-parameter values
    """
    args = deepcopy(base_args)
    # define model parameters possible values
    grid_params = OrderedDict()
    grid_params['epochs'] = [15, 20]
    grid_params['batch_size'] = [512, 1024]
    grid_params['optimizer'] = ['adam', 'nadam']
    grid_params['dropout'] = [0.1, 0.2]
    grid_params['model_size'] = [160, 200]
    grid_params['num_layers'] = [2, 3]

    # define parameter combinations
    grid_space = np.array(np.meshgrid(grid_params['epochs'],
                                      grid_params['batch_size'],
                                      grid_params['optimizer'],
                                      grid_params['dropout'],
                                      grid_params['model_size'],
                                      grid_params['num_layers'])).T.reshape(-1,len(grid_params))

    # perform 3-fold cross validation for each parameter combinations
    print('[INFO] Grid-search will optimize the following hyper-parameters:', list(grid_params.keys()))

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
        print('[INFO] Performing ' + str(cv_samples) + '-fold cross-validation using the hyper-parameter set', cell)

        args.optimizer = cell[2]
        args.dropout = float(cell[3])
        args.model_size = int(cell[4])
        args.num_layers = int(cell[5])

        # rotate the block used for evaluation for each sample
        for _ in range(cv_samples):
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
            model = get_model(args, num_tags,
                              max_slen, num_words, wemb_dim, wemb_matrix,
                              max_wlen, num_chars, cemb_dim, cemb_matrix)
            history = model.fit(X_cv_train, np.array(y_cv_train), batch_size = int(cell[1]), epochs=int(cell[0]), validation_split=0.0, verbose=1)
            
            # obtain accuracy on the evaluation block
            p_cv_dev = model.predict(X_cv_dev, verbose=0)
            p_cv_dev = np.argmax(p_cv_dev, axis=-1) + 1
            true_cv_dev = np.argmax(y_cv_dev, axis=-1) + 1
            current_acc += strict_accuracy_N(true_cv_dev, p_cv_dev, ignore_y)

        # average accuracy scores over the number of folds
        current_acc = current_acc / cv_samples
        print('[INFO] The accuracy with the given hyper-parameters is', current_acc)
        if current_acc > best_acc:
            best_acc = current_acc
            best_params = cell

    # set the best parameters and return
    print('[INFO] The best set of hyper-parameters found is', best_params)
    args.epochs = int(best_params[0])
    args.batch_size = int(best_params[1])
    args.optimizer = best_params[2]
    args.dropout = float(best_params[3])
    args.model_size = int(best_params[4])
    args.num_layers = int(best_params[5])
    return args


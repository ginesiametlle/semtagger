#!/usr/bin/python3
# this script defines the structure of possible neural models

from copy import deepcopy

from keras.models import Model, Input
from keras.layers import Dense, Reshape, Conv2D, LeakyReLU, LSTM, GRU
from keras.layers import add, concatenate
from keras.layers import Embedding, BatchNormalization, Dropout, GaussianNoise
from keras.layers import TimeDistributed, Bidirectional
from keras_contrib.layers import CRF

from models.metrics import strict_accuracy
from utils.keras_mapper import get_optimizer, get_loss


def get_layer(args, num_units):
    """
    Obtains a neural recurrent layer
        Inputs:
            - args: command line arguments
            - num_units: number of neurons in a layer

        Returns:
            the layer defined by the command line arguments
    """
    if args.model == 'lstm':
        return LSTM(units=num_units,
                    activation=args.hidden_activation,
                    dropout=args.dropout,
                    recurrent_dropout=args.dropout,
                    return_sequences=True)
    if args.model == 'gru':
        return GRU(units=num_units,
                   activation=args.hidden_activation,
                   dropout=args.dropout,
                   recurrent_dropout=args.dropout,
                   return_sequences=True)
    if args.model == 'blstm':
        return Bidirectional(LSTM(units=num_units,
                                  activation=args.hidden_activation,
                                  dropout=args.dropout,
                                  recurrent_dropout=args.dropout,
                                  return_sequences=True),
                             merge_mode = 'concat')
    if args.model == 'bgru':
        return Bidirectional(GRU(units=num_units,
                                 activation=args.hidden_activation,
                                 dropout=args.dropout,
                                 recurrent_dropout=args.dropout,
                                 return_sequences=True),
                             merge_mode = 'concat')
    return None


def get_model(base_args, num_tags=0, max_slen=0, num_words=0, wemb_dim=0, wemb_matrix=None, max_wlen=0, num_chars=0, cemb_dim=0, cemb_matrix=None, optimizer=None, dropout=None, model_size=None, num_layers=None):
    """
    Obtains a neural model as a combination of layers
        Inputs:
            - base_args: command line arguments
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
            the compiled Keras neural model defined by the command line arguments
    """
    ## REDEFINE BASE PARAMETERS
    args = deepcopy(base_args)
    if optimizer:
        args.optimizer = optimizer
    if dropout:
        args.dropout = dropout
    if model_size:
        args.model_size = model_size
    if num_layers:
        args.num_layers = num_layers

    ## DEFINE NETWORK
    if args.use_words:
        # word input layer
        word_input = Input(shape=(max_slen,))
        # word embedding layer
        word_model = Embedding(input_dim=num_words,
                               output_dim=wemb_dim,
                               weights = [wemb_matrix],
                               input_length = max_slen,
                               trainable = bool(args.word_embeddings_trainable))(word_input)

    if args.use_chars:
        # character input layer
        char_input = Input(shape=(max_slen, max_wlen))
        # character embedding layer
        x = Reshape((max_slen * max_wlen, ))(char_input)
        x = Embedding(input_dim=num_chars,
                               output_dim=cemb_dim,
                               weights = [cemb_matrix],
                               input_length = max_slen * max_wlen,
                               trainable = bool(args.char_embeddings_trainable))(x)
        x = Reshape((max_slen, max_wlen, cemb_dim))(x)

        # build word-like features from character features using a residual network
        # the residual network is constructed by stacking residual blocks
        shortcut = x
        for _ in range(min(1, args.resnet_depth)):
            # build a residual block
            x = Conv2D(max_slen, kernel_size=(3, 3), padding='same', data_format='channels_first')(x)
            if args.batch_normalization:
                x = BatchNormalization()(x)
            x = LeakyReLU()(x)
            if args.dropout > 0:
                x = Dropout(args.dropout)(x)

            x = Conv2D(max_slen, kernel_size=(3, 3), padding='same', data_format='channels_first')(x)
            if args.batch_normalization:
                x = BatchNormalization()(x)

            # merge input and shortcut
            x = add([shortcut, x])
            x = LeakyReLU()(x)
            shortcut = x

        # finish building the character model
        char_model = Reshape((max_slen, max_wlen * cemb_dim))(x)

	# concat word and character features if needed
    if args.use_words and args.use_chars:
        model = concatenate([word_model, char_model])
    elif args.use_words:
        model = word_model
    elif args.use_chars:
        model = char_model

    # noise layer
    if args.noise_sigma:
        model = GaussianNoise(args.noise_sigma)(model)

    # batch normalization layer
    if args.batch_normalization:
        model = BatchNormalization(model)

    # recurrent layers
    num_units = args.model_size
    for _ in range(args.num_layers):
        layer = get_layer(args, num_units)
        model = layer(model)
        # batch normalization layer
        if args.batch_normalization:
            model = BatchNormalization(model)
        # halve hidden units for each new layer
        num_units = int(num_units / 2)

    # output layer
    if args.output_activation == 'crf':
        model = TimeDistributed(Dense(num_units, activation='relu'))(model)
        crf = CRF(num_tags, learn_mode='marginal')
        out = crf(model)
    else:
        out = TimeDistributed(Dense(num_tags, activation='softmax'))(model)


    ## DEFINE INPUT AND OUTPUT
    model_output = out
    if args.use_words and args.use_chars:
        model_input = [word_input, char_input]
    elif args.use_words:
        model_input = word_input
    elif args.use_chars:
        model_input = char_input

    model = Model(input=model_input, output=model_output)


    ## COMPILE NETWORK
    # for now this is only a single loss function
    # we use negative log-likelihood when the last layer is a CRF layer
    if args.output_activation == 'crf':
        model_losses = [crf.loss_function]
    else:
        model_losses = [get_loss(args.loss)]
    model_loss_weights = [1.0]

    # define metrics
    # we employ Keras default accuracy and our strict accuracy metric
    if args.output_activation == 'crf':
        model_metrics = [crf.accuracy, strict_accuracy]
    else:
        model_metrics = ['accuracy', strict_accuracy]

    # define optimizer
    model_opt = get_optimizer(args.optimizer)

    # compilation
    model.compile(optimizer=model_opt,
                  loss=model_losses,
                  loss_weights=model_loss_weights,
                  metrics=model_metrics)

    # return the built model ready to be used
    return model



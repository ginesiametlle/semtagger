#!/usr/bin/python3
# this script defines the structure of possible neural models

from keras.models import Model, Input
from keras.initializers import glorot_normal
from keras.layers import Dense, LSTM, GRU
from keras.layers import Embedding, BatchNormalization, Dropout, GaussianNoise
from keras.layers import TimeDistributed, Bidirectional
from keras_contrib.layers import CRF

from models.metrics import strict_accuracy
from utils.mapper_keras import get_optimizer, get_loss


def get_layer(args, num_units):
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


def get_model(args, max_wlen, num_words, wemb_dim, wemb_matrix, num_tags): #max_clen, num_chars, cemb_dim, cemb_matrix):

    if args.use_words:
        # word input layer
        word_input = Input(shape=(max_wlen,))
        # word embedding layer
        word_model = Embedding(input_dim=num_words,
                               output_dim=wemb_dim,
                               weights = [wemb_matrix],
                               input_length = max_wlen,
                               trainable = False)(word_input)

    if args.use_chars:
        # character input layer
        char_input = Input(shate=(max_clen,))
        # character embedding layer
        char_model = Embedding(input_dim=num_chars,
                               output_dim=cemb_dim,
                               weights = [cemb_matrix],
                               input_length = max_clen,
                               trainable = False)(char_input)

	# concat word and character features
    if args.use_words and args.use_chars:
        model = merge([word_model, char_model], mode='concat')
    elif args.use_words:
        model = word_model
    elif args.use_chars:
        model = char_model

    # noise layer
    if args.noise_sigma > 0:
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
        model = TimeDistributed(Dense(args.model_size, activation='relu'))(model)
        crf = CRF(num_tags)
        out = crf(model)

    if args.output_activation == 'softmax':
        out = TimeDistributed(Dense(num_tags, activation='softmax'))(model)

    # define inputs and outputs
    if args.use_words and args.use_chars:
        model_input = [word_input, char_input]
    elif args.use_words:
        model_input = word_input
    elif args.use_chars:
        model_input = char_input

    model_output = out
    model = Model(input=model_input, output=model_output)

    # define losses
    # we use negative log-likelihood when the last layer is a CRF layer
    if args.output_activation == 'crf':
        model_losses = [crf.loss_function]
    else:
        model_losses = [get_loss(args.loss)]

    model_loss_weights = [1.0]

    # define metrics
    if args.output_activation == 'crf':
        model_metrics = [crf.accuracy, strict_accuracy]
    else:
        model_metrics = ['accuracy', strict_accuracy]

    # define optimizer
    model_opt = get_optimizer(args.optimizer)

    # compile model
    model.compile(optimizer=model_opt,
                  loss=model_losses,
                  loss_weights=model_loss_weights,
                  metrics=model_metrics)

    # return the built model ready to be used
    return model


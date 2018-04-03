#!/usr/bin/python3
# this script defines the structure of possible neural models

from keras.models import Model, Input
from keras.initializers import glorot_normal
from keras.layers import Dense, LSTM, GRU
from keras.layers import Embedding, BatchNormalization, Dropout
from keras.layers import TimeDistributed, Bidirectional
from keras_contrib.layers import CRF

from utils.mapper_keras import get_optimizer, get_loss
from models.metrics import strict_accuracy



def get_layer(args, num_units):
    base_layer = GRU(units=num_units, activation=args.hidden_activation, dropout=args.dropout, recurrent_dropout=args.dropout, return_sequences=True)
    if args.model == 'lstm' or args.model == 'blstm':
        base_layer = LSTM(units=num_units, activation=args.hidden_activation, dropout=args.dropout, recurrent_dropout=args.dropout, return_sequences=True)
    if args.model == 'blstm' or args.model == 'bgru':
        base_layer = Bidirectional(base, merge_mode='concat')
    return base_layer


def get_model(args, max_len, num_words, emb_dim, num_tags, rnd_seed):
    # word input layer
    word_input = Input(shape=(max_len, ), dtype = 'int32')
    # word embedding layer
    model = Embedding(num_words, emb_dim, embeddings_initializer = glorot_normal(rnd_seed), input_length = max_len)(word_input)
    # batch normalization
    if args.batch_normalization:
        model = BatchNormalization(model)
    # bidirectional lstm layers
    num_units = args.model_size
    for _ in range(args.num_layers):
        layer = get_layer(args, num_units)
        model = layer(model)
        # batch normalization
        if args.batch_normalization:
            model = BatchNormalization(model)
        # halve hidden units for each new layer
        num_units = int(num_units / 2)
    # output layer
    if args.output_activation == 'softmax':
        tag_output = TimeDistributed(Dense(num_tags, activation=args.activation))(model)
    else:
        # add a dense layer followed by a crf layer
        model = TimeDistributed(Dense(args.model_size, activation=args.activation))(model)
        crf = CRF(num_tags)
        tag_output = crf(model)
    # define input and output
    model_input = [word_input, ]
    model_output = [tag_output, ]
    model = Model(input = model_input, output = model_output)
    return model


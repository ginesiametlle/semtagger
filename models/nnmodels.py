#!/usr/bin/python3
# this script defines the structure of possible neural models

from keras.models import Model, Input
from keras.layers import Dense, Conv1D, LSTM, GRU
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


def get_model(args, num_tags=0, max_wlen=0, num_words=0, wemb_dim=0, wemb_matrix=None, max_clen=0, num_chars=0, cemb_dim=0, cemb_matrix=None):
    """
    Obtains a neural model as a combination of layers
        Inputs:
            - args: command line arguments
            - num_tags: number of output tags
            - max_wlen: maximum number of words in a sentence
            - num_words: size of the word embedding vocabulary
            - wemb_dim: dimensionality of the word embedding vectors
            - wemb_matrix: word embedding matrix
            - max_clen: maximum number of characters in a sentence
            - num_chars: size of the character embedding vocabulary
            - cemb_dim: dimensionality of the character embedding vectors
            - cemb_matrix: character embedding matrix
        Returns:
            the compiled Keras neural model defined by the command line arguments
    """
    ## DEFINE NETWORK
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
        char_input = Input(shape=(max_clen,))
        # character embedding layer
        char_model = Embedding(input_dim=num_chars,
                               output_dim=cemb_dim,
                               weights = [cemb_matrix],
                               input_length = max_clen,
                               trainable = False)(char_input)

    # TODO: derive word features from character embeddings using a resnet
    # we employ temporal (1D) convolutions
	#https://blog.waya.ai/deep-residual-learning-9610bb62c355
 	# https://gist.github.com/mjdietzx/0cb95922aac14d446a6530f87b3a04ce

	# concat word and character features, if needed
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
        crf = CRF(num_tags)
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


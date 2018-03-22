#!/usr/bin/python3
# this script defines how input symbols are mapped to numerical values

import numpy as np
from keras.preprocessing.sequence import pad_sequences

def wordsents2sym(sents, max_len, word2idx, tag2idx, def_word, def_tag, pad_word, pad_tag):
    """
    Given a list of sentences, return a list of integers
        Inputs:
            - sents: list of sentences represented as a list of (word, tag) pairs
			- max_length: maximum allowed sentence length
            - word2idx: dictionary mapping words to indices in the embedding matrix
            - tag2idx: dictionary mapping tags to integers
            - def_word: special word to use by default
            - def_tag: special tag to use by default
            - pad_word: special word to use for padding
            - pad_tag: special tag to use for padding
		Outputs:
			- X: input feature vector
            - y: output feature vector (X * W = y)
    """
    X = []
    y = []
    nb_classes = len(tag2idx)

    try:
        print('[INFO] Reshaping word data...')
        # map each words to indices and tags to categorial vectors
        for sent in sents:
            sent_X = np.zeros((len(sent),), dtype = np.int32)
            sent_y = np.zeros((len(sent), nb_classes), dtype = np.int32)
            for i, elem in enumerate(sent):
                word = elem[0]
                tag = elem[1]
                if word in word2idx:
                    sent_X[i] = word2idx[word]
                else:
                    sent_X[i] = word2idx[def_word]
                if tag in tag2idx:
                    sent_y[i, tag2idx[tag]] = 1
                else:
                    sent_y[i, tag2idx[def_tag]] = 1
            X.append(sent_X)
            y.append(sent_y)

        # add padding up to the maximum allowed length
        X = pad_sequences(X, maxlen = max_len, dtype = np.int32,
                          padding="post", value = word2idx[pad_word])
        y = pad_sequences(y, maxlen = max_len, dtype = np.int32,
                          padding="post", value = tag2idx[pad_tag])
    except Exception as e:
        print('[ERROR] Exception in `sents2sym`:', e)

    print('[INFO] Number of classes (sem-tags):', nb_classes)
    print('[INFO] Input word data shape:', np.asarray(X).shape)
    print('[INFO] Output word data shape:', np.asarray(y).shape)

    return X, y, nb_classes

#!/usr/bin/python3
# this script defines how input symbols are mapped to numerical values

import numpy as np
from keras.preprocessing.sequence import pad_sequences


def wordsents2sym(sents, max_len, word2idx, tag2idx, def_word, def_tag, pad_word, pad_tag):
    """
    Given a list of sentences, return a list of integers
        Inputs:
            - sents: list of sentences represented as a list of (word, tag) pairs
			- max_len: maximum allowed sentence length
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
        # map each word to indices and tags to categorial vectors
        for sent in sents:
            sent_X = np.zeros((len(sent),), dtype = np.int32)
            sent_y = np.zeros((len(sent), nb_classes), dtype = np.int32)
            for i, elem in enumerate(sent):
                word = elem[0]
                if word in word2idx:
                    sent_X[i] = word2idx[word]
                else:
                    sent_X[i] = word2idx[def_word]

                if len(elem) > 1:
                    tag = elem[1]
                    if tag in tag2idx:
                        sent_y[i][tag2idx[tag]-1] = 1
                    else:
                        sent_y[i][tag2idx[def_tag]-1] = 1

            X.append(sent_X)
            y.append(sent_y)

        # add padding up to the maximum allowed length
        X = pad_sequences(X, maxlen = max_len, dtype = np.int32,
                          padding="post", value = word2idx[pad_word])

        pad_class = np.zeros((nb_classes,), dtype = np.int32)
        pad_class[tag2idx[pad_tag]-1] = 1
        y = pad_sequences(y, maxlen = max_len, dtype = np.int32,
                          padding="post", value = pad_class)

    except Exception as e:
        print('[ERROR] Exception in `wordsents2sym`:', e)

    print('[INFO] Number of classes (sem-tags):', nb_classes)
    print('[INFO] Input word data shape:', X.shape)
    print('[INFO] Output data shape:', y.shape)
    return X, y


def charsents2sym(sents, max_slen, max_wlen, char2idx, def_char, pad_char):
    """
    Given a list of sentences, return a list of integers
        Inputs:
            - sents: list of sentences represented as a list of characters
			- max_slen: maximum length of a sentence
            - max_wlen: maximum length of a word
            - char2idx: dictionary mapping words to indices in the embedding matrix
            - def_char: special word to use by default
            - pad_chars: special characters to use for padding
		Outputs:
			- X: input feature vector
    """
    X = []

    try:
        # map each character-decomposed word to indices
        for sent in sents:
            sent_X = []
            for word in sent:
                sent_X.append([])
                for char in word:
                    if char in char2idx:
                        sent_X[-1].append(char2idx[char])
                    else:
                        sent_X[-1].append(char2idx[def_char])
            # add padding up to the maximum allowed word length
            sent_pad = pad_sequences(sent_X, maxlen = max_wlen, dtype = np.int32,
                          padding="post", value = char2idx[pad_char['pad']])
            X.append(sent_pad)

        pad_word = pad_sequences([[char2idx[pad_char['begin']], char2idx[pad_char['end']]]], maxlen = max_wlen,
                                  padding="post", value=char2idx[pad_char['pad']])[0]

        X = pad_sequences(X, maxlen = max_slen, dtype = np.int32,
                        padding="post", value = pad_word)

    except Exception as e:
        print('[ERROR] Exception in `charsents2sym`:', e)

    print('[INFO] Input character data shape:', np.asarray(X).shape)
    return X


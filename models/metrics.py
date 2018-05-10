#!/usr/bin/python3
# this script implements evaluation metrics

from keras import backend as K


def strict_accuracy(act, pred):
    '''
    Calculate accuracy for each batch
    This metric does not factor padding symbols
    '''

    act_argm  = K.argmax(act, axis=-1)   # Indices of act. classes
    pred_argm = K.argmax(pred, axis=-1)  # Indices of pred. classes

    incorrect = K.cast(K.not_equal(act_argm, pred_argm), dtype='float32')
    correct   = K.cast(K.equal(act_argm, pred_argm), dtype='float32')
    padding   = K.cast(K.equal(K.sum(act), 0), dtype='float32')
    start     = K.cast(K.equal(act_argm, 0), dtype='float32')
    end       = K.cast(K.equal(act_argm, 1), dtype='float32')

    pad_start     = K.maximum(padding, start)
    pad_start_end = K.maximum(pad_start, end) # 1 where pad, start or end

    # Subtract pad_start_end from correct, then check equality to 1
    # E.g.: act: [pad, pad, pad, <s>, tag, tag, tag, </s>]
    #      pred: [pad, tag, pad, <s>, tag, tag, err, </s>]
    #   correct: [1,     0,   1,   1,   1,   1,   0,    1]
    #     p_s_e: [1,     1,   1,   1,,  0,   0,   0,    1]
    #  corr-pse: [0,    -1,   0,   0,   1,   1,   0,    0] # Subtraction
    # actu_corr: [0,     0,   0,   0,   1,   1,   0,    0] # Check equality to 1
    corr_preds   = K.sum(K.cast(K.equal(correct - pad_start_end, 1), dtype='float32'))
    incorr_preds = K.sum(K.cast(K.equal(incorrect - pad_start_end, 1), dtype='float32'))
    total = corr_preds + incorr_preds
    accuracy = corr_preds / total

    return accuracy


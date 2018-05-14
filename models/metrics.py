#!/usr/bin/python3
# this script implements evaluation metrics

import sys
import tensorflow as tf
from keras import backend as K


def strict_accuracy(act, pred):
    """
    Computes accuracy for each batch without factoring in padding symbols
    """
    # values of actual classes
    act_argm  = K.argmax(act, axis=-1)
    # values of predicted classes
    pred_argm = K.argmax(pred, axis=-1)
    # determines where the tags are incorrect (1) or not (0)
    incorrect = K.cast(K.not_equal(act_argm, pred_argm), dtype='float32')
    # determines where the tags are correct (1) or not (0)
    correct   = K.cast(K.equal(act_argm, pred_argm), dtype='float32')
    # determines where the tag is a padding tag (1) or not (0)
    padding   = K.cast(K.equal(K.sum(act), 0), dtype='float32')
    # subtract padding from correct predictions and check equality to 1
    corr_preds = K.sum(K.cast(K.equal(correct - padding, 1), dtype='float32'))
    incorr_preds = K.sum(K.cast(K.equal(incorrect - padding, 1), dtype='float32'))
    total = corr_preds + incorr_preds
    # actual accuracy without padding
    accuracy = corr_preds / total
    return accuracy


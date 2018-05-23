#!/usr/bin/python3
# this script implements evaluation metrics

import tensorflow as tf
from keras import backend as K


def strict_accuracy_K(act, pred):
    """
    Keras metric that computes the accuracy of tagged sentences for each batch
    Predictions with a categorical vector of [1 0 0 ... 0] are not factored in
        Inputs:
            - act: array of actual categorical vectors
            - pred: array of predicted categorical vectors
        Outputs:
            - accuracy score
    """
    # numerical values of the actual classes
    act_argm = K.argmax(act, axis=-1)
    # numerical values of the predicted classes
    pred_argm = K.argmax(pred, axis=-1)
    # determines where the classes are incorrect or not
    incorrect = K.cast(K.not_equal(act_argm, pred_argm), dtype='float32')
    # determines where the classes are correct or not
    correct = K.cast(K.equal(act_argm, pred_argm), dtype='float32')
    # determines where the classes are ignored or not
    padding = K.cast(K.equal(act_argm, 0), dtype='float32')
    # subtract padding from correct predictions and check equality to 1
    corr_preds = K.sum(K.cast(K.equal(correct - padding, 1), dtype='float32'))
    incorr_preds = K.sum(K.cast(K.equal(incorrect - padding, 1), dtype='float32'))
    total_preds = corr_preds + incorr_preds
    # actual accuracy without padding
    accuracy = corr_preds / total_preds
    return accuracy


def strict_accuracy_N(act, pred, ignore_class=0):
    """
    Computes the accuracy of an array of tagged sentences
    Actual values which match `ignore_class` are not factored in
        Inputs:
            - act: array of actual numerical vectors
            - pred: array of predicted numerical vectors
            - ignore_class: numerical value to be ignored
        Outputs:
            - accuracy score
    """
    # number of correct predictions
    corr_preds = 0
    # number of predictions
    total_preds = 0
    # compute values via iterating over sentences
    for sent in zip(act, pred):
        act_classes = sent[0]
        pred_classes = sent[1]
        for t in range(len(act_classes)):
            if act_classes[t] != ignore_class:
                total_preds += 1
                if pred_classes[t] == act_classes[t]:
                    corr_preds += 1
    # actual accuracy without padding
    return corr_preds / total_preds


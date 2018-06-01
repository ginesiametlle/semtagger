#!/usr/bin/python3
# this parses configuration arguments

import sys
import argparse


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--raw_pmb_data', help='File containing sem-tagged sentences from the PMB', type=str, default='')
    parser.add_argument('--raw_extra_data', help='File containing sem-tagged sentences from an extra source', type=str, default='')
    parser.add_argument('--output_words', help='File containing processed word sem-tagged sentences', type=str, default='')
    parser.add_argument('--output_chars', help='File containing processed character sem-tagged sentences', type=str, default='')
    parser.add_argument('--word_embeddings', help='File containing pretrained word embeddings', type=str, default='')
    parser.add_argument('--char_embeddings', help='File containing pretrained character embeddings', type=str, default='')
    parser.add_argument('--word_embeddings_trainable', help='Train word embedding weights', type=int, default=0)
    parser.add_argument('--char_embeddings_trainable', help='Train character embedding weights', type=int, default=0)
    parser.add_argument('--use_words', help='Use word features', type=int, default=1)
    parser.add_argument('--use_chars', help='Use character features', type=int, default=0)
    parser.add_argument('--output_model', help='Output model file', type=str, default='')
    parser.add_argument('--output_model_info', help='Output model parameters file', type=str, default='')
    parser.add_argument('--lang', help='Language code as in ISO 639-1', type=str, default='en')
    parser.add_argument('--model', help='Type of neural model', type=str, default='bgru')
    parser.add_argument('--epochs', help='Number of training epochs', type=int, default=25)
    parser.add_argument('--model_size', help='Number of units on the first layer', type=int, default=150)
    parser.add_argument('--num_layers', help='Number of layers', type=int, default=2)
    parser.add_argument('--noise_sigma', help='Parameter to control introduced noise', type=float, default=0.0)
    parser.add_argument('--hidden_activation', help='Activation function for the hidden units', type=str, default='relu')
    parser.add_argument('--output_activation', help='Activation function for the output unit', type=str, default='crf')
    parser.add_argument('--loss', help='Loss function', type=str, default='categorical_cross_entropy')
    parser.add_argument('--optimizer', help='Optimization algorithm to use', type=str, default='nadam')
    parser.add_argument('--dropout', help='Dropout rate to use per layer', type=float, default=0.25)
    parser.add_argument('--batch_size', help='Batch size', type=int, default=150)
    parser.add_argument('--batch_normalization', help='Apply batch normalization to recurrent layers', type=int, default=1)
    parser.add_argument('--verbose', help='Keras verbosity mode', type=int, default=1)
    parser.add_argument('--test_size', help='Proportion of the sentences to use as a test set', type=float, default=0.0)
    parser.add_argument('--dev_size', help='Proportion of the sentences to use as a development set', type=float, default=0.0)
    parser.add_argument('--grid_search', help='Optimize hyper-parameters using cross-validation', type=int, default=0)
    parser.add_argument('--sent_len_perc', help='Maximum length of a sentence as a percentile in word-based features', type=float, default=0.90)
    parser.add_argument('--word_len_perc', help='Maximum length of a word as a percentile in character-based features', type=float, default=0.98)
    parser.add_argument('--multi_word', help='Handle multi-word expressions', type=int, default=1)
    parser.add_argument('--resnet_depth', help='Depth of the residual network applied on character vectors', type=int, default=8)
    parser.add_argument('--input_pred_file', help='Unlabelled data to predict using a model', type=str, default='')
    parser.add_argument('--output_pred_file', help='Prediction output', type=str, default='')
    return parser.parse_known_args()[0]


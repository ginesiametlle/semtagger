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
    parser.add_argument('--use_words', help='Use word features', type=int, default=1)
    parser.add_argument('--use_chars', help='Use character features', type=int, default=0)
    parser.add_argument('--output_model', help='Output directory for mappings and the model file', type=str, required=True)
    parser.add_argument('--lang', help='Language code as in ISO 639-1', type=str, default='en')
    parser.add_argument('--model', help='Type of neural model', type=str, default='bgru')
    parser.add_argument('--epochs', help='Number of training epochs', type=int, default=10)
    parser.add_argument('--model_size', help='Number of units on the first layer', type=int, default=50)
    parser.add_argument('--num_layers', help='Number of layers', type=int, default=2)
    parser.add_argument('--noise_sigma', help='Parameter to control introduced noise', type=float, default=0.1)
    parser.add_argument('--hidden_activation', help='Activation function for the hidden units', type=str, default='relu')
    parser.add_argument('--output_activation', help='Activation function for the output unit', type=str, default='crf')
    parser.add_argument('--loss', help='Loss function', type=str, default='mse')
    parser.add_argument('--optimizer', help='Optimization algorithm to use', type=str, default='adam')
    parser.add_argument('--dropout', help='Dropout rate to use per layer', type=float, default=0.1)
    parser.add_argument('--batch_size', help='Batch size', type=int, default=32)
    parser.add_argument('--batch_normalization', help='Apply batch normalization to recurrent layers', type=int, default=1)
    parser.add_argument('--verbose', help='Keras verbosity mode', type=int, default=1)
    parser.add_argument('--test_size', help='Proportion of the sentences to use as a test set', type=float, default=0.0)
    parser.add_argument('--dev_size', help='Proportion of the sentences to use as a development set', type=float, default=0.0)
    parser.add_argument('--grid_search', help='Estimate hyperparameters using cross-validation', type=int, default=0)
    parser.add_argument('--max_len_perc', help='Maximum length of a word sequence as a percentile', type=float, default=0.9)
    parser.add_argument('--multi_word', help='Handle multi-word expressions', type=int, default=1)
    return parser.parse_known_args()[0]


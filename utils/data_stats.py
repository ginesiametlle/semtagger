#!/usr/bin/python3
# this script produces graphical visualizations

import sys
import os

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use('agg')
from matplotlib import pyplot as plt

import pygal
import cairosvg
import seaborn as sns

import operator
import itertools
from functools import reduce

from sklearn.metrics import confusion_matrix


def plot_dist_tags(sents, vocab, outimg, outfile, padwords=[]):
    """
    Plot the amount of words per tag in a dataset
            Inputs:
                - sents: list of sentences represented as a list of (word, tag, sym) items
                - words: set containing words in the vocabulary
                - outimg: output image
                - outfile: output text file
                - padwords: set of words to be ignored
    """
    count = {}
    for sent in sents:
        if sent:
            for word, tag, sym in sent:
                if word not in padwords:
                    if tag not in count:
                        count[tag] = [0,0]
                    count[tag][0] += 1
                    if sym not in vocab:
                        count[tag][1] += 1

    xdata = sorted(count.keys(), key=lambda x: count[x][0], reverse=True)
    ydata_oov = list([count[x][1] for x in xdata])
    ydata_rest = list([count[x][0] - count[x][1] for x in xdata])

    line_chart = pygal.StackedBar(width=1600, height=800, x_label_rotation=-45, y_title='Number of words')
    line_chart.x_labels = xdata
    line_chart.add('OOVs', ydata_oov)
    line_chart.add('IVs + OOVs', ydata_rest)

    # circumvent potential svg styling problems
    line_chart.render_to_file(outimg)
    cairosvg.svg2svg(url=outimg, write_to=outimg)

    # output text
    with open(outfile, 'w') as tfile:
        tfile.write("tag\t#OOVs\t#IVs\t#words\tratio\n")
        for i in range(len(xdata)):
            tfile.write(str(xdata[i]) + '\t')
            tfile.write(str(ydata_oov[i]) + '\t')
            tfile.write(str(ydata_rest[i]) + '\t')
            tfile.write(str(ydata_oov[i] + ydata_rest[i]) + '\t')
            tfile.write(str(ydata_oov[i] / (ydata_oov[i] + ydata_rest[i])) + '\n')


def plot_accuracy(history, keys, labels, test_acc, outfile):
    """
    Plot the obtained accuracy scores against training epochs
            Inputs:
                - history: object obtained from calling Keras fit() function
                - keys: key values to access the metrics in history
                - labels: names of the metrics that match keys
                - test_acc: accuracy obtained on the test set (constant)
                - outfile: output file
    """
    hist = pd.DataFrame(history.history)

    chart = pygal.Line(width=1600, height=800, x_label_rotation=0, x_title='Number of training epochs', y_title='Sem-tagging accuracy')
    xdata = [x+1 for x in range(len(hist[keys[0]]))]
    chart.x_labels = xdata

    # plot all metrics
    for i in range(len(keys)):
        key = keys[i]
        label = labels[i]
        chart.add(label, hist[key], show_dots=False, stroke_style={'width': 6, 'dasharray': '3, 8', 'linecap': 'round', 'linejoin': 'round'})

    # plot a horizontal line representing accuracy on the test set
    ytest=[test_acc] * len(xdata)
    chart.add(None, ytest, show_dots=False, stroke_style={'width': 2})

    chart.render_to_file(outfile)
    cairosvg.svg2svg(url=outfile, write_to=outfile)


def plot_confusion_matrix(predicted, true, lengths, classes, outfile, ymap, vocab=[], normalize=True):
    """
    Plot a confusion matrix
            Inputs:
                - predicted: predicted labels
                - true: true labels
                - lengths: original length of each sentence (excluding padding characters)
                - classes: set containing all output classes
                - outfile: output file
                - ymap: map to transform predicted and true labels
                - vocab: set containing all words in the vocabulary
                - normalize: normalize confusion matrix
    """
	# turn the obtained labels to readable string classes (exclude pads)
    y_pred = predicted
    y_true = true

    if ymap:
        y_pred = []
        y_true = []
        for i in range(len(true)):
            y_pred += [ymap[yi] for yi in predicted[i][:lengths[i]]]
            y_true += [ymap[yi] for yi in true[i][:lengths[i]]]

    # compute confusion matrix
    cm = confusion_matrix(y_true, y_pred, labels=classes)
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("[INFO] Normalized confusion matrix")
    else:
        print('[INFO] Confusion matrix without normalization')
    cm = pd.DataFrame(cm, index=classes, columns=classes)

    # transform confusion matrix to a heatmap and output
    fig, ax = plt.subplots(figsize=(25,25))
    b = sns.heatmap(cm, fmt='', ax=ax, cmap="BuPu", square=True, xticklabels=True, yticklabels=True)
    b.set_xlabel("Predicted tags", fontsize=20)
    b.set_ylabel("True tags", fontsize=20)
    b.set_xticklabels(b.get_yticklabels(), fontsize=12)
    b.set_yticklabels(b.get_yticklabels(), fontsize=12)
    plt.draw()
    plt.savefig(outfile)


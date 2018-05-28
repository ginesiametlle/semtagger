#!/usr/bin/python3
# this script produces graphical visualizations

import sys
import os

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use('agg')
from matplotlib import pyplot as plt

from collections import Counter

import pygal
import cairosvg
import seaborn as sns

from sklearn.metrics import confusion_matrix


def plot_dist_tags(sents, vocab, outimg, outfile, padwords=[]):
    """
    Plot the amount of IV and OOV words per tag in a dataset
            Inputs:
                - sents: list of sentences represented as a list of (word, tag, sym) items
                - words: set containing words in the vocabulary
                - outimg: output image
                - outfile: output text file
                - padwords: set of words to be ignored
    """
    # compute the number of IV and OOV words per tag
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

    # output in svg format
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

    # output in text format
    with open(outfile, 'w') as tfile:
        tfile.write("tag\t#OOVs\t#IVs\t#words\tratio\n")
        for i in range(len(xdata)):
            tfile.write(str(xdata[i]) + '\t')
            tfile.write(str(ydata_oov[i]) + '\t')
            tfile.write(str(ydata_rest[i]) + '\t')
            tfile.write(str(ydata_oov[i] + ydata_rest[i]) + '\t')
            tfile.write(str(ydata_oov[i] / (ydata_oov[i] + ydata_rest[i])) + '\n')


def plot_dist_lengths(lengths, step, limit, outimg):
    """
    Plot a length distribution over sentences
            Inputs:
                - lengths: list of numerical lenghts
                - limit: maximum allowed length
                - step: step in the horizontal axis
                - outimg: output image
    """
    # count number of occurrences for each length value
    c = Counter(lengths)
    max_val = max(c.keys())

    # output in svg format
    xdata = list(range(1, max_val + step + 1))
    ydata_used = [c[k] if k in c and k <= limit else 0 for k in xdata]
    ydata_unused = [c[k] if k in c and k > limit else 0 for k in xdata]

    line_chart = pygal.StackedBar(width=1600, height=800, x_label_rotation=-45,
                                  x_title = 'Number of words', y_title = 'Number of sentences',
                                  show_minor_x_labels=False)
    line_chart.x_labels = xdata
    line_chart.x_labels_major = list(set([x // step * step for x in xdata]))
    line_chart.add('Discarded data', ydata_unused)
    line_chart.add('Used data', ydata_used)

    # circumvent potential svg styling problems
    line_chart.render_to_file(outimg)
    cairosvg.svg2svg(url=outimg, write_to=outimg)


def plot_accuracy(history, keys, labels, test_acc, outimg, outfile):
    """
    Plot the obtained accuracy scores against training epochs
            Inputs:
                - history: object obtained from calling Keras fit() function
                - keys: key values to access the metrics in history
                - labels: names of the metrics which match `keys`
                - test_acc: accuracy obtained on the test set
                - outimg: output image
                - outfile: output text file
    """
    # build chart
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

    # output in svg format
    chart.render_to_file(outimg)
    cairosvg.svg2svg(url=outimg, write_to=outimg)

    # output in text format
    with open(outfile, 'w') as tfile:
        header = str(keys[0]) + ''.join(["\t" + str(key) for key in keys[1:]])
        tfile.write(header + '\n')
        for i in range(len(hist[keys[0]])):
            tfile.write(str(hist[keys[0]][i]))
            for key in keys[1:]:
                 tfile.write("\t" + str(hist[key][i]))
            tfile.write("\n")


def plot_confusion_matrix(act, pred, classes, ignore_class, ymap, outfile, vocab=[], normalize=True):
    """
    Plot a confusion matrix
            Inputs:
                - act: array of actual numerical vectors
                - pred: array of predicted numerical vectors
                - classes: set containing all output classes to consider
                - ignore_class: numerical value to be ignored
                - outfile: output file
                - ymap: map to class numerical values
                - vocab: set containing all words in the vocabulary
                - normalize: normalize confusion matrix
    """
	# turn class numerical values into readable strings using the mapping provided
    y_true = act
    y_pred = pred
    if ymap:
        y_pred = []
        y_true = []
        for i in range(len(act)):
            for j in range(len(act[i])):
                if act[i][j] != ignore_class:
                    y_pred += [ymap[pred[i][j]]]
                    y_true += [ymap[act[i][j]]]

    # compute confusion matrix
    cm = confusion_matrix(y_true, y_pred, labels=classes)
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("[INFO] Normalized confusion matrix")
    else:
        print('[INFO] Confusion matrix without normalization')
    cm = pd.DataFrame(cm, index=classes, columns=classes)

    # transform confusion matrix to a heatmap and output
    fig, ax = plt.subplots(figsize=(25, 25))
    b = sns.heatmap(cm, fmt='', ax=ax, cmap="BuPu", square=True, xticklabels=True, yticklabels=True)
    b.set_xlabel("Predicted sem-tags", fontsize=20)
    b.set_ylabel("Actual sem-tags", fontsize=20)
    b.set_xticklabels(b.get_yticklabels(), fontsize=12)
    b.set_yticklabels(b.get_yticklabels(), fontsize=12)
    plt.draw()
    plt.savefig(outfile)


import codecs
import numpy as np
import sys

from loader import read_conll_file, load_embeddings_file
from models import lstm, lstm_crf, bi_lstm, bi_lstm_crf
from metrics import *

# this here will evaluate a model that was trained and saved

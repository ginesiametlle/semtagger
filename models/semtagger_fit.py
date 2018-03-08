import codecs
import numpy as np
import sys

from loader import read_conll_file, load_embeddings_file
from models import lstm, lstm_crf, bi_lstm, bi_lstm_crf


# here we will load the embeddings, train and perform cross validaiton, plus saving the model 

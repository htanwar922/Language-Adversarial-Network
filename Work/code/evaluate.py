#!/usr/bin/env ipython

#@title evaluate.py

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models, optimizers, preprocessing, losses
from tensorflow.keras import backend as K
from tensorflow.data import Dataset

import numpy as np
import os, io
from pathlib import Path
os.chdir(os.path.dirname(__file__))

from options import *
from vocab import *
from data import *
from utils import *
from layers import *
from models import *

# Save/Load model and weights
E, A, F, P, Q, EA, EAF, EAFP, EAFQ, LAN = load_models(current_ckpt, __version__=opt.__version__)

"""# Evaluate trained LAN and other models"""

# Overall LAN results on unseen target data
for (inputs_src, lengths_src, labels_src), (inputs_tgt, lengths_tgt, labels_tgt) in zip(test_src, test_tgt):
    log.info(f" LAN test : ")
    inputs = tf.concat([inputs_src, inputs_tgt], axis=0)
    lengths = tf.concat([lengths_src, lengths_tgt], axis=0)
    labels = tf.concat([labels_src, labels_tgt], axis=0)
    lang_labels_src = tf.broadcast_to([1], shape=labels_src.shape)
    lang_labels_tgt = tf.broadcast_to([-1], shape=labels_tgt.shape)
    lang_labels = tf.concat([lang_labels_src, lang_labels_tgt], axis=0)
    print(LAN.evaluate(x=[inputs_src, lengths_src], y=[labels_src, lang_labels_src], epochs=1))
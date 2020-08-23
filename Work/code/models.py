#!/usr/bin/env ipython

#@title models.py

"""## Custom models (CKPT0)
- ### E - Embedding layer (from vocab).
- ### A - Averaging layer (defined above).
- ### EA - Model of above layers with reviews and review-lengths as inputs (keras.Input()).
- ### F - DAN Feature Extractor (extract features from above EA model).
- ### EAF - Overall DAN Feature Extractor (takes inputs as reviews and review lengths and gives features as outputs).
- ### P - Semantic classifier over top of EAF model.
- ### EAFP - Overall Semantic classifier (takes inputs as reviews and review lengths and gives softmax star-rating labels as outputs). Loss is taken to be sparse_categorical_crossentropy loss.
- ### Q - Language Detector over top of EAF model.
- ### EAFQ - Overall Language detector (takes inputs as reviews and review lengths and gives language score as output). Extracts language identification features without star-rating labels for adversarial training of F and EAF models. Loss is taken to be hinge loss.
- ### LAN - Overall model with two branches namely EAFP and EAFQ wih shared EAF base and different tops P, Q.
"""

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
from layers import *

def absexp_1(x):
    return tf.clip_by_value(tf.math.expm1(tf.cast(tf.abs(x), dtype=tf.float64)) * tf.sign(x), -1.7e+308, 1.7e+308)

def scce(y_true, y_pred):
    return losses.SparseCategoricalCrossentropy()(y_true, y_pred)

def hinge(ll_lang, ll_pred):
    return losses.Hinge()(ll_true, ll_pred)

def total_loss(y_ll_true, y_ll_pred):
    y_true, ll_true = zip(*y_ll_true)
    y_pred, ll_pred = zip(*y_ll_pred)
    return scce(y_true, y_pred) + opt._lambda * hinge(ll_true, ll_pred)

class lstm_EA(keras.Model):
    def __init__(self, E, A):
        super(lstm_EA, self).__init__()
        self.E = E
        self.A = A

    def call(self, inputs):
        x = self.E(inputs)
        return self.A(x)

log.info(f' Current ckpt : {current_ckpt}')
if opt.last_ckpt == current_ckpt:
    log.info(f' Running ckpt {current_ckpt}')
    TRAIN = True
    opt.num_labels = 6
    opt._lambda = 1.
    opt.F_layers = 2
    opt.P_layers = 2
    opt.Q_layers = 2
    opt.F_activation = 'relu'   # absexp_1
    opt.P_activation = 'relu'
    opt.Q_activation = 'relu'
    #opt.model = 'dan' #'lstm'
    opt.lstm_hidden = 256

    num_layers, hidden_size, dropout, batch_norm, activation = opt.F_layers, opt.hidden_size, opt.dropout, opt.F_bn, opt.F_activation
    F = keras.Sequential()
    for i in range(num_layers):
        if dropout > 0: F.add(layers.Dropout(rate=dropout, name=f'Dropout_{i}'))
        if i == 0:
            if opt.model == 'dan': F.add(layers.Dense(units=hidden_size, input_shape=(vocab.emb_size,), activation=activation, name=f'DenseAbsExp_{i}'))
            if opt.model == 'lstm': F.add(layers.Dense(units=hidden_size, input_shape=(opt.lstm_hidden*2,), activation=activation, name=f'DenseAbsExp_{i}'))
        else: F.add(layers.Dense(units=hidden_size, input_shape=(hidden_size,), activation=activation, name=f'DenseAbsExp_{i}'))
        if batch_norm: F.add(layers.BatchNormalization(input_shape=(hidden_size,), name=f'BatchNorm_{i}'))    # same shape as input    # use training=False when making inference from model (model.predict, model.evaluate?)
    #F.add(layers.LeakyReLU(alpha=0.3))
    #F.add(layers.ReLU())
    F.add(layers.Dense(units=hidden_size, input_shape=(hidden_size,), activation=activation, name=f'DenseAbsExpFinal_{i}'))

    num_layers, hidden_size, output_size, dropout, batch_norm, activation = opt.P_layers, opt.hidden_size, opt.num_labels, opt.dropout, opt.P_bn, opt.P_activation
    P = models.Sequential()
    P.add(keras.Input((opt.hidden_size,)))
    for i in range(num_layers):
        if dropout > 0: P.add(layers.Dropout(rate=dropout))
        P.add(layers.Dense(units=hidden_size, input_shape=(hidden_size,), activation=opt.activation, name=f'DenseAbsExp_{i}'))
        if batch_norm: P.add(layers.BatchNormalization())
        #P.add(layers.ReLU())
    #P.add(layers.Dense(units=output_size, input_shape=(hidden_size,), activation='tanh'))
    P.add(layers.Dense(units=output_size, input_shape=(hidden_size,), activation='softmax', name=f'DenseSoftmax_{i}'))

    num_layers, hidden_size, dropout, batch_norm, activation = opt.Q_layers, opt.hidden_size, opt.dropout, opt.Q_bn, opt.Q_activation
    Q = keras.Sequential()
    for i in range(num_layers):
        if dropout > 0: Q.add(layers.Dropout(rate=dropout, name=f'Dropout_{i}'))
        Q.add(layers.Dense(units=hidden_size, input_shape=(hidden_size,), activation=activation, name=f'DenseAbsExp_{i}'))
        if batch_norm: Q.add(layers.BatchNormalization(input_shape=(hidden_size,), name=f'BathcNorm_{i}'))
    Q.add(layers.Dense(units=hidden_size, input_shape=(hidden_size,), activation=activation, name=f'DenseAbsExpFinal_{i}'))
    Q.add(layers.Dense(units=1, input_shape=(hidden_size,), activation='tanh', name=f'DenseTanh'))
    #Q.add(layers.Softmax())

    E = vocab.init_embed_layer()
    if opt.model == 'dan': A = Averaging(toks=[vocab.unk_idx, vocab.bos_idx, vocab.eos_idx], vector_length=opt.vector_length)
    if opt.model == 'lstm': A = layers.Bidirectional(layers.LSTM(opt.lstm_hidden))

    shape = (opt.max_seq_len,)
    inputs, lengths = keras.Input(shape), keras.Input(())

    if opt.model == 'dan': embeddings = E(inputs); outputs_EA = A(embeddings, lengths); EA = keras.Model(inputs=[inputs, lengths], outputs=outputs_EA)
    if opt.model == 'lstm': EA = lstm_EA(E, A); outputs_EA = EA(inputs)
    
    outputs_EAF = F(outputs_EA)
    EAF = keras.Model(inputs=[inputs, lengths], outputs=outputs_EAF, name="FeatureExtractor_AE")

    outputs_EAFP = P(outputs_EAF)
    EAFP = keras.Model(inputs=[inputs, lengths], outputs=outputs_EAFP, name="SemanticClassifier_FAE")

    outputs_EAFQ = Q(outputs_EAF)
    EAFQ = keras.Model(inputs=[inputs, lengths], outputs=outputs_EAFQ, name="LanguageDetector_FAE")

    LAN = keras.Model(inputs=[inputs, lengths], outputs=[outputs_EAFP, outputs_EAFQ], name="LAN")
    save_models(current_ckpt, __version__=opt.__version__)
else:
    log.info(f' Skipping to ckpt {opt.last_ckpt}')

print(LAN.summary())
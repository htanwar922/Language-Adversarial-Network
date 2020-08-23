#!/usr/bin/env ipython

#@title train.py

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

"""## IF CRASHED previously:
<br> <i>Run till here and jump to next MARKUP Checkpoint</i>
"""

# Ckpt0
current_ckpt = 0
TRAIN = current_ckpt == opt.last_ckpt
print(f'TRAIN FRESH : {TRAIN}')

"""## Checking if all models are defined"""

# Train setup
log.info(f' Current ckpt : {current_ckpt}')
if opt.last_ckpt == current_ckpt:
    log.info(f' Running ckpt {current_ckpt}')
    log.info('Checking outputs and initializing...')
    E.trainable=False
    EA.trainable=False
    for (inputs, lengths, labels) in train_src.take(1).take(1):
        print(inputs)
        x = LAN([inputs[:25], lengths[:25]])
        print((x))

"""## Set training rates and other train statistics"""

# Training statistics : learning_rates
log.info(f' Current ckpt : {current_ckpt}')
if opt.last_ckpt == current_ckpt:
    log.info(f' Running ckpt {current_ckpt}')
    opt.learning_rate, opt.Q_learning_rate = 1e-2, 1e-3  # 1e-3 is default
    opt.learning_rate, opt.Q_learning_rate

"""# Train inner models (EAFP, EAFQ) without/with embeddings-training

## Training without training the embeddings
"""

# Setting the embeddings non-trainable
log.info(f' Current ckpt : {current_ckpt}')
if opt.last_ckpt == current_ckpt:
    log.info(f' Running ckpt {current_ckpt}')
    E.trainable = False
    EA.trainable = False

# EA layers
log.info(f' Current ckpt : {current_ckpt}')
if opt.last_ckpt == current_ckpt:
    log.info(f' Running ckpt {current_ckpt}')
    print("Trainable EA layers : ")
    print(*[x.name + '\n' for x in EA.trainable_variables])

"""### Training F and P on labeled source data"""

# Trainable layers for EAFP training with fixed embeddings
log.info(f' Current ckpt : {current_ckpt}')
if opt.last_ckpt == current_ckpt:
    log.info(f' Running ckpt {current_ckpt}')
    print("Trainable EAFP layers : ")
    print(*[x.name + '\n' for x in EAFP.trainable_variables])

# Training F and P : sparse categorical
if opt.last_ckpt == current_ckpt:
    log.info('Training EAFP model with src data with fixed embeddings...')
    TRAIN = True
    EAFP.compile(optimizer=optimizers.Adam(opt.learning_rate), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    epochs = 20
    for epoch in trange(epochs):
        batch_no = 0
        for (inputs, lengths, labels) in train_src:
            log.info(f"Training on batch no : {batch_no}"); batch_no += 1
            print(scce(labels, EAFP.predict([inputs, lengths])))
            history = EAFP.fit(x=[inputs, lengths], y=labels, epochs=1)
else:
    log.info(f' Skipping to ckpt {opt.last_ckpt}')

"""### Evaluation of Sentiment Classifier"""

# P results : Unseen target data loss and  accuracy
log.info(f' Current ckpt : {current_ckpt}')
if opt.last_ckpt == current_ckpt:
    log.info(f' Running ckpt {current_ckpt}')
    for inputs, lengths, labels in train_src:
        print("SRC test : \n", EAFP.evaluate(x=[inputs, lengths], y=labels))
    for inputs, lengths, labels in train_tgt:
        print("TGT test : \n", EAFP.evaluate(x=[inputs, lengths], y=labels))

"""### Training F and Q on language-labeled source-target data"""

# Trainable layers for EAFQ training with fixed embeddings
log.info(f' Current ckpt : {current_ckpt}')
if opt.last_ckpt == current_ckpt:
    log.info(f' Running ckpt {current_ckpt}')
    [x.name for x in EAFQ.trainable_variables]

# train.py : Training F and Q : adversarial : hinge loss
log.info(f' Current ckpt : {current_ckpt}')
if opt.last_ckpt == current_ckpt:
    log.info(f' Running ckpt {current_ckpt}')
    log.info('Training EAFQ model with src and tgt data with lang_labels and fixed embeddings...')
    TRAIN = True
    #opt.Q_learning_rate = 1e-4
    EAFQ.compile(optimizer=optimizers.Adam(opt.Q_learning_rate), loss='hinge', metrics=['accuracy'])
    opt.Q_iterations = 5
    for epoch in trange(opt.Q_iterations):
        batch_no = 0
        for (inputs_src, lengths_src, labels_src), (inputs_tgt, lengths_tgt, labels_tgt) in zip(train_src, train_tgt):
            log.info(f"Training on batch no : {batch_no}"); batch_no += 1
            inputs = tf.concat([inputs_src, inputs_tgt], axis=0)
            lengths = tf.concat([lengths_src, lengths_tgt], axis=0)
            lang_labels_src = tf.broadcast_to([1], shape=labels_src.shape)
            lang_labels_tgt = tf.broadcast_to([-1], shape=labels_tgt.shape)
            lang_labels = tf.concat([lang_labels_src, lang_labels_tgt], axis=0)
            #print(lang_labels)
            history = EAFQ.fit(x=[inputs, lengths], y=lang_labels, epochs=1)
else:
    log.info(f' Skipping to ckpt {opt.last_ckpt}')

"""### Evaluation of Sentiment Classifier and language detector"""

# P results : Unseen target data loss and  accuracy after F-Q training
log.info(f' Current ckpt : {current_ckpt}')
if opt.last_ckpt == current_ckpt:
    log.info(f' Running ckpt {current_ckpt}')
    EAFP.evaluate([inputs_tgt, lengths_tgt], labels_tgt, verbose=2)

# Q results
log.info(f' Current ckpt : {current_ckpt}')
if opt.last_ckpt == current_ckpt:
    log.info(f' Running ckpt {current_ckpt}')
    EAFQ.compile(optimizer=optimizers.Adam(opt.Q_learning_rate), loss='hinge', metrics=['accuracy', keras.metrics.Hinge()])
    print(EAFQ.predict([inputs_src, lengths_src]), '\n', EAFQ.evaluate([inputs_src, lengths_src], lang_labels_src, verbose=2), '\n', EAFQ.predict([inputs_tgt, lengths_tgt]), '\n', EAFQ.evaluate([inputs_tgt, lengths_tgt], lang_labels_tgt, verbose=2))

"""## Save the trained models before embedding training (CKPT1)
<br>Continue by loading from this saved checkpoint.
"""

# Save/Load model and weights
current_ckpt += 1
log.info(f' Current ckpt : {current_ckpt}')
if opt.last_ckpt == current_ckpt:
    log.info(f' Running ckpt {current_ckpt}')
    E, A, F, P, Q, EA, EAF, EAFP, EAFQ, LAN = load_models(current_ckpt, __version__=opt.__version__)
elif opt.last_ckpt < current_ckpt:
    log.info("Done/Overstepped last_ckpt before...")
    #input("Interrupt execution to not save else enter anything : ")
    save_models(current_ckpt, __version__=opt.__version__)
    with open(opt.crash_logs, 'w') as foo: foo.write(str(current_ckpt))
    if last_ckpt != -1 and last_ckpt < current_ckpt:
else:
    log.info(f' Skipping to ckpt {opt.last_ckpt}')

"""## Training with trainable embeddings
<br> <i>Each step will take time so keep saving the models and don't change tab for long either</i>
"""

# Setting the embeddings trainable
log.info(f' Current ckpt : {current_ckpt}')
if opt.last_ckpt == current_ckpt:
    log.info(f' Running ckpt {current_ckpt}')
    E.trainable = True
    EA.trainable = True

"""### Training Sentiment Classifier (EAFP) on labeled source data"""

# Trainable layers for EAFP training with trainable embeddings
log.info(f' Current ckpt : {current_ckpt}')
if opt.last_ckpt == current_ckpt:
    log.info(f' Running ckpt {current_ckpt}')
    [x.name for x in EAFP.trainable_variables]

# train.py : Training Embeddings, F and P : sparse categorical
log.info(f' Current ckpt : {current_ckpt}')
if opt.last_ckpt == current_ckpt:
    log.info(f' Running ckpt {current_ckpt}')
    log.info('Training EAFP model with src data with fixed embeddings...')
    TRAIN = True
    EAFP.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    epochs = 1
    for epoch in trange(epochs):
        batch_no = 0
        for (inputs, lengths, labels) in train_src:
            log.info(f" Training on batch no : {batch_no}"); batch_no += 1
            print(scce(labels, EAFP.predict([inputs, lengths])))
            history = EAFP.fit(x=[inputs, lengths], y=labels, epochs=1)
else:
    log.info(f' Skipping to ckpt {opt.last_ckpt}')

"""## Save the trained models after embedding training of EAFP (CKPT2)
<br>Continue by loading from this saved checkpoint.
"""

# Save/Load model and weights
current_ckpt += 1
log.info(f' Reached ckpt {current_ckpt}')
log.info(f' Current ckpt : {current_ckpt}')
if opt.last_ckpt == current_ckpt:
    log.info(f' Running ckpt {current_ckpt}')
    E, A, F, P, Q, EA, EAF, EAFP, EAFQ, LAN = load_models(current_ckpt, __version__=opt.__version__)
elif opt.last_ckpt < current_ckpt:
    log.info("Done/Overstepped last_ckpt before...")
    #input("Interrupt execution to not save else enter anything : ")
    save_models(current_ckpt, __version__=opt.__version__)
    with open(opt.crash_logs, 'w') as foo: foo.write(str(current_ckpt))
    if last_ckpt != -1 and last_ckpt < current_ckpt:
else:
    log.info(f' Skipping to ckpt {opt.last_ckpt}')

"""###Training Language Detector (EAFQ) on language-labeled target data"""

# Trainable layers for EAFQ training with trainable embeddings
log.info(f' Current ckpt : {current_ckpt}')
if opt.last_ckpt == current_ckpt:
    log.info(f' Running ckpt {current_ckpt}')
    [x.name for x in EAFQ.trainable_variables]

# train.py : Training Embeddings, F and Q  : hinge loss
log.info(f' Current ckpt : {current_ckpt}')
if opt.last_ckpt == current_ckpt:
    log.info(f' Running ckpt {current_ckpt}')
    log.info('Training EAFQ model with src and tgt data with lang_labels and fixed embeddings...')
    TRAIN = True
    #opt.Q_learning_rate = 1e-5
    EAFQ.compile(optimizer=optimizers.Adam(opt.Q_learning_rate), loss='hinge', metrics=['accuracy'])
    opt.Q_iterations = 1
    for epoch in trange(opt.Q_iterations):
        batch_no = 0
        for (inputs_src, lengths_src, labels_src), (inputs_tgt, lengths_tgt, labels_tgt) in zip(train_src, train_tgt):
            log.info(f" Training on batch no : {batch_no}"); batch_no += 1
            inputs = tf.concat([inputs_src, inputs_tgt], axis=0)
            lengths = tf.concat([lengths_src, lengths_tgt], axis=0)
            lang_labels_src = tf.broadcast_to([1], shape=labels_src.shape)
            lang_labels_tgt = tf.broadcast_to([-1], shape=labels_tgt.shape)
            lang_labels = tf.concat([lang_labels_src, lang_labels_tgt], axis=0)
            history = EAFQ.fit(x=[inputs, lengths], y=lang_labels, epochs=1)

"""## Save the trained models after embedding training of EAFQ (CKPT3)
<br>Continue by loading from this saved checkpoint.
"""

# Save/Load model and weights
current_ckpt += 1
log.info(f' Current ckpt : {current_ckpt}')
if opt.last_ckpt == current_ckpt:
    log.info(f' Running ckpt {current_ckpt}')
    E, A, F, P, Q, EA, EAF, EAFP, EAFQ, LAN = load_models(current_ckpt, __version__=opt.__version__)
elif opt.last_ckpt < current_ckpt:
    log.info("Done/Overstepped last_ckpt before...")
    #input("Interrupt execution to not save else enter anything : ")
    save_models(current_ckpt, __version__=opt.__version__)
    with open(opt.crash_logs, 'w') as foo: foo.write(str(current_ckpt))
    if last_ckpt != -1 and last_ckpt < current_ckpt:
else:
    log.info(f' Skipping to ckpt {opt.last_ckpt}')

"""# Training LAN with traininable embeddings
<br><i>Will take a day<i>
"""

# Trainable layers for LAN training with trainable embeddings
log.info(f' Current ckpt : {current_ckpt}')
if opt.last_ckpt == current_ckpt:
    log.info(f' Running ckpt {current_ckpt}')
    [x.name for x in LAN.trainable_variables]

# Training all - EA, F, P and Q : sparse categorical + adversarial : total loss
#opt.Q_learning_rate = 1e-5
log.info(f' Current ckpt : {current_ckpt}')
if opt.last_ckpt == current_ckpt:
    log.info(f' Running ckpt {current_ckpt}')
    log.info('Training EAFQ model with src and tgt data with lang_labels and fixed embeddings...')
    EAFQ.compile(optimizer=optimizers.Adam(opt.learning_rate), loss='hinge', metrics=['accuracy'])
    LAN.compile(optimizer=optimizers.Adam(opt.learning_rate), loss=total_loss, metrics=['accuracy'])
    opt.Q_iterations = 5
    for epoch in trange(opt.Q_iterations):
        batch_no = 0
        for (inputs_src, lengths_src, labels_src), (inputs_tgt, lengths_tgt, labels_tgt) in zip(train_src, train_tgt):
            log.info(f" Training on batch no : {batch_no}"); batch_no += 1
            inputs = tf.concat([inputs_src, inputs_tgt], axis=0)
            lengths = tf.concat([lengths_src, lengths_tgt], axis=0)
            labels = tf.concat([labels_src, labels_tgt], axis=0)
            lang_labels_src = tf.broadcast_to([1], shape=labels_src.shape)
            lang_labels_tgt = tf.broadcast_to([-1], shape=labels_tgt.shape)
            lang_labels = tf.concat([lang_labels_src, lang_labels_tgt], axis=0)
            history_EAFQ = EAFQ.fit(x=[inputs, lengths], y=lang_labels, epochs=1)
            history_LAN = LAN.fit(x=[inputs_src, lengths_src], y=[labels_src, lang_labels_src], epochs=1)

"""## Save the trained models after embedding training of LAN (CKPT4)
<br>Continue by loading from this saved checkpoint.
"""

# Save/Load model and weights
current_ckpt += 1
log.info(f' Current ckpt : {current_ckpt}')
if opt.last_ckpt == current_ckpt:
    log.info(f' Running ckpt {current_ckpt}')
    E, A, F, P, Q, EA, EAF, EAFP, EAFQ, LAN = load_models(current_ckpt, __version__=opt.__version__)
elif opt.last_ckpt < current_ckpt:
    log.info("Done/Overstepped last_ckpt before...")
    #input("Interrupt execution to not save else enter anything : ")
    save_models(current_ckpt, __version__=opt.__version__)
    with open(opt.crash_logs, 'w') as foo: foo.write(str(current_ckpt))
    if last_ckpt != -1 and last_ckpt < current_ckpt:
else:
    log.info(f' Skipping to ckpt {opt.last_ckpt}')

"""### Evaluate Sentiment classifier and Language Detector"""

# Sentiment classifier results on unseen target data
log.info(f' Current ckpt : {current_ckpt}')
if opt.last_ckpt == current_ckpt:
    log.info(f' Running ckpt {current_ckpt}')
    for inputs, lengths, labels in test_src:
        print("SRC test : \n", EAFP.evaluate(x=[], y=labels_tgt))
    for inputs, lengths, labels in test_tgt:
        print("TGT test : \n", EAFP.evaluate(x=[], y=labels_tgt))

# Language Detector results
log.info(f' Current ckpt : {current_ckpt}')
if opt.last_ckpt == current_ckpt:
    log.info(f' Running ckpt {current_ckpt}')
    for (inputs_src, lengths_src, labels_src), (inputs_tgt, lengths_tgt, labels_tgt) in zip(test_src, test_tgt):
        log.info(f" LD test : ")
        inputs = tf.concat([inputs_src, inputs_tgt], axis=0)
        lengths = tf.concat([lengths_src, lengths_tgt], axis=0)
        labels = tf.concat([labels_src, labels_tgt], axis=0)
        lang_labels_src = tf.broadcast_to([1], shape=labels_src.shape)
        lang_labels_tgt = tf.broadcast_to([-1], shape=labels_tgt.shape)
        lang_labels = tf.concat([lang_labels_src, lang_labels_tgt], axis=0)
        print(EAFQ.evaluate(x=[inputs, lengths], y=lang_labels, epochs=1))

"""# Evaluate trained LAN and other models"""

# Overall LAN results on unseen target data
log.info(f' Current ckpt : {current_ckpt}')
if opt.last_ckpt == current_ckpt:
    log.info(f' Running ckpt {current_ckpt}')
    for (inputs_src, lengths_src, labels_src), (inputs_tgt, lengths_tgt, labels_tgt) in zip(test_src, test_tgt):
        log.info(f" LAN test : ")
        inputs = tf.concat([inputs_src, inputs_tgt], axis=0)
        lengths = tf.concat([lengths_src, lengths_tgt], axis=0)
        labels = tf.concat([labels_src, labels_tgt], axis=0)
        lang_labels_src = tf.broadcast_to([1], shape=labels_src.shape)
        lang_labels_tgt = tf.broadcast_to([-1], shape=labels_tgt.shape)
        lang_labels = tf.concat([lang_labels_src, lang_labels_tgt], axis=0)
        print(LAN.evaluate(x=[inputs_src, lengths_src], y=[labels_src, lang_labels_src], epochs=1))

"""# Save models and weights (CKPT_FINAL)"""

# Save model and weights
# Save/Load model and weights
current_ckpt = 'FINAL'
log.info(f' Current ckpt : {current_ckpt}')
if opt.last_ckpt == current_ckpt:
    log.info(f' Running ckpt {current_ckpt}')
    E, A, F, P, Q, EA, EAF, EAFP, EAFQ, LAN = load_models(current_ckpt, __version__=opt.__version__)
elif TRAIN:
    log.info("Done/Overstepped last_ckpt before...")
    #input("Interrupt execution to not save else enter anything : ")
    save_models(current_ckpt, __version__=opt.__version__)
    with open(opt.crash_logs, 'w') as foo: foo.write(str(current_ckpt))
    if last_ckpt != -1 and last_ckpt < current_ckpt:

# train.py
#if __name__ == "__main__":
#    train(opt)
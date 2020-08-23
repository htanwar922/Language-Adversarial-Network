#!/usr/bin/env ipython

#@title utils.py

"""## Some useful functions"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models, preprocessing

import pdb
import numpy as np

from options import *

DEBUG = lambda x: print(x)

def argmax32(arr, axis=-1, dtype=opt.label_dtype):
    return tf.cast(np.argmax(arr, axis=-1), dtype=dtype)

def get_lines(infile, encoding='utf-8'):
    if os.sep != '\\': return int(subprocess.Popen(f"wc -l \"{str(Path(infile))}\"", shell=True, stdout=subprocess.PIPE).stdout.read().split()[0])
    with io.open(Path(infile), encoding=encoding) as foo:
        lines = sum(1 for line in foo)  #os.path.getsize(infile)
    return lines

def reached_ckpt(ckpt_no):
    opt.current_ckpt = ckpt_no

def load_models(current_ckpt=None, __version__=opt.__version__):
    LAN = models.load_model(Path(opt.ckpt_prefix) / f'ckpt_{current_ckpt}')
    print(LAN.layers, '\n')

    _, E, _, A, F, P, Q = LAN.layers
    print(E, A, F, P, Q, '\n')
    print(F.layers, P.layers, Q.layers, '\n')

    print(LAN.inputs)
    shape = (opt.max_seq_len,)
    inputs, lengths = LAN.inputs

    embeddings = E(inputs)
    outputs_EA = A(embeddings, lengths)
    EA = keras.Model(inputs=[inputs, lengths], outputs=outputs_EA)

    outputs_EAF = F(outputs_EA)
    EAF = keras.Model(inputs=[inputs, lengths], outputs=outputs_EAF, name="FeatureExtractor_AE")

    outputs_EAFP = P(outputs_EAF)
    EAFP = keras.Model(inputs=[inputs, lengths], outputs=outputs_EAFP, name="SemanticClassifier_FAE")

    outputs_EAFQ = Q(outputs_EAF)
    EAFQ = keras.Model(inputs=[inputs, lengths], outputs=outputs_EAFQ, name="LanguageDetector_FAE")

    verified = verify_models(EA, EAF, EAFP, EAFQ)

    return (E, A, F, P, Q, EA, EAF, EAFP, EAFQ, LAN)

def verify_models(EA, EAF, EAFP, EAFQ):
    print(EA.layers, '\n')
    print(EAF.layers, '\n')
    print(EAFP.layers, '\n')
    print(EAFQ.layers, '\n')
    return int(input("Enter 0 if verified OK : "))

def save_models(current_ckpt, __version__=opt.__version__):
    LAN.save(Path(opt.ckpt_prefix) / f'ckpt_{current_ckpt}')
    #LAN.save_weights(Path(opt.ckpt_prefix) / f'saved_weights')

if __name__ == "__main__" and not opt.notebook:
    print(get_lines("bwe/vectors/wiki.multi.en.vec"))

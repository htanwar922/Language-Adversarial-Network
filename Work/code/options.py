#!/usr/bin/env ipython

#@title options.py

# -*- coding: utf-8 -*-
"""## Command-line arguments parser"""
"""LAN_v5.ipynb https://colab.research.google.com/github/htanwar922/Language-Adversarial-Network/blob/master/Work/LAN_v5.ipynb
"""

import tensorflow as tf

import os, sys, logging, argparse
from pathlib import Path

__version__ = int(input("Enter LAN version number (to use for saving and loading purposes) : "))
os.chdir(os.path.dirname(Path(__file__)))

# command-line arguments
#sys.argv = [__file__, '--learning_rate', '0.1', '--Q_learning_rate', '0.1', '--clipvalue', '10', '--epochs', '5', '--n_vecs', '-1', '--train_size_src', '-1', '--train_size_tgt', '-1', '--batch_size', '50000', '--vector_length', '10']#, '--no_F_bn', '--no_P_bn', '--no_Q_bn']
#sys.argv += ['--notebook', 'True']
parser = argparse.ArgumentParser()

#platform arguments
parser.add_argument('--notebook', type=bool, default=False)

# dataset arguments
parser.add_argument('--data_path', default=None)
parser.add_argument('--src_lang', default='en')
parser.add_argument('--tgt_lang', default='fr')
parser.add_argument('--train_size_src', type=int, default=None)        # use all
parser.add_argument('--train_size_tgt', type=int, default=None)        # use all
parser.add_argument('--num_labels', type=int, default=5+1)            # max reviews rating
parser.add_argument('--iterate', action='store_true')                # read through iterations
parser.add_argument('--label_dtype', default=tf.int32)

# sequences and vocab arguments
parser.add_argument('--max_seq_len', type=int, default=100)            # None for no truncate
parser.add_argument('--unk_tok', type=str, default='<unk>')
parser.add_argument('--bos_tok', type=str, default='<s>')
parser.add_argument('--eos_tok', type=str, default='</s>')

# training arguments
parser.add_argument('--epochs', type=int, default=5)
parser.add_argument('--random_seed', type=int, default=1)
parser.add_argument('--batch_size', type=int, default=10000)
parser.add_argument('--buffer_size', type=int, default=40000)
parser.add_argument('--learning_rate', type=float, default=0.05)
parser.add_argument('--Q_learning_rate', type=float, default=0.05)

# bwe arguments
parser.add_argument('--emb_filename', default='')
parser.add_argument('--n_vecs', type=int, default=-1)
parser.add_argument('--random_emb', action='store_true')
parser.add_argument('--fix_unk', action='store_true')                # use a fixed <unk> token for all words without pretrained embeddings when building vocab
parser.add_argument('--emb_size', type=int, default=300)
parser.add_argument('--pre_trained_src_emb_file', type=str, default='bwe/vectors/wiki.multi.en.vec')
parser.add_argument('--pre_trained_tgt_emb_file', type=str, default='bwe/vectors/wiki.multi.fr.vec')

# Feature Extractor
parser.add_argument('--model', default='lstm')                        # dan or lstm or cnn
parser.add_argument('--fix_emb', action='store_true')
parser.add_argument('--vector_length', type=int, default=1)
# for LSTM model
parser.add_argument('--attn', default='dot')                        # attention mechanism (for LSTM): avg, last, dot
parser.add_argument('--bidir_rnn', dest='bidir_rnn', action='store_true', default=True)        # bi-directional LSTM
parser.add_argument('--sum_pooling/', dest='avg_pooling', action='store_false')
parser.add_argument('--avg_pooling/', dest='avg_pooling', action='store_true')
# for CNN model
parser.add_argument('--kernel_num', type=int, default=400)
parser.add_argument('--kernel_sizes', type=int, nargs='+', default=[3,4,5])

# for layers and all models
parser.add_argument('--F_layers', type=int, default=1)
parser.add_argument('--P_layers', type=int, default=1)
parser.add_argument('--Q_layers', type=int, default=1)

parser.add_argument('--q_critic', type=int, default=5)    # Q iterations
parser.add_argument('--_lambda', type=float, default=0.1)

parser.add_argument('--F_bn/', dest='F_bn', action='store_true')
parser.add_argument('--no_F_bn/', dest='F_bn', action='store_false')
parser.add_argument('--P_bn/', dest='P_bn', action='store_true', default=True)
parser.add_argument('--no_P_bn/', dest='P_bn', action='store_false')
parser.add_argument('--Q_bn/', dest='Q_bn', action='store_true', default=True)
parser.add_argument('--no_Q_bn/', dest='Q_bn', action='store_false')

parser.add_argument('--hidden_size', type=int, default=900)
parser.add_argument('--dropout', type=float, default=0)
parser.add_argument('--activation', type=str, default='linear')

parser.add_argument('--clip_Q', type=bool, default=False)
parser.add_argument('--clipvalue', type=float, default=0.01)
parser.add_argument('--clip_lim_FP', type=float, default=None)

# general arguments
parser.add_argument('--device', type=str, default='cuda')
parser.add_argument('--debug/', dest='debug', action='store_true')
parser.add_argument('--__version__', type=int, default=__version__)

# crash arguments
parser.add_argument('--start_fresh/', dest='start_fresh', action='store_false', default=False)
parser.add_argument('--saved_models', default=Path(f'./saved_models/lan_v{__version__}'))
parser.add_argument('--logs', default=Path(f'./saved_models/lan_v{__version__}/logs.txt'))
parser.add_argument('--crash_logs', default=Path(f'./saved_models/lan_v{__version__}/crash_logs.txt'))
parser.add_argument('--ckpt_prefix', default=Path(f'./saved_models/lan_v{__version__}/ckpts'))
parser.add_argument('--last_ckpt', default=0)

opt = parser.parse_args()

opt.saved_models = Path(opt.saved_models)
opt.logs = Path(opt.logs)
opt.crash_logs = Path(opt.crash_logs)
opt.ckpt_prefix = Path(opt.ckpt_prefix)

if not tf.config.list_physical_devices('GPU'): opt.device = 'CPU'

import errno
logging.basicConfig(stream=sys.stderr, level=logging.DEBUG if opt.debug else logging.INFO)
log = logging.getLogger(__name__)
if not os.path.isdir(os.path.dirname(opt.logs)): os.mkdir(os.path.dirname(opt.logs))
open(opt.logs, 'w').close()
if not os.path.exists(os.path.dirname(opt.logs)):
    try: os.makedirs(os.path.dirname(opt.logs))
    except OSError as exc:
        if exc.errno != errno.EEXIST: raise # Guard against race condition
with open(opt.logs, "w") as f: pass
fh = logging.FileHandler(opt.logs)  #Path(opt.saved_models) / 'log.txt'
log.addHandler(fh)

if not os.path.exists(os.path.dirname(opt.crash_logs)):
    try:
        os.makedirs(os.path.dirname(opt.crash_logs))
        opt.start_fresh = True
    except OSError as exc:
        if exc.errno != errno.EEXIST: raise # Guard against race condition

#opt.start_fresh = True
if opt.start_fresh or not os.path.isfile(opt.crash_logs):
    with open(opt.crash_logs, 'w') as foo:
        foo.write('0')
else:
    with open(opt.crash_logs, 'r') as foo:
        opt.last_ckpt = foo.read()
        try:
            opt.last_ckpt = int(opt.last_ckpt)
        except ValueError:
            pass

if __name__ == "__main__":
    print("src_embeddings: ", opt.pre_trained_src_emb_file)
    print("tgt_embeddings: ", opt.pre_trained_tgt_emb_file)
    print("debugging: ", opt.debug)
    print(f'Starting fresh : {opt.start_fresh}')
    print(f'Save models at : {opt.saved_models}')
    print(f'Logging at : {opt.logs}')
    print(f'Crash logging at : {opt.crash_logs}')
    print(f'Checkpoints prefixed in : {opt.ckpt_prefix}')
    log.info(' Start...')
    log.info(f' LAN Version {opt.__version__}')
    print(f'Resuming from checkpoint : {opt.last_ckpt}')
    opt.current_ckpt = 0

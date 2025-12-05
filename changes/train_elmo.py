import logging
logging.getLogger("tensorflow").setLevel(logging.ERROR)
import argparse

import numpy as np

from bilm.training import train, load_options_latest_checkpoint, load_vocab
from bilm.data import BidirectionalLMDataset


def main(args):
    # load the vocab
    vocab = load_vocab(args.vocab_file, 50)

    # define the options
    batch_size = 128  # batch size for each GPU
    # batch_size = 64
    n_gpus = 2

    # number of tokens in training data (this for 1B Word Benchmark)
    # n_train_tokens = 262996448 # 85% of full data has this number of tokens, 20M
    # change this later (90 * # of files)
    
    #n_train_tokens = 5400000 # 10%
    #n_train_tokens = 10800000 # 20%
    #n_train_tokens = 16200000 # 30%
    #n_train_tokens = 21600000 # 40%
    #n_train_tokens = 27000000 # 50%
    n_train_tokens = 32400000 # 60%
    #n_train_tokens = 35100000 # 65%
    #n_train_tokens = 37800000 # 70%
    #n_train_tokens = 43200000 # 80%
    #n_train_tokens = 48600000 # 90%
    #n_train_tokens = 54000000 # 100%


    options = {
     'bidirectional': True,
     #'bidirectional': False,

     'char_cnn': {'activation': 'relu',
      'embedding': {'dim': 16}, #it was 16 for catELMo 4 layer and 8 layer when embedding size is 1024.
      'filters': [
       [1, 32],
       [2, 32],
       [3, 64],
       [4, 128],
       [5, 256],
       [6, 512],
       [7, 1024]
                 ],
      'max_characters_per_token': 50,
      'n_characters': 261,
      'n_highway': 2},
    
     'dropout': 0.1,
    
     'lstm': {
      'cell_clip': 3,
      'dim': 4096,
      'n_layers': 4,
      'proj_clip': 3,
#     'projection_dim': 512,
      'projection_dim': 1024,
      'use_skip_connections': True},
    
     'all_clip_norm_val': 10.0,
    
#      'n_epochs': 10,
#      'n_train_tokens': n_train_tokens,
#      'batch_size': batch_size,
#      'n_tokens_vocab': vocab.size,
#      'unroll_steps': 20,
#      'n_negative_samples_batch': 8192,
     'n_epochs': 1, # default is 10, but we will go with 1.
     'n_train_tokens': n_train_tokens,
     'batch_size': batch_size,
     'n_tokens_vocab': 23,
     'unroll_steps': 20,
     'n_negative_samples_batch': 20,
    }
    

    prefix = args.train_prefix
    data = BidirectionalLMDataset(prefix, vocab, test=False,
                                      shuffle_on_load=True)

    tf_save_dir = args.save_dir
    tf_log_dir = args.save_dir
    train(options, data, n_gpus, tf_save_dir, tf_log_dir)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--save_dir', help='Location of checkpoint files')
    parser.add_argument('--vocab_file', help='Vocabulary file')
    parser.add_argument('--train_prefix', help='Prefix for train files')

    args = parser.parse_args()
    main(args)


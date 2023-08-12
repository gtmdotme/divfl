import os
import argparse
import importlib
import random
from loguru import logger

import numpy as np
import sklearn # load sklearn before TF, else error on scholar cluster
import tensorflow as tf

from flearn.utils.model_utils import read_data
logger.info("modules imported")

# GLOBAL PARAMETERS
TRAINERS = ['fedavg', 'fedprox', 'feddane', 'fedddane', 'fedsgd', 'fedprox_origin']
DATASETS = ['celeba', 'sent140', 'nist', 'shakespeare', 'mnist', 
'synthetic_iid', 'synthetic_0_0', 'synthetic_0.5_0.5', 'synthetic_1_1', 'synthetic_cluster']  # NIST is EMNIST in the paepr

MODEL_PARAMS = {
    'sent140.bag_dnn': (2,), # num_classes
    'sent140.stacked_lstm': (25, 2, 100), # seq_len, num_classes, num_hidden 
    'sent140.stacked_lstm_no_embeddings': (25, 2, 100), # seq_len, num_classes, num_hidden
    'nist.mclr': (26,),  # num_classes
    'nist.cnn':(10,),
    'mnist.mclr': (10,), # num_classes
    'mnist.cnn': (10,),  # num_classes
    'shakespeare.stacked_lstm': (80, 80, 256), # seq_len, emb_dim, num_hidden
    'synthetic.mclr': (10, ), # num_classes
    'celeba.cnn': (2,)
}


def parse_inputs():
    """ Parse command line arguments or load defaults """
    parser = argparse.ArgumentParser()

    parser.add_argument('--trainer',
                        help='name of trainer;',
                        type=str,
                        choices=TRAINERS,
                        default='fedavg')
    parser.add_argument('--dataset',
                        help='name of dataset;',
                        type=str,
                        choices=DATASETS,
                        default='nist')
    parser.add_argument('--model',
                        help='name of model;',
                        type=str,
                        default='mclr')
    parser.add_argument('--num_rounds',
                        help='number of rounds to simulate;',
                        type=int,
                        default=-1)
    parser.add_argument('--eval_every',
                        help='evaluate every ____ rounds;',
                        type=int,
                        default=-1)
    parser.add_argument('--clients_per_round',
                        help='number of clients trained per round;',
                        type=int,
                        default=-1)
    parser.add_argument('--batch_size',
                        help='batch size when clients train on data;',
                        type=int,
                        default=10)
    parser.add_argument('--num_epochs', 
                        help='number of epochs when clients train on data;',
                        type=int,
                        default=1)
    parser.add_argument('--num_iters',
                        help='number of iterations when clients train on data;',
                        type=int,
                        default=1)
    parser.add_argument('--learning_rate',
                        help='learning rate for inner solver;',
                        type=float,
                        default=0.003)
    parser.add_argument('--mu',
                        help='constant for prox;',
                        type=float,
                        default=0)
    parser.add_argument('--seed',
                        help='seed for randomness;',
                        type=int,
                        default=0)
    parser.add_argument('--drop_percent',
                        help='percentage of slow devices',
                        type=float,
                        default=0.1)
    parser.add_argument('--clientsel_algo',
                        help='Client Selection Algorithm',
                        type=str,
                        default='random')
    parser.add_argument('--Ls0',
                        help='Constant for grad. similarity',
                        type=int,
                        default=2)
    parser.add_argument('--sim_metric',
                        help='similarity metric',
                        type=str,
                        default='grad')
    parser.add_argument('--m_interval',
                        help='frequency of sending gradient metric for submodular',
                        type=int,
                        default=1)

    try: hyper_params = vars(parser.parse_args())
    except IOError as msg: parser.error(str(msg))

    return hyper_params


def main():
    logger.info("main start")
    
    # suppress tf warnings
    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.WARN)
    logger.info("tf warnings suppressed")

    # parse command line arguments
    hyper_params = parse_inputs()
    logger.info("inputs read")

    # read data
    train_path = os.path.join('data', hyper_params['dataset'], 'data', 'train')
    test_path = os.path.join('data', hyper_params['dataset'], 'data', 'test')
    dataset = read_data(train_path, test_path)
    logger.info("data read")

    # Set seeds
    random.seed(1 + hyper_params['seed'])
    np.random.seed(12 + hyper_params['seed'])
    tf.compat.v1.set_random_seed(123 + hyper_params['seed'])
    # tf.random.set_seed(123 + hyper_params['seed'])

    # dynamically import selected model
    if hyper_params['dataset'].startswith("synthetic"):  # all synthetic datasets use the same model
        model_path = '%s.%s.%s.%s' % ('flearn', 'models', 'synthetic', hyper_params['model'])
    else:
        model_path = '%s.%s.%s.%s' % ('flearn', 'models', hyper_params['dataset'], hyper_params['model'])
    mod = importlib.import_module(model_path)
    Model = getattr(mod, 'Model')
    # add selected model parameter
    hyper_params['model_params'] = MODEL_PARAMS['.'.join(model_path.split('.')[2:])]

    # dynamically import selected trainer
    opt_path = 'flearn.trainers.%s' % hyper_params['trainer']
    mod = importlib.import_module(opt_path)
    Trainer = getattr(mod, 'Server')

    # print argument settings
    maxLen = max([len(ii) for ii in hyper_params.keys()]);
    fmtString = '\t%' + str(maxLen) + 's : %s';
    print('Arguments:')
    for keyPair in sorted(hyper_params.items()): print(fmtString % keyPair)

    # run federated training
    t = Trainer(hyper_params, Model, dataset)
    logger.info("trainer created")
    t.train()
    
if __name__ == '__main__':
    main()

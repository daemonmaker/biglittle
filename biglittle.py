import ipdb
import time
import numpy as np
import sys
import os
import cPickle as pkl

from logistic_sgd import load_data

import theano
from theano import function
from theano import tensor as T
from theano import config
from theano import shared

from layer import HiddenLayer, HiddenBlockLayer
import connections as cnx

from utils import *
from train_big import test_big_and_little_train_big


if __name__ == '__main__':
    rng = np.random.RandomState()
    batch_size = 128
    #n_epochs = 1000

#    test_big_and_little_train_both(
#        rng,
#        batch_size,
#        learning_rate=0.01,
#        n_epochs=n_epochs
#    )

    test_big_and_little_train_big(
        rng,
        batch_size=8500,
        learning_rate=LinearChangeRate(0.011, -0.01, 0.01, 'learning_rate'),
        momentum_rate=LinearChangeRate(0.001, 0.000001, 0.99, 'momentum_rate'),
        n_epochs=10,
        #L1_reg=0.0001,
        #L2_reg=0.0001,
        restore_parameters=False,
        select_top_active=True,
        mult_small_net_params=False,
        zero_last_layer_params=False,
        train_little_net=False,
        train_big_net=True
    )

#    test_static_activations(
#        rng,
#        batch_size,
#        learning_rate=0.01,
#        n_epochs=n_epochs,
#        L1_reg=0,
#        L2_reg=0
#    )
#
#    test_associations(
#        rng,
#        batch_size,
#        learning_rate=0.01,
#        n_epochs=n_epochs,
#        L1_reg=0,
#        L2_reg=0
#    )

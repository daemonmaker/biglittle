#! /usr/bin/env python

import ipdb
import time
from datetime import datetime
import numpy as np
import sys
import os
import cPickle as pkl

import theano
from theano import function
from theano import tensor as T
from theano import config
from theano import shared
from theano.tensor.shared_randomstreams import RandomStreams

from utils import *
from layer import HiddenLayer, HiddenRandomBlockLayer
from timing_stats import TimingStats as TS


def setup_mode_parameters(data, k_per, n_out, n_units_per):
    print "... Building layers"

    # Shared variable used for always activating one block in a layer as in the
    # input and output layer
    one_block_idxs = shared(
        np.zeros((batch_size, 1), dtype='int64'),
        name='one_block_idxs'
    )

    hidden_dims = (n_out, n_units_per)
    return [
        {
            'n_in': (1, data.train_set_x.shape[2].eval()),
            'n_out': hidden_dims,
            'in_idxs': one_block_idxs,
            'k': k_per,
            'activation': T.tanh,
        },
        {
            'n_in': hidden_dims,
            'n_out': hidden_dims,
            'k': k_per,
            'activation': T.tanh,
        },
        #{
        #     'n_in': hidden_dims,
        #     'n_out': hidden_dims,
        #     'k': k_per,
        #     'activation': T.tanh,
        # },
        # {
        #     'n_in': hidden_dims,
        #     'n_out': hidden_dims,
        #     'k': k_per,
        #     'activation': T.tanh,
        # },
        {
            'n_in': hidden_dims,
            'n_out': (1, 10),
            'out_idxs': one_block_idxs,
            'k': 1.,
            'activation': None,
        }
    ]


class MNIST():
    def __init__(self):
        dataset = 'mnist.pkl.gz'
        datasets = load_data(dataset, True)

        self.train_set_x, self.train_set_y = datasets[0]
        self.valid_set_x, self.valid_set_y = datasets[1]
        self.test_set_x, self.test_set_y = datasets[2]

        self.n_train_batches = self.train_set_x.get_value(
            borrow=True
        ).shape[0] / batch_size
        self.n_valid_batches = self.valid_set_x.get_value(
            borrow=True
        ).shape[0] / batch_size
        self.n_test_batches = self.test_set_x.get_value(
            borrow=True
        ).shape[0] / batch_size


def build_big_model(
        data,
        parameters,
        rng,
        batch_size,
        learning_rate,
        n_hids,
        k_per=0.05,
        n_epochs=1000,
        L1_reg=0.0,
        L2_reg=0.0001,
        zero_last_layer_params=False,
):
    index = T.lscalar('index')
    input = T.tensor3('input', dtype=config.floatX)
    y = T.ivector('y')

    print "Building models"
    # Create network structure

    b_layers = []
    for i, params in enumerate(parameters):
        params['batch_size'] = batch_size
        if 'name' not in params.keys():
            params['name'] = 'b_layer_%d' % i
        if 'in_idxs' not in params.keys():
            params['in_idxs'] = b_layers[-1].out_idxs
        new_layer = HiddenRandomBlockLayer(**params)
        print new_layer
        b_layers.append(new_layer)

    if zero_last_layer_params:
        b_layers[-1].W.set_value(0*b_layers[-1].W.get_value())
        b_layers[-1].b.set_value(0*b_layers[-1].b.get_value())

    print "... Building top active updates"
    top_active = []
    b_activation = input
    b_activations = [b_activation]
    for i in range(len(b_layers)):
        b_activation = b_layers[i].output(b_activation)
        b_activations.append(b_activation)
        #top_active.append((
        #    top_actives[i],
        #    T.argsort(T.abs_(l_activation))[:, :l_layers[i].k]
        #))

    print "... Building costs and errors"
    # T.nnet.softmax takes a matrix not a tensor so we only calculate the
    # linear component at the last layer and here we reshape and then
    # apply the softmax
    #b_activation = T.nnet.softmax(((b_activation*b_activation)**2).sum(axis=2))
    #b_activation = relu_softmax(((b_activation*b_activation)**2).sum(axis=2))
    #b_activation = T.nnet.softmax(T.mean(b_activation, axis=2))
    #b_activation = relu_softmax(T.mean(b_activation, axis=2))
    #b_activation = T.nnet.softmax(T.max(b_activation, axis=2))
    #b_activation = relu_softmax(T.max(b_activation, axis=2))
    b_shp = b_activation.shape
    #b_activation = relu_softmax(b_activation.reshape((b_shp[0], b_shp[2])))
    b_activation = T.nnet.softmax(b_activation.reshape((b_shp[0], b_shp[2])))
    b_activations.append(b_activation)
    b_cost = add_regularization(
        b_layers,
        b_layers[-1].cost(b_activation, y),
        L1_reg,
        L2_reg
    )
    b_error = b_layers[-1].error(b_activation, y)

    print "... Building parameter updates"
    consider_constants = []
    for i in range(len(b_layers)):
        for param in b_layers[i].params:
            consider_constants += [b_layers[i].in_idxs, b_layers[i].out_idxs]
    b_grads = []
    b_param_updates = []
    for i in range(len(b_layers)):
        for param in b_layers[i].params:
            b_gparam = T.grad(
                b_cost,
                param,
                consider_constant=consider_constants
            )
            b_grads.append(b_gparam)
            b_param_updates.append((param, param - learning_rate*b_gparam))

    print "... Compiling big net train function"
    b_updates = b_param_updates

    train_model = function(
        [index],
        [b_cost],
        updates=b_updates,
        givens={
            input: data.train_set_x[index*batch_size:(index+1)*batch_size],
            y: data.train_set_y[index*batch_size:(index+1)*batch_size]
        }
    )

    print "... Compiling big net test function"
    test_model = function(
        [index],
        b_error,
        givens={
            input: data.test_set_x[index*batch_size:(index+1)*batch_size],
            y: data.test_set_y[index*batch_size:(index+1)*batch_size]
        }
    )

    print "... Compiling big net validate function"
    validate_model = function(
        [index],
        b_error,
        givens={
            input: data.valid_set_x[index*batch_size:(index+1)*batch_size],
            y: data.valid_set_y[index*batch_size:(index+1)*batch_size]
        }
    )

    return {
        'train_model': train_model,
        'test_model': test_model,
        'validate_model': validate_model
    }


def train(
        train_model,
        test_model,
        validate_model,
        learning_rate,
        shared_learning_rate
):
    def summarize_rates():
        print "Learning rate: ", learning_rate.rate

    # early-stopping parameters
    patience = 10000  # look as this many examples regardless
    patience_increase = 10  # wait this much longer when a new best is
                           # found
    improvement_threshold = 0.995  # a relative improvement of this much is
                                   # considered significant
    validation_frequency = min(data.n_train_batches, patience / 2)
                                  # go through this many
                                  # minibatche before checking the network
                                  # on the validation set; in this case we
                                  # check every epoch

    best_params = None
    this_validation_loss = 0
    this_validation_loss_b = 0
    best_validation_loss = np.inf
    best_validation_loss_b = best_validation_loss
    best_iter = 0
    test_score = 0.
    test_score_b = 0.
    accum_b = 0
    epoch = 0
    train_time_accum_b = 0
    done_looping = False

    timers = ['train', 'valid', 'train']
    ts = TS(['epoch', 'valid'])
    ts_b = TS(timers)

    summarize_rates()

    ts.start()
    while (epoch < n_epochs) and (not done_looping):
        epoch = epoch + 1
        ts.start('epoch')
        for minibatch_index in xrange(data.n_train_batches):
            ts_b.start('train')
            minibatch_avg_cost_b = train_model(minibatch_index)
            ts_b.end('train')
            #print "0: ", b_layers[-5].in_idxs.get_value()
            #print "1: ", b_layers[-4].in_idxs.get_value()
            #print "2: ", b_layers[-3].in_idxs.get_value()
            #print "3: ", b_layers[-2].in_idxs.get_value()
            #print "4: ", b_layers[-1].in_idxs.get_value()

            minibatch_avg_cost_b = minibatch_avg_cost_b[0]
            accum_b = accum_b + minibatch_avg_cost_b

            #print "minibatch_avg_cost: " + str(minibatch_avg_cost) + " minibatch_avg_cost_b: " + str(minibatch_avg_cost_b)
            #print l_layers[0].W.get_value().sum(), l_layers[1].W.get_value().sum(), b_layers[0].W.get_value().sum(), b_layers[1].W.get_value().sum()
            #print "A: ", np.max(np.abs(b_layers[0].W.get_value())), np.max(np.abs(b_layers[0].b.get_value())), np.max(np.abs(b_layers[1].W.get_value())), np.max(np.abs(b_layers[1].b.get_value()))
            #print "B: ", np.abs(b_layers[0].W.get_value()).sum(), np.abs(b_layers[0].b.get_value()).sum(), np.abs(b_layers[1].W.get_value()).sum(), np.abs(b_layers[1].b.get_value()).sum()
            #print "C: ", np.abs(np.array(minibatch_avg_cost_b[1])).sum(), np.abs(np.array(minibatch_avg_cost_b[2])).sum(), np.abs(np.array(minibatch_avg_cost_b[3])).sum(), np.abs(np.array(minibatch_avg_cost_b[4])).sum()

            # iteration number
            iter = (epoch - 1) * data.n_train_batches + minibatch_index

            if (iter + 1) % validation_frequency == 0:
                ts.end('epoch')
                ts.reset('epoch')

                ts_b.reset('train')
                accum_b = accum_b / validation_frequency
                b_summary = ("minibatch_avg_cost_b: %f, time: %f"
                             % (accum_b, ts_b.accumed['train'][-1][1]))
                accum_b = 0

                print "%s" % (b_summary)

                # compute zero-one loss on validation set
                summary = ('epoch %i, minibatch %i/%i'
                           % (epoch, minibatch_index + 1, data.n_train_batches))

                validation_losses_b = [validate_model(i) for i
                                       in xrange(data.n_valid_batches)]
                this_validation_loss_b = np.mean(validation_losses_b)
                #this_validation_loss_b = 0
                b_summary = ('big validation error %f %% '
                             % (this_validation_loss_b * 100.))

                print ("%s %s" % (summary, b_summary))
                #ipdb.set_trace()

                # if we got the best validation score until now
                this_validation_loss = this_validation_loss_b

                if this_validation_loss < best_validation_loss:
                    #improve patience if loss improvement is good enough
                    if this_validation_loss < best_validation_loss *  \
                       improvement_threshold:
                        patience = max(patience, iter * patience_increase)

                    best_validation_loss_b = this_validation_loss_b
                    best_validation_loss = best_validation_loss_b

                    best_iter = iter

                    # test it on the test set
                    test_losses_b = [test_model(i) for i
                                     in xrange(data.n_test_batches)]
                    test_score_b = np.mean(test_losses_b)
                    #test_score_b = 0
                    b_summary = 'big: %f' % (test_score_b * 100.)

                    print ('     epoch %i, minibatch %i/%i,'
                           ' test error of best model %s'
                           % (epoch, minibatch_index + 1,
                              data.n_train_batches, b_summary))

                learning_rate.update()

                shared_learning_rate.set_value(learning_rate.rate)

                summarize_rates()

            if patience <= iter:
                    done_looping = True
                    break

    ts.end()
    print('Optimization complete. Best validation score of %f %% '
          'obtained at iteration %i, with test performance %f %%' %
          (best_validation_loss_b * 100., best_iter + 1, test_score_b * 100.))
    print >> sys.stderr, ('The code for file ' +
                          os.path.split(__file__)[1] +
                          ' ran for %s' % ts)

    return ts.diffs['epoch']


if __name__ == '__main__':
    build_model_func = build_big_model
    rng = np.random.RandomState()
    batch_size = 10
    n_epochs = 1000
    learning_rate = LinearChangeRate(0.011, -0.01, 0.01, 'learning_rate')
    n_hids_len = 5
    n_hids = (pow(10, y) for y in range(2, 2+n_hids_len))
    n_hids = (25,)
    n_units_per = 20
    k_per = 1

    print "Loading Data"
    print "... MNIST"
    data = MNIST()

    epoch_times = np.zeros((n_hids_len, 1))
    counter = 0
    for n_hid in n_hids:
        try:
            shared_learning_rate = shared(
                np.array(learning_rate.rate, dtype=config.floatX),
                name='learning_rate'
            )

            k = int(n_hid*k_per)
            print "k_per: %d, k: %d" % (k_per, k)

            models = build_model_func(
                data,
                setup_mode_parameters(data, k_per, n_hid, n_units_per),
                rng,
                batch_size=batch_size,
                learning_rate=shared_learning_rate,
                n_hids=n_hid,
                k_per=k_per,
                n_epochs=n_epochs,
                #L1_reg=0.0001,
                L2_reg=0.0001,
            )

            print "Training"
            b_epoch_tim = train(
                learning_rate=learning_rate,
                shared_learning_rate=shared_learning_rate,
                **models
            )
        except MemoryError:
            b_epoch_time = -1

        print "b_epoch_time: %f" % (b_epoch_time)
        epoch_times[counter] = [b_epoch_time]
        counter = counter + 1

    print epoch_times
    ipdb.set_trace()

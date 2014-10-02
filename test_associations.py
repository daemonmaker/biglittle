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

from utils import *
from layer import HiddenLayer, HiddenBlockLayer


def test_associations(
        rng,
        batch_size=1,
        learning_rate=0.01,
        n_epochs=1000,
        L1_reg=0.0,
        L2_reg=0.0001
):
    print "Loading data"
    print "... MNIST"
    dataset = 'mnist.pkl.gz'
    datasets = load_data(dataset)

    train_set_x, train_set_y = datasets[0]

    x_size = train_set_x.shape[1].eval()

    temp = train_set_x.get_value(borrow=True, return_internal_type=True)
    train_set_x = shared(
        temp.reshape((
            temp.shape[0],
            1,
            temp.shape[1]
        )),
        borrow=True,
        name='train_set_x'
    )

    valid_set_x, valid_set_y = datasets[1]

    temp = valid_set_x.get_value(borrow=True, return_internal_type=True)
    valid_set_x = shared(
        temp.reshape((
            temp.shape[0],
            1,
            temp.shape[1]
        )),
        borrow=True,
        name='valid_set_x'
    )

    test_set_x, test_set_y = datasets[2]

    temp = test_set_x.get_value(borrow=True, return_internal_type=True)
    test_set_x = shared(
        temp.reshape((
            temp.shape[0],
            1,
            temp.shape[1]
        )),
        borrow=True,
        name='valid_set_x'
    )

    n_train_batches = train_set_x.get_value(borrow=True).shape[0] / batch_size
    n_valid_batches = valid_set_x.get_value(borrow=True).shape[0] / batch_size
    n_test_batches = test_set_x.get_value(borrow=True).shape[0] / batch_size

    n_in = x_size
    n_units_per = 32
    n_out = 500

    dims = []

    print "... Activation patterns"
    ins, outs = cnx.load('associations.pkl')

    def zero_weights(W, W_name, in_idxs, out_idxs):
        ipdb.set_trace()
        W_mask = np.zeros_like(W)
        for o_idxs in out_idxs:
            #idxs = np.setdiff1d(in_idxs, out_idxs)
            #W_mask = 1.
            W_mask[in_idxs, o_idxs, :, :] = 1.
        ipdb.set_trace()
        return shared(W_mask, name=W_name+'_mask')

    #in_idxs = []
    #out_idxs = []
    #for i in range(len(ins)):
    #    dims.append((ins[i].shape[0], outs[i].shape[0]))
    #    in_idxs.append(shared(
    #        repeat(ins[i]),
    #        name='in_idxs_%i' % i
    #    ))
    #    out_idxs.append(shared(
    #        repeat(outs[i]),
    #        name='out_idxs_%i' % i
    #    ))

    print "Building model"
    index = T.lscalar('index')
    x = T.tensor3('x', dtype=config.floatX)
    y = T.ivector('y')

    layers = []

    n_in = 1
    n_out = 500
    in_idxs_0 = shared(cnx.all_idxs(n_in, batch_size), 'in_idxs_0')
    out_idxs_0 = shared(cnx.all_idxs(n_out, batch_size), name='out_idxs_0')
    layers.append(
        HiddenBlockLayer(
            (n_in, x_size),
            (n_out, n_units_per),
            in_idxs_0,
            out_idxs_0,
            batch_size,
            activation=T.tanh,
            name='layer_' + str(len(layers))
        )
    )

    n_in = n_out
    n_out = 10
    in_idxs_1 = out_idxs_0
    out_idxs_1 = shared(cnx.all_idxs(n_out, batch_size), name='out_idxs_1')
    layers.append(
        HiddenBlockLayer(
            (n_in, n_units_per),
            (n_out, n_units_per),
            out_idxs_0,
            out_idxs_1,
            batch_size,
            None,
            name='layer_' + str(len(layers))
        )
    )
    #layers[-1].W.set_value(0*layers[-1].W.get_value())

    for idx, layer in enumerate(layers):
        layer.W_mask = zero_weights(
            layer.W.get_value(),
            layer.W.name, ins[0],
            outs[0],
        )

    print "... Building cost and error equations"
    activation = x
    for layer in layers:
        activation = layer.output(activation)
    activation = T.nnet.softmax(T.mean(activation, axis=2))
    cost = add_regularization(
        layers,
        layers[-1].cost(activation, y),
        L1_reg,
        L2_reg
    )
    error = layers[-1].error(activation, y)

    print "... Building parameter updates"
    param_updates = []
    for layer in layers:
        for param in layer.params:
            gparam = T.grad(cost, param)
            param_updates.append((param, param - learning_rate*gparam))

    print "... Compiling train function"
    train_model = function(
        [index],
        cost,
        updates=param_updates,
        givens={
            x: train_set_x[index*batch_size:(index+1)*batch_size],
            y: train_set_y[index*batch_size:(index+1)*batch_size]
        }
    )

    print "... Compiling test function"
    test_model = function(
        [index],
        error,
        givens={
            x: test_set_x[index*batch_size:(index+1)*batch_size],
            y: test_set_y[index*batch_size:(index+1)*batch_size]
        }
    )

    print "... Compiling validate function"
    validate_model = function(
        [index],
        error,
        givens={
            x: valid_set_x[index*batch_size:(index+1)*batch_size],
            y: valid_set_y[index*batch_size:(index+1)*batch_size]
        }
    )

    ###############
    # TRAIN MODEL #
    ###############
    print '... training'

    # early-stopping parameters
    patience = 10000  # look as this many examples regardless
    patience_increase = 100  # wait this much longer when a new best is
                           # found
    improvement_threshold = 0.995  # a relative improvement of this much is
                                   # considered significant
    validation_frequency = min(n_train_batches, patience / 2)
                                  # go through this many
                                  # minibatche before checking the network
                                  # on the validation set; in this case we
                                  # check every epoch

    best_params = None
    best_validation_loss = np.inf
    best_iter = 0
    test_score = 0.
    start_time = time.clock()

    epoch = 0
    done_looping = False
    accum = 0
    while (epoch < n_epochs) and (not done_looping):
        epoch = epoch + 1
        for minibatch_index in xrange(n_train_batches):

            minibatch_avg_cost = train_model(minibatch_index)
            accum = accum + minibatch_avg_cost
            # iteration number
            iter = (epoch - 1) * n_train_batches + minibatch_index

            if (iter + 1) % validation_frequency == 0:
                accum = accum / validation_frequency
                print "minibatch_avg_cost: ", accum

                # compute zero-one loss on validation set
                validation_losses = [validate_model(i) for i
                                     in xrange(n_valid_batches)]
                this_validation_loss = np.mean(validation_losses)

                print('epoch %i, minibatch %i/%i, validation error %f %%' %
                     (epoch, minibatch_index + 1, n_train_batches,
                      this_validation_loss * 100.))

                # if we got the best validation score until now
                if this_validation_loss < best_validation_loss:
                    #improve patience if loss improvement is good enough
                    if this_validation_loss < best_validation_loss *  \
                       improvement_threshold:
                        patience = max(patience, iter * patience_increase)

                    best_validation_loss = this_validation_loss
                    best_iter = iter

                    # test it on the test set
                    test_losses = [test_model(i) for i
                                   in xrange(n_test_batches)]
                    test_score = np.mean(test_losses)

                    print(('     epoch %i, minibatch %i/%i, test error of '
                           'best model %f %%') %
                          (epoch, minibatch_index + 1, n_train_batches,
                           test_score * 100.))

            if patience <= iter:
                    done_looping = True
                    break

    end_time = time.clock()
    print(('Optimization complete. Best validation score of %f %% '
           'obtained at iteration %i, with test performance %f %%') %
          (best_validation_loss * 100., best_iter + 1, test_score * 100.))
    print >> sys.stderr, ('The code for file ' +
                          os.path.split(__file__)[1] +
                          ' ran for %.2fm' % ((end_time - start_time) / 60.))

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


def test_big_and_little_train_both(
        rng,
        batch_size=1,
        learning_rate=0.01,
        n_epochs=1000,
        L1_reg=0.0,
        L2_reg=0.0001
):
    l_learning_rate = learning_rate
    b_learning_rate = 10*learning_rate

    index = T.lscalar('index')
    l_x = T.matrix('l_x', dtype=config.floatX)
    b_x = T.tensor3('b_x', dtype=config.floatX)
    y = T.ivector('y')

    print "Loading Data"
    dataset = 'mnist.pkl.gz'
    datasets = load_data(dataset)

    train_set_x, train_set_y = datasets[0]
    valid_set_x, valid_set_y = datasets[1]
    test_set_x, test_set_y = datasets[2]

    n_train_batches = train_set_x.get_value(borrow=True).shape[0] / batch_size
    n_valid_batches = valid_set_x.get_value(borrow=True).shape[0] / batch_size
    n_test_batches = test_set_x.get_value(borrow=True).shape[0] / batch_size

    print "Building models"
    print "... Building layers"
    # Create network structure
    x_size = train_set_x.shape[1].eval()
    n_in = x_size
    n_units_per = 32
    n_out = 500
    l_layers = []
    b_layers = []

    l_layers.append(
        HiddenLayer(
            n_in,
            n_out,
            batch_size,
            #k=0.05,
            k=1,
            activation=T.tanh,
            name='l_layer_' + str(len(l_layers))
        )
    )

    in_idxs_0 = shared(
        np.zeros((batch_size, 1), dtype='int64'),
        name='in_idxs_0'
    )
    b_layers.append(
        HiddenBlockLayer(
            (1, x_size),
            (n_out, n_units_per),
            in_idxs_0,
            l_layers[-1].top_active,
            batch_size,
            activation=T.tanh,
            name='b_layer_' + str(len(b_layers))
        )
    )

    #n_in = n_out
    #n_out = 100
    #k_activations = 0.12
    #l_layers.append(
    #    HiddenLayer(
    #        n_in,
    #        n_out,
    #        k=k_activations,
    #        name='l_layer_' + str(len(l_layers))
    #    )
    #)
    #b_layers.append(HiddenBlockLayer(n_in, n_out, batch_size))

    n_in = n_out
    n_out = 10
    l_layers.append(
        HiddenLayer(
            n_in,
            n_out,
            batch_size,
            k=1,
            activation=T.nnet.softmax,
            name='l_layer_' + str(len(l_layers))
        )
    )
    l_layers[-1].W.set_value(0*l_layers[-1].W.get_value())

    # T.nnet.softmax takes a matrix not a tensor so just calculate the linear
    # component in the layer and apply the softmax later
    #out_idxs_n = shared(
    #    np.repeat(
    #        np.arange(n_out, dtype='int64').reshape(1, n_out),
    #        batch_size,
    #        axis=0
    #    ),
    #    name='out_idxs_' + str(len(l_layers))
    #)
    b_layers.append(HiddenBlockLayer(
        (n_in, n_units_per),
        (n_out, n_units_per),
        l_layers[-2].top_active,
        l_layers[-1].top_active,
        #out_idxs_n,
        batch_size,
        None,
        name='b_layer_' + str(len(b_layers))
    ))
    #b_layers[-1].W.set_value(0*b_layers[-1].W.get_value())

    print "... Building top active updates"
    top_active = []
    l_activation = l_x
    b_activation = b_x
    b_activations = [b_activation]
    for i in range(len(l_layers)):
        l_activation = l_layers[i].output(l_activation)
        b_activation = b_layers[i].output(b_activation)
        b_activations.append(b_activation)
        top_active.append((
            l_layers[i].top_active,
            T.argsort(T.abs_(l_activation))[:, :l_layers[i].k]
        ))

    print "... Building costs and errors"
    l_cost = add_regularization(
        l_layers,
        l_layers[-1].cost(l_activation, y),
        L1_reg,
        L2_reg
    )
    l_error = l_layers[-1].error(l_activation, y)

    # T.nnet.softmax takes a matrix not a tensor so we only calculate the
    # linear component at the last layer and here we reshape and then
    # apply the softmax
    #b_activation = T.nnet.softmax(((b_activation*b_activation)**2).sum(axis=2))
    #b_activation = relu_softmax(((b_activation*b_activation)**2).sum(axis=2))
    b_activation = T.nnet.softmax(T.mean(b_activation, axis=2))
    #b_activation = relu_softmax(T.mean(b_activation, axis=2))
    #b_activation = T.nnet.softmax(T.max(b_activation, axis=2))
    #b_activation = relu_softmax(T.max(b_activation, axis=2))
    b_activations.append(b_activation)
    b_cost = add_regularization(
        b_layers,
        b_layers[-1].cost(b_activation, y),
        L1_reg,
        L2_reg
    )
    b_error = b_layers[-1].error(b_activation, y)

    print "... Building parameter updates"
    l_grads = []
    l_param_updates = []
    b_grads = []
    b_param_updates = []
    for i in range(len(l_layers)):
        for param in l_layers[i].params:
            gparam = T.grad(l_cost, param)
            l_grads.append(gparam)
            l_param_updates.append((param, param - l_learning_rate*gparam))

        for param in b_layers[i].params:
            gparam = T.grad(
                b_cost,
                param,
                consider_constant=[b_layers[i].in_idxs, b_layers[i].out_idxs]
            )
            b_grads.append(gparam)
            b_param_updates.append((param, param - b_learning_rate*gparam))

    print "... Compiling little net train function"
    l_train_model = function(
        [index],
        [l_cost, l_x, y],
        updates=top_active+l_param_updates,
        givens={
            l_x: train_set_x[index*batch_size:(index+1)*batch_size],
            y: train_set_y[index*batch_size:(index+1)*batch_size]
        }
    )

    print "... Compiling big net train function"
    temp = train_set_x.get_value(borrow=True, return_internal_type=True)
    train_set_x_b = shared(
        temp.reshape((
            temp.shape[0],
            1,
            temp.shape[1]
        )),
        borrow=True,
        name='train_set_x_b'
    )
    b_train_model = function(
        [index],
        [b_cost],
        updates=b_param_updates,
        givens={
            b_x: train_set_x_b[index*batch_size:(index+1)*batch_size],
            y: train_set_y[index*batch_size:(index+1)*batch_size]
        }
    )

    #theano.printing.debugprint(b_train_model)
    #ipdb.set_trace()

#    verify_layers(batch_size, b_layers, train_set_x_b, train_set_y)
#    temp = verify_cost(
#        b_cost,
#        b_layers,
#        b_x,
#        y,
#        batch_size,
#        train_set_x_b,
#        train_set_y
#    )
#    T.verify_grad(
#        temp,
#        [b_layers[0].W.get_value(), b_layers[1].W.get_value()],
#        rng=rng
#    )

    print "... Compiling little net test function"
    l_test_model = function(
        [index],
        l_error,
        givens={
            l_x: test_set_x[index*batch_size:(index+1)*batch_size],
            y: test_set_y[index*batch_size:(index+1)*batch_size]
        }
    )

    print "... Compiling big net test function"
    temp = test_set_x.get_value(borrow=True, return_internal_type=True)
    test_set_x_b = shared(
        temp.reshape((
            temp.shape[0],
            1,
            temp.shape[1]
        )),
        borrow=True,
        name='test_set_x_b'
    )
    b_test_model = function(
        [index],
        b_error,
        givens={
            b_x: test_set_x_b[index*batch_size:(index+1)*batch_size],
            y: test_set_y[index*batch_size:(index+1)*batch_size]
        }
    )

    print "... Compiling little net validate function"
    l_validate_model = function(
        [index],
        l_error,
        givens={
            l_x: valid_set_x[index*batch_size:(index+1)*batch_size],
            y: valid_set_y[index*batch_size:(index+1)*batch_size]
        }
    )

    print "... Compiling big net validate function"
    temp = valid_set_x.get_value(borrow=True, return_internal_type=True)
    valid_set_x_b = shared(
        temp.reshape((
            temp.shape[0],
            1,
            temp.shape[1]
        )),
        borrow=True,
        name='valid_set_x_b'
    )
    b_validate_model = function(
        [index],
        b_error,
        givens={
            b_x: valid_set_x_b[index*batch_size:(index+1)*batch_size],
            y: valid_set_y[index*batch_size:(index+1)*batch_size]
        }
    )

    print "Training"

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
    accum_b = 0
    while (epoch < n_epochs) and (not done_looping):
        epoch = epoch + 1
        for minibatch_index in xrange(n_train_batches):

            minibatch_avg_cost = l_train_model(minibatch_index)
            minibatch_avg_cost_b = b_train_model(
                minibatch_index,
                learning_rate.rate
            )

            #print "minibatch_avg_cost: " + str(minibatch_avg_cost) + " minibatch_avg_cost_b: " + str(minibatch_avg_cost_b)
            #print l_layers[0].W.get_value().sum(), l_layers[1].W.get_value().sum(), b_layers[0].W.get_value().sum(), b_layers[1].W.get_value().sum()
            #print "A: ", np.max(np.abs(b_layers[0].W.get_value())), np.max(np.abs(b_layers[0].b.get_value())), np.max(np.abs(b_layers[1].W.get_value())), np.max(np.abs(b_layers[1].b.get_value()))
            #print "B: ", np.abs(b_layers[0].W.get_value()).sum(), np.abs(b_layers[0].b.get_value()).sum(), np.abs(b_layers[1].W.get_value()).sum(), np.abs(b_layers[1].b.get_value()).sum()
            #print "C: ", np.abs(np.array(minibatch_avg_cost_b[1])).sum(), np.abs(np.array(minibatch_avg_cost_b[2])).sum(), np.abs(np.array(minibatch_avg_cost_b[3])).sum(), np.abs(np.array(minibatch_avg_cost_b[4])).sum()
            minibatch_avg_cost = minibatch_avg_cost[0]
            minibatch_avg_cost_b = minibatch_avg_cost_b[0]
            accum = accum + minibatch_avg_cost
            accum_b = accum_b + minibatch_avg_cost_b

            # iteration number
            iter = (epoch - 1) * n_train_batches + minibatch_index

            if (iter + 1) % validation_frequency == 0:
                accum = accum / validation_frequency
                accum_b = accum_b / validation_frequency
                print "minibatch_avg_cost: ", accum, \
                    "minibatch_avg_cost_b: ", accum_b
                accum = 0
                accum_b = 0

                # compute zero-one loss on validation set
                validation_losses = [l_validate_model(i) for i
                                     in xrange(n_valid_batches)]
                this_validation_loss = np.mean(validation_losses)

                validation_losses_b = [b_validate_model(i) for i
                                       in xrange(n_valid_batches)]
                this_validation_loss_b = np.mean(validation_losses_b)
                #this_validation_loss_b = 0

                print('epoch %i, minibatch %i/%i, validation error %f %% '
                      '(%f %%)' %
                     (epoch, minibatch_index + 1, n_train_batches,
                      this_validation_loss * 100.,
                      this_validation_loss_b * 100.))
                #ipdb.set_trace()

                # if we got the best validation score until now
                if this_validation_loss < best_validation_loss:
                    #improve patience if loss improvement is good enough
                    if this_validation_loss < best_validation_loss *  \
                       improvement_threshold:
                        patience = max(patience, iter * patience_increase)

                    best_validation_loss = this_validation_loss
                    best_iter = iter

                    # test it on the test set
                    test_losses = [l_test_model(i) for i
                                   in xrange(n_test_batches)]
                    test_score = np.mean(test_losses)

                    test_losses_b = [b_test_model(i) for i
                                     in xrange(n_test_batches)]
                    test_score_b = np.mean(test_losses_b)
                    #test_score_b = 0

                    print(('     epoch %i, minibatch %i/%i, test error of '
                           'best model %f %% (%f %%)') %
                          (epoch, minibatch_index + 1, n_train_batches,
                           test_score * 100., test_score_b * 100.))

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

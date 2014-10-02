import ipdb
import time
from datetime import datetime
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
from timing_stats import TimingStats as TS


def test_big_and_little_train_big(
        rng,
        batch_size,
        learning_rate,
        momentum_rate,
        n_epochs=1000,
        L1_reg=0.0,
        L2_reg=0.0001,
        restore_parameters=False,
        select_top_active=False,
        mult_small_net_params=False,
        zero_last_layer_params=False,
        train_little_net=False,
        train_big_net=True
):
    def summarize_rates():
        print "Learning rate: ", learning_rate.rate, \
            "Momentum: ", momentum.get_value()

    assert(train_big_net or train_little_net)

    l_learning_rate = shared(
        np.array(learning_rate.rate, dtype=config.floatX),
        name='learning_rate'
    )
    b_learning_rate = shared(
        np.array(learning_rate.rate, dtype=config.floatX),
        name='learning_rate'
    )
    momentum = shared(
        np.array(momentum_rate.rate, dtype=config.floatX),
        name='momentum'
    )

    index = T.lscalar('index')
    l_x = T.matrix('l_x', dtype=config.floatX)
    b_x = T.tensor3('b_x', dtype=config.floatX)
    y = T.ivector('y')

    print "Loading Data"
    print "... MNIST"
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
    y_size = train_set_y.shape[0].eval()
    n_in = x_size
    n_units_per = 1
    n_out = 5000
    l_layers = []
    b_layers = []
    l_params = None

    # Shared variable used for always activating one block in a layer as in the
    # input and output layer
    one_block_idxs = shared(
        np.zeros((batch_size, 1), dtype='int64'),
        name='one_block_idxs'
    )

    l_layers.append(
        HiddenLayer(
            n_in,
            n_out,
            batch_size,
            k=0.1,
            activation=T.tanh,
            name='l_layer_' + str(len(l_layers))
        )
    )

    if mult_small_net_params:
        l_params = l_layers[-1].params

    b_layers.append(
        HiddenBlockLayer(
            (1, x_size),
            (n_out, n_units_per),
            one_block_idxs,
            l_layers[-1].top_active,
            batch_size,
            activation=T.tanh,
            name='b_layer_' + str(len(b_layers)),
            l_params=l_params,
            l_param_map=[('x', 1, 0, 'x'), (0, 'x')]
        )
    )

    n_in = n_out
    l_layers.append(
        HiddenLayer(
            n_in,
            n_out,
            batch_size,
            k=0.1,
            activation=T.tanh,
            name='l_layer_' + str(len(l_layers))
        )
    )

    if mult_small_net_params:
        l_params = l_layers[-1].params

    b_layers.append(
        HiddenBlockLayer(
            (n_in, n_units_per),
            (n_out, n_units_per),
            l_layers[-2].top_active,
            l_layers[-1].top_active,
            #out_idxs_n,
            batch_size,
            activation=T.tanh,
            name='b_layer_' + str(len(b_layers)),
            l_params=l_params,
            l_param_map=[(0, 1, 'x', 'x'), (0, 'x')]
        )
    )

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
    if zero_last_layer_params:
        l_layers[-1].W.set_value(0*l_layers[-1].W.get_value())
        l_layers[-1].b.set_value(0*l_layers[-1].b.get_value())

    if mult_small_net_params:
        l_params = l_layers[-1].params

    b_layers.append(
        HiddenBlockLayer(
            (n_in, n_units_per),
            (1, n_out),
            l_layers[-2].top_active,
            one_block_idxs,
            batch_size,
            None,
            name='b_layer_' + str(len(b_layers)),
            l_params=l_params,
            l_param_map=[(0, 'x', 'x', 1), ('x', 0)]
        )
    )
    if zero_last_layer_params:
        b_layers[-1].W.set_value(0*b_layers[-1].W.get_value())
        b_layers[-1].b.set_value(0*b_layers[-1].b.get_value())


    if train_little_net or select_top_active:
        for layer in l_layers:
            print "\t%s" % layer

    if train_big_net:
        for layer in b_layers:
            print layer

    if restore_parameters:
        print "... Restoring weights of little model"
        restore_parameters(
            'parameters_20_20_l1_0.0001_l2_0.0001.pkl',
            l_layers
        )

    #for l_layer in l_layers:
    #    for param in l_layer.params:
    #        param.set_value(np.ones_like(param.get_value()))

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
            b_gparam = T.grad(
                b_cost,
                param,
                #consider_constant=[b_layers[i].in_idxs, b_layers[i].out_idxs]
            )
            b_velocity = shared(
                np.zeros_like(param.get_value(), dtype=theano.config.floatX),
                param.name + '_velocity'
            )
            b_param_updates.append(
                (b_velocity, momentum*b_velocity - b_learning_rate*b_gparam)
            )
            b_grads.append(b_gparam)
            b_param_updates.append((param, param + b_velocity))

        #if b_layers[i].l_params is not None:
            #for param in b_layers[i].l_params:
                #l_gparam = T.grad(
                #    b_cost,
                #    param
                #)
                #l_velocity = shared(
                #    np.zeros_like(param.get_value()),
                #    param.name + '_velocity'
                #)
                #b_param_updates.append((
                #    l_velocity, momentum*l_velocity - b_learning_rate*l_gparam
                #))
                #l_grads.append(l_gparam)
                #b_param_updates.append((param, param + l_velocity))
                #b_param_updates.append((
                #    param, param - 0.0001*l_gparam
                #))

    print "... Compiling little net train function"
    l_updates = []
    if select_top_active:
        l_updates = l_updates + top_active

    if train_little_net:
        l_updates = l_updates + l_param_updates

    l_train_model = function(
        [index],
        [l_cost, l_x, y],
        updates=l_updates,
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

    b_updates = []
    if train_big_net:
        b_updates = b_updates + b_param_updates

    b_train_model = function(
        [index],
        [b_cost],
        updates=b_updates,
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
    patience_increase = 10  # wait this much longer when a new best is
                           # found
    improvement_threshold = 0.995  # a relative improvement of this much is
                                   # considered significant
    validation_frequency = min(n_train_batches, patience / 2)
                                  # go through this many
                                  # minibatche before checking the network
                                  # on the validation set; in this case we
                                  # check every epoch

    best_params = None
    this_validation_loss = 0
    this_validation_loss_l = 0
    this_validation_loss_b = 0
    best_validation_loss = np.inf
    best_validation_loss_l = best_validation_loss
    best_validation_loss_b = best_validation_loss
    best_iter = 0
    test_score = 0.
    test_score_l = 0.
    test_score_b = 0.
    accum_l = 0
    accum_b = 0
    epoch = 0
    train_time_accum_l = 0
    train_time_accum_b = 0
    done_looping = False

    timers = ['train', 'valid', 'train']
    ts = TS(['epoch', 'valid'])
    ts_l = TS(timers)
    ts_b = TS(timers)

    summarize_rates()

    ts.start()
    while (epoch < n_epochs) and (not done_looping):
        epoch = epoch + 1
        ts.start('epoch')
        for minibatch_index in xrange(n_train_batches):
            if train_little_net or select_top_active:
                ts_l.start('train')
                minibatch_avg_cost_l = l_train_model(minibatch_index)
                ts_l.end('train')

                minibatch_avg_cost_l = minibatch_avg_cost_l[0]
                if np.isnan(minibatch_avg_cost_l):
                    print "minibatch_avg_cost_l: %f" % minibatch_avg_cost_l
                    ipdb.set_trace()
                accum_l = accum_l + minibatch_avg_cost_l

            if train_big_net:
                ts_b.start('train')
                minibatch_avg_cost_b = b_train_model(minibatch_index)
                ts_b.end('train')

                minibatch_avg_cost_b = minibatch_avg_cost_b[0]
                accum_b = accum_b + minibatch_avg_cost_b

            #print "minibatch_avg_cost: " + str(minibatch_avg_cost) + " minibatch_avg_cost_b: " + str(minibatch_avg_cost_b)
            #print l_layers[0].W.get_value().sum(), l_layers[1].W.get_value().sum(), b_layers[0].W.get_value().sum(), b_layers[1].W.get_value().sum()
            #print "A: ", np.max(np.abs(b_layers[0].W.get_value())), np.max(np.abs(b_layers[0].b.get_value())), np.max(np.abs(b_layers[1].W.get_value())), np.max(np.abs(b_layers[1].b.get_value()))
            #print "B: ", np.abs(b_layers[0].W.get_value()).sum(), np.abs(b_layers[0].b.get_value()).sum(), np.abs(b_layers[1].W.get_value()).sum(), np.abs(b_layers[1].b.get_value()).sum()
            #print "C: ", np.abs(np.array(minibatch_avg_cost_b[1])).sum(), np.abs(np.array(minibatch_avg_cost_b[2])).sum(), np.abs(np.array(minibatch_avg_cost_b[3])).sum(), np.abs(np.array(minibatch_avg_cost_b[4])).sum()

            # iteration number
            iter = (epoch - 1) * n_train_batches + minibatch_index

            if (iter + 1) % validation_frequency == 0:
                ts.end('epoch')
                ts.reset('epoch')

                l_summary = ""
                if train_little_net or select_top_active:
                    ts_l.reset('train')
                    accum_l = accum_l / validation_frequency
                    l_summary = ("minibatch_avg_cost_l: %f, time: %f"
                                 % (accum_l, ts_l.accumed['train'][-1][1]))
                    accum_l = 0
                    train_time_accum_l = 0

                b_summary = ""
                if train_big_net:
                    ts_b.reset('train')
                    accum_b = accum_b / validation_frequency
                    b_summary = ("minibatch_avg_cost_b: %f, time: %f"
                                 % (accum_b, ts_b.accumed['train'][-1][1]))
                    accum_b = 0

                print "%s %s" % (l_summary, b_summary)

                # compute zero-one loss on validation set
                summary = ('epoch %i, minibatch %i/%i'
                           % (epoch, minibatch_index + 1, n_train_batches))

                l_summary = ""
                if train_little_net or select_top_active:
                    validation_losses_l = [l_validate_model(i) for i
                                           in xrange(n_valid_batches)]
                    this_validation_loss_l = np.mean(validation_losses_l)
                    l_summary = ('little validation error %f %% '
                                 % (this_validation_loss_l * 100.))

                b_summary = ""
                if train_big_net:
                    validation_losses_b = [b_validate_model(i) for i
                                           in xrange(n_valid_batches)]
                    this_validation_loss_b = np.mean(validation_losses_b)
                    #this_validation_loss_b = 0
                    b_summary = ('big validation error %f %% '
                                 % (this_validation_loss_b * 100.))

                print ("%s %s %s" % (summary, l_summary, b_summary))
                #ipdb.set_trace()

                # if we got the best validation score until now
                if train_big_net:
                    this_validation_loss = this_validation_loss_b
                elif train_little_net:
                    this_validation_loss = this_validation_loss_l

                if this_validation_loss < best_validation_loss:
                    #improve patience if loss improvement is good enough
                    if this_validation_loss < best_validation_loss *  \
                       improvement_threshold:
                        patience = max(patience, iter * patience_increase)

                    best_validation_loss_l = this_validation_loss_l
                    best_validation_loss_b = this_validation_loss_b

                    if train_big_net:
                        best_validation_loss = best_validation_loss_b
                    elif train_little_net:
                        best_validation_loss = best_validation_loss_l

                    best_iter = iter

                    # test it on the test set
                    l_summary = ""
                    if train_little_net:
                        test_losses_l = [l_test_model(i) for i
                                         in xrange(n_test_batches)]
                        test_score_l = np.mean(test_losses_l)
                        l_summary = 'little: %f' % (test_score_l * 100.)

                    b_summary = ""
                    if train_big_net:
                        test_losses_b = [b_test_model(i) for i
                                         in xrange(n_test_batches)]
                        test_score_b = np.mean(test_losses_b)
                        #test_score_b = 0
                        b_summary = 'big: %f' % (test_score_b * 100.)

                    print ('     epoch %i, minibatch %i/%i,'
                           ' test error of best model %s %s'
                           % (epoch, minibatch_index + 1,
                              n_train_batches, l_summary, b_summary))

                learning_rate.update()

                if train_little_net:
                    l_learning_rate.set_value(learning_rate.rate)

                if train_big_net:
                    b_learning_rate.set_value(learning_rate.rate)

                momentum_rate.update()
                momentum.set_value(momentum_rate.rate)

                summarize_rates()

            if patience <= iter:
                    done_looping = True
                    break

    ts.end()
    print('Optimization complete. Best validation score of %f %% (%f %%) '
          'obtained at iteration %i, with test performance %f %% (%f %%)' %
          (best_validation_loss_l * 100., best_validation_loss_b * 100.,
           best_iter + 1, test_score_l * 100., test_score_b * 100.))
    print >> sys.stderr, ('The code for file ' +
                          os.path.split(__file__)[1] +
                          ' ran for %s' % ts)

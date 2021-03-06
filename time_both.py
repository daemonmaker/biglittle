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
from layer import HiddenLayer, HiddenBlockLayer
from timing_stats import TimingStats as TS


def test_train_little_only(
        rng,
        batch_size,
        learning_rate,
        n_hids,
        n_epochs=1000,
        L1_reg=0.0,
        L2_reg=0.0001,
        zero_last_layer_params=False,
):
    def summarize_rates():
        print "Learning rate: ", learning_rate.rate

    l_learning_rate = shared(
        np.array(learning_rate.rate, dtype=config.floatX),
        name='learning_rate'
    )

    index = T.lscalar('index')
    l_x = T.matrix('l_x', dtype=config.floatX)
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
    n_out = n_hids
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

    n_in = n_out
#    l_layers.append(
#        HiddenLayer(
#            n_in,
#            n_out,
#            batch_size,
#            k=0.1,
#            activation=T.tanh,
#            name='l_layer_' + str(len(l_layers))
#        )
#    )

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

    for layer in l_layers:
        print "\t%s" % layer

    #for l_layer in l_layers:
    #    for param in l_layer.params:
    #        param.set_value(np.ones_like(param.get_value()))

    print "... Building top active updates"
    top_active = []
    l_activation = l_x
    for i in range(len(l_layers)):
        l_activation = l_layers[i].output(l_activation)

    print "... Building costs and errors"
    l_cost = add_regularization(
        l_layers,
        l_layers[-1].cost(l_activation, y),
        L1_reg,
        L2_reg
    )
    l_error = l_layers[-1].error(l_activation, y)

    print "... Building parameter updates"
    l_grads = []
    l_param_updates = []
    for i in range(len(l_layers)):
        for param in l_layers[i].params:
            gparam = T.grad(l_cost, param)
            l_grads.append(gparam)
            l_param_updates.append((param, param - l_learning_rate*gparam))

    print "... Compiling little net train function"
    l_updates = l_param_updates

    l_train_model = function(
        [index],
        [l_cost, l_x, y],
        updates=l_updates,
        givens={
            l_x: train_set_x[index*batch_size:(index+1)*batch_size],
            y: train_set_y[index*batch_size:(index+1)*batch_size]
        }
    )

    print "... Compiling little net test function"
    l_test_model = function(
        [index],
        l_error,
        givens={
            l_x: test_set_x[index*batch_size:(index+1)*batch_size],
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
    accum_l = 0
    epoch = 0
    train_time_accum_l = 0
    done_looping = False

    timers = ['train', 'valid', 'train']
    ts = TS(['epoch', 'valid'])
    ts_l = TS(timers)

    summarize_rates()

    ts.start()
    while (epoch < n_epochs) and (not done_looping):
        epoch = epoch + 1
        ts.start('epoch')
        for minibatch_index in xrange(n_train_batches):
            ts_l.start('train')
            minibatch_avg_cost_l = l_train_model(minibatch_index)
            ts_l.end('train')

            minibatch_avg_cost_l = minibatch_avg_cost_l[0]
            if np.isnan(minibatch_avg_cost_l):
                print "minibatch_avg_cost_l: %f" % minibatch_avg_cost_l
                ipdb.set_trace()
            accum_l = accum_l + minibatch_avg_cost_l

            # iteration number
            iter = (epoch - 1) * n_train_batches + minibatch_index

            if (iter + 1) % validation_frequency == 0:
                ts.end('epoch')
                ts.reset('epoch')

                ts_l.reset('train')
                accum_l = accum_l / validation_frequency
                l_summary = ("minibatch_avg_cost_l: %f, time: %f"
                             % (accum_l, ts_l.accumed['train'][-1][1]))
                accum_l = 0
                train_time_accum_l = 0

                print "%s" % (l_summary)

                # compute zero-one loss on validation set
                summary = ('epoch %i, minibatch %i/%i'
                           % (epoch, minibatch_index + 1, n_train_batches))

                validation_losses_l = [l_validate_model(i) for i
                                       in xrange(n_valid_batches)]
                this_validation_loss_l = np.mean(validation_losses_l)
                l_summary = ('little validation error %f %% '
                             % (this_validation_loss_l * 100.))

                print ("%s %s" % (summary, l_summary))
                #ipdb.set_trace()

                # if we got the best validation score until now
                this_validation_loss = this_validation_loss_l

                if this_validation_loss < best_validation_loss:
                    #improve patience if loss improvement is good enough
                    if this_validation_loss < best_validation_loss *  \
                       improvement_threshold:
                        patience = max(patience, iter * patience_increase)

                    best_validation_loss_l = this_validation_loss_l
                    best_validation_loss = best_validation_loss_l

                    best_iter = iter

                    # test it on the test set
                    test_losses_l = [l_test_model(i) for i
                                     in xrange(n_test_batches)]
                    test_score_l = np.mean(test_losses_l)
                    l_summary = 'little: %f' % (test_score_l * 100.)

                    print ('     epoch %i, minibatch %i/%i,'
                           ' test error of best model %s'
                           % (epoch, minibatch_index + 1,
                              n_train_batches, l_summary))

                learning_rate.update()

                l_learning_rate.set_value(learning_rate.rate)

                summarize_rates()

            if patience <= iter:
                    done_looping = True
                    break

    ts.end()
    print('Optimization complete. Best validation score of %f %% '
          'obtained at iteration %i, with test performance %f %%' %
          (best_validation_loss_l * 100., best_iter + 1, test_score_l * 100.))
    print >> sys.stderr, ('The code for file ' +
                          os.path.split(__file__)[1] +
                          ' ran for %s' % ts)

    return ts.diffs['epoch']


def test_train_big_only(
        rng,
        batch_size,
        learning_rate,
        n_hids,
        n_epochs=1000,
        L1_reg=0.0,
        L2_reg=0.0001,
        zero_last_layer_params=False,
):
    def summarize_rates():
        print "Learning rate: ", learning_rate.rate

    b_learning_rate = shared(
        np.array(learning_rate.rate, dtype=config.floatX),
        name='learning_rate'
    )

    index = T.lscalar('index')
    b_x = T.tensor3('b_x', dtype=config.floatX)
    y = T.ivector('y')

    print "Loading Data"
    print "... MNIST"
    dataset = 'mnist.pkl.gz'
    datasets = load_data(dataset, True)

    train_set_x, train_set_y = datasets[0]
    valid_set_x, valid_set_y = datasets[1]
    test_set_x, test_set_y = datasets[2]

    n_train_batches = train_set_x.get_value(borrow=True).shape[0] / batch_size
    n_valid_batches = valid_set_x.get_value(borrow=True).shape[0] / batch_size
    n_test_batches = test_set_x.get_value(borrow=True).shape[0] / batch_size

    print "Building models"
    print "... Building layers"
    # Create network structure
    x_size = train_set_x.shape[2].eval()
    y_size = train_set_y.shape[0].eval()
    n_in = x_size
    n_units_per = 20
    n_out = n_hids
    b_layers = []
    l_params = None

    k_per = 0.05
    k = int(n_out*k_per)
    print "k_per: %d, k: %d" % (k_per, k)

    # Shared variable used for always activating one block in a layer as in the
    # input and output layer
    one_block_idxs = shared(
        np.zeros((batch_size, 1), dtype='int64'),
        name='one_block_idxs'
    )

    top_actives = []
    top_actives.append(shared(
        np.repeat(
            np.arange(k).reshape(1, k),
            batch_size,
            axis=0
        ),
        name="top_actives_%d" % len(top_actives)
    ))

    b_layers.append(
        HiddenBlockLayer(
            (1, x_size),
            (n_out, n_units_per),
            one_block_idxs,
            top_actives[-1],
            batch_size,
            activation=T.tanh,
            name='b_layer_' + str(len(b_layers)),
            l_params=l_params,
            l_param_map=[('x', 1, 0, 'x'), (0, 'x')]
        )
    )

    n_in = n_out
#    top_actives.append(shared(
#        np.repeat(
#            np.arange(k).reshape(1, k),
#            batch_size,
#            axis=0
#        ),
#        name="top_actives_%d" % len(top_actives)
#    ))
#
#    b_layers.append(
#        HiddenBlockLayer(
#            (n_in, n_units_per),
#            (n_out, n_units_per),
#            top_actives[-2],
#            top_actives[-1],
#            #out_idxs_n,
#            batch_size,
#            activation=T.tanh,
#            name='b_layer_' + str(len(b_layers)),
#            l_params=l_params,
#            l_param_map=[(0, 1, 'x', 'x'), (0, 'x')]
#        )
#    )

    n_out = 10
    top_actives.append(shared(
        np.repeat(
            np.arange(k).reshape(1, k),
            batch_size,
            axis=0
        ),
        name="top_actives_%d" % len(top_actives)
    ))

    b_layers.append(
        HiddenBlockLayer(
            (n_in, n_units_per),
            (1, n_out),
            top_actives[-2],
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


    for layer in b_layers:
        print layer

    print "... Building top active updates"
    top_active = []
    b_activation = b_x
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
    b_grads = []
    b_param_updates = []
    for i in range(len(b_layers)):
        for param in b_layers[i].params:
            b_gparam = T.grad(
                b_cost,
                param,
                #consider_constant=[b_layers[i].in_idxs, b_layers[i].out_idxs]
            )
            b_grads.append(b_gparam)
            b_param_updates.append((param, param - b_learning_rate*b_gparam))

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

    print "... Compiling big net train function"
    b_updates = b_param_updates

    b_train_model = function(
        [index],
        [b_cost],
        updates=b_updates,
        givens={
            b_x: train_set_x[index*batch_size:(index+1)*batch_size],
            y: train_set_y[index*batch_size:(index+1)*batch_size]
        }
    )

    print "... Compiling big net test function"
    b_test_model = function(
        [index],
        b_error,
        givens={
            b_x: test_set_x[index*batch_size:(index+1)*batch_size],
            y: test_set_y[index*batch_size:(index+1)*batch_size]
        }
    )

    print "... Compiling big net validate function"
    b_validate_model = function(
        [index],
        b_error,
        givens={
            b_x: valid_set_x[index*batch_size:(index+1)*batch_size],
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
        for minibatch_index in xrange(n_train_batches):
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

                ts_b.reset('train')
                accum_b = accum_b / validation_frequency
                b_summary = ("minibatch_avg_cost_b: %f, time: %f"
                             % (accum_b, ts_b.accumed['train'][-1][1]))
                accum_b = 0

                print "%s" % (b_summary)

                # compute zero-one loss on validation set
                summary = ('epoch %i, minibatch %i/%i'
                           % (epoch, minibatch_index + 1, n_train_batches))

                validation_losses_b = [b_validate_model(i) for i
                                       in xrange(n_valid_batches)]
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
                    test_losses_b = [b_test_model(i) for i
                                     in xrange(n_test_batches)]
                    test_score_b = np.mean(test_losses_b)
                    #test_score_b = 0
                    b_summary = 'big: %f' % (test_score_b * 100.)

                    print ('     epoch %i, minibatch %i/%i,'
                           ' test error of best model %s'
                           % (epoch, minibatch_index + 1,
                              n_train_batches, b_summary))

                learning_rate.update()

                b_learning_rate.set_value(learning_rate.rate)

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
    rng = np.random.RandomState()
    batch_size = 1
    n_epochs = 1
    learning_rate = LinearChangeRate(0.011, -0.01, 0.01, 'learning_rate')
    n_hids_len = 5
    n_hids = (pow(10, y) for y in range(2, 2+n_hids_len))

    epoch_times = np.zeros((n_hids_len, 2))
    counter = 0
    for n_hid in n_hids:
        try:
            l_epoch_time = test_train_little_only(
                rng,
                batch_size=batch_size,
                learning_rate=learning_rate,
                n_hids=n_hid,
                n_epochs=n_epochs,
                #L1_reg=0.0001,
                L2_reg=0.0001,
            )
        except MemoryError:
            l_epoch_time = -1

        try:
            b_epoch_time = test_train_big_only(
                rng,
                batch_size=batch_size,
                learning_rate=learning_rate,
                n_hids=n_hid,
                n_epochs=n_epochs,
                #L1_reg=0.0001,
                L2_reg=0.0001,
            )
        except MemoryError:
            b_epoch_time = -1

        print "l_epoch_time: %f, b_epoch_time: %f" % (l_epoch_time, b_epoch_time)
        epoch_times[counter, :] = [l_epoch_time, b_epoch_time]
        counter = counter + 1

    print epoch_times
    ipdb.set_trace()

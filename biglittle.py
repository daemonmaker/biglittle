import ipdb
import time
import numpy as np
import sys
import os

from logistic_sgd import load_data

from theano import function
from theano import tensor as T
from theano import config
from theano import shared

from layer import HiddenLayer, HiddenBlockLayer


def test_biglittle():
    batch_size = 1
    l_learning_rate = 0.01
    b_learning_rate = 0.01
    n_epochs = 1000

    index = T.lscalar('index')
    l_x = T.matrix('l_x', dtype=config.floatX)
    b_x = T.tensor3('b_x', dtype=config.floatX)
    y = T.ivector('y')
    #in_idxs = T.lmatrix('iIdx_' + str(self.name))
    #out_idxs = T.lmatrix('oIdx_' + str(self.name))

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
    # Create network structure
    l_layers = []
    b_layers = []
    x_size = 28*28
    n_in = x_size
    n_units_per_in = 1
    n_out = 500
    n_units_per_out = 1

    l_layers.append(
        HiddenLayer(
            n_in,
            n_out,
            batch_size,
            #k=0.045,
            k=1,
            name='l_layer_' + str(len(l_layers))
        )
    )

    in_idxs_0 = shared(
        np.ones((batch_size, 1), dtype='int64'),
        name='in_idxs_0'
    )
    b_layers.append(
        HiddenBlockLayer(
            (1, x_size),
            (n_out, n_units_per_out),
            in_idxs_0,
            l_layers[-1].top_active,
            batch_size,
            name='l_layer_' + str(len(l_layers))
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
            #k=0.3,
            k=1,
            activation=T.nnet.softmax,
            name='l_layer_' + str(len(l_layers))
        )
    )

    # T.nnet.softmax takes a matrix not a tensor so just calculate the linear
    # component in the layer and apply the softmax later
    b_layers.append(HiddenBlockLayer(
        (n_in, n_units_per_in),
        (n_out, n_units_per_out),
        l_layers[-2].top_active,
        l_layers[-1].top_active,
        batch_size,
        None
    ))

    print "... Building top active updates"
    top_active = []
    l_activation = l_x
    b_activation = b_x
    for i in range(len(l_layers)):
        l_activation = l_layers[i].output(l_activation)
        b_activation = b_layers[i].output(b_activation)
        #top_active.append((
        #    l_layers[i].top_active,
        #    T.cast(T.argsort(l_activation)[:, :l_layers[i].k], 'int64')
        #))

    print "... Building costs and errors"
    l_cost = l_layers[-1].cost(l_activation, y)
    for layer in l_layers:  # Add L2 penalty on weights
        l_cost = l_cost + 0.0001*(layer.W**2).sum()
    l_error = l_layers[-1].error(l_activation, y)

    # T.nnet.softmax takes a matrix not a tensor so we only calculate the
    # linear component at the last layer and here we reshape and then
    # apply the softmax
    b_activation = T.nnet.softmax(T.mean((b_activation*b_activation), axis=2))
    b_cost = b_layers[-1].cost(b_activation, y)
    for layer in b_layers:  # Add L2 penalty on weights
        b_cost = b_cost + 0.0001*(layer.W**2).sum()
    b_error = b_layers[-1].error(b_activation, y)

    print "... Building parameter updates"
    l_param_updates = []
    b_param_updates = []
    for i in range(len(l_layers)):
        for param in l_layers[i].params:
            gparam = T.grad(l_cost, param)
            l_param_updates.append((param, param - l_learning_rate*gparam))

        for param in b_layers[i].params:
            gparam = T.grad(b_cost, param)
            b_param_updates.append((param, param - b_learning_rate*gparam))

    print "... Compiling little net train function"
    l_train_model = function(
        [index],
        [l_layers[0].output(l_x), l_cost],
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
        [b_layers[0].output(b_x), b_cost],
        updates=b_param_updates,
        givens={
            b_x: train_set_x_b[index*batch_size:(index+1)*batch_size],
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
    patience_increase = 2  # wait this much longer when a new best is
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

    while (epoch < n_epochs) and (not done_looping):
        epoch = epoch + 1
        for minibatch_index in xrange(n_train_batches):

            a, minibatch_avg_cost = l_train_model(minibatch_index)
            b, minibatch_avg_cost_b = b_train_model(minibatch_index)
            print "minibatch_avg_cost: " + str(minibatch_avg_cost) + " minibatch_avg_cost_b: " + str(minibatch_avg_cost_b)
            ipdb.set_trace()

            # iteration number
            iter = (epoch - 1) * n_train_batches + minibatch_index

            if (iter + 1) % validation_frequency == 0:
                # compute zero-one loss on validation set
                validation_losses = [l_validate_model(i) for i
                                     in xrange(n_valid_batches)]
                this_validation_loss = np.mean(validation_losses)

                validation_losses_b = [b_validate_model(i) for i
                                       in xrange(n_valid_batches)]
                this_validation_loss_b = np.mean(validation_losses_b)

                print('epoch %i, minibatch %i/%i, validation error %f %% '
                      '(%f %%)' %
                     (epoch, minibatch_index + 1, n_train_batches,
                      this_validation_loss * 100.,
                      this_validation_loss_b * 100.))

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

test_biglittle()

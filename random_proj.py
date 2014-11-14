#! /usr/bin/env python

import ipdb
import time
from datetime import datetime
import numpy as np
import sys
import os
import cPickle as pkl
from itertools import product

import theano
from theano import function
from theano import tensor as T
from theano import config
from theano import shared
from theano.tensor.shared_randomstreams import RandomStreams

from utils import *
from layer import HiddenLayer, HiddenRandomBlockLayer
from timing_stats import TimingStats as TS


class Experiments(object):
    def __init__(self, input_dim, num_classes):
        self.input_dim = input_dim
        self.num_classes = num_classes

        # Shared variable used for always activating one block in a layer
        # as in the input and output layer
        self.one_block_idxs = shared(
            np.zeros((batch_size, 1), dtype='int64'),
            name='one_block_idxs'
        )

        self.experiments = {}
        self.results = {}

    def get(self, idx):
        return self.experiments[idx]

    def add(self, idx, params):
        n_hids = params['n_hids']
        n_units_per = params['n_units_per']
        k_pers = params['k_pers']
        activations = params['activations']

        new_exp = []
        new_exp.append({
            'n_in': self.input_dim,
            'n_hids': n_hids[0],
            'n_units_per': n_units_per,
            'in_idxs': self.one_block_idxs,
            'k': k_pers[0],
            'activation': activations[0]
        })
        for i in range(1, len(k_pers)):
            new_exp.append({
                'n_in': n_hids[i-1],
                'n_hids': n_hids[i],
                'n_units_per': n_units_per,
                'k': k_pers[i],
                'activation': activations[i]
            })
        new_exp.append({
            'n_in': n_hids[-1],
            'n_hids': self.num_classes,
            'n_units_per': n_units_per,
            'out_idxs': self.one_block_idxs,
            'k': 1,
            'activation': activations[-1]
        })

        self.experiments[idx] = new_exp
        self.results[idx] = {}

    def save(self, exp_id, k, v):
        self.results[exp_id][k] = v


class MNIST():
    def __init__(self, reshape_data=False):
        dataset = 'mnist.pkl.gz'
        datasets = load_data(dataset, reshape_data)

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


class Model(object):
    def __init__(
            self,
            parameters,
            batch_size,
            learning_rate,
            L1_reg=0.0,
            L2_reg=0.0001,
            zero_last_layer_params=False
    ):
        self.parameters = parameters

        self.index = T.lscalar('index')
        self.y = T.ivector('y')

        assert(batch_size > 0)
        self.batch_size = batch_size

        self.learning_rate = learning_rate

        assert(L1_reg >= 0.)
        self.L1_reg = L1_reg

        assert(L2_reg >= 0.)
        self.L2_reg = L2_reg

        print "... Building layers"
        self.layers = self.build_layers()

        if zero_last_layer_params:
            layers[-1].W.set_value(0*layers[-1].W.get_value())
            layers[-1].b.set_value(0*layers[-1].b.get_value())

        # Summarize layers
        for layer in self.layers:
            print layer

    def build_layers(self):
        raise NotImplementedError('build_layers')

    def calculate_activation(self):
        raise NotImplementedError('calculate_activation')

    def convert_data(self):
        raise NotImplementedError('convert_data')

    def build_functions(self):
        layers = self.layers
        input = self.input
        activation = self.calculate_activation()
        index = self.index
        y = self.y
        L1_reg = self.L1_reg
        L2_reg = self.L2_reg

        print "... Building costs"
        cost = add_regularization(
            layers,
            layers[-1].cost(activation, y),
            L1_reg,
            L2_reg
        )

        print "... Building errors"
        error = layers[-1].error(activation, y)

        print "... Building parameter updates"
        consider_constants = []
        for layer in layers:
            if hasattr(layer, 'in_idxs') and hasattr(layer, 'out_idxs'):
                consider_constants += [layer.in_idxs, layer.out_idxs]

        grads = []
        param_updates = []
        for i in range(len(layers)):
            for param in layers[i].params:
                gparam = T.grad(
                    cost,
                    param,
                    consider_constant=consider_constants
                )
                grads.append(gparam)
                param_updates.append(
                    (param, param - self.learning_rate*gparam)
                )

        print "... Compiling train function"
        updates = param_updates

        train_model = function(
            [index],
            [cost],
            updates=updates,
            givens={
                input: data.train_set_x[index*batch_size:(index+1)*batch_size],
                y: data.train_set_y[index*batch_size:(index+1)*batch_size]
            }
        )

        print "... Compiling test function"
        test_model = function(
            [index],
            error,
            givens={
                input: data.test_set_x[index*batch_size:(index+1)*batch_size],
                y: data.test_set_y[index*batch_size:(index+1)*batch_size]
            }
        )

        print "... Compiling validate function"
        validate_model = function(
            [index],
            error,
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


class SparseBlockModel(Model):
    reshape_data = True

    def __init__(
            self,
            data,
            parameters,
            batch_size,
            learning_rate,
            L1_reg=0.0,
            L2_reg=0.0001,
            zero_last_layer_params=False
    ):
        self.input = T.tensor3('input', dtype=config.floatX)

        super(SparseBlockModel, self).__init__(
            parameters,
            batch_size,
            learning_rate,
            L1_reg,
            L2_reg,
            zero_last_layer_params
        )

    def build_layers(self):
        layers = []
        activation = self.input
        num_layers = len(self.parameters)
        for i, params in enumerate(self.parameters):
            constructor_params = {
                'x': activation,
                'n_in': (params['n_in'], params['n_units_per']),
                'n_out': (params['n_hids'], params['n_units_per']),
                'batch_size': batch_size,
                'k': params['k'],
                'activation': params['activation'],
                'name': 'b_layer_%d' % i
            }
            if i == 0:
                constructor_params['n_in'] = (1, params['n_in'])
            elif i == (num_layers - 1):
                constructor_params['n_out'] = (1, params['n_hids'])

            # The input indices should either be specified as in the first and
            # last layers or be the output indices of the previous layer.
            if 'in_idxs' in params.keys():
                constructor_params['in_idxs'] = params['in_idxs']
            else:
                constructor_params['in_idxs'] = layers[-1].out_idxs

            if 'out_idxs' in params.keys():
                constructor_params['out_idxs'] = params['out_idxs']

            layers.append(HiddenRandomBlockLayer(**constructor_params))
            activation = layers[-1].output(activation)

        return layers

    def calculate_activation(self):
        print "... Calculating activation"
        top_active = []
        activation = self.input
        for i in range(len(self.layers)):
            activation = self.layers[i].output(activation)
            #top_active.append((
            #    top_actives[i],
            #    T.argsort(T.abs_(l_activation))[:, :l_layers[i].k]
            #))
        self.top_active = top_active

        # T.nnet.softmax takes a matrix not a tensor so we only calculate the
        # linear component at the last layer and here we reshape and then
        # apply the softmax
        #activation = T.nnet.softmax(((activation*activation)**2).sum(axis=2))
        #activation = relu_softmax(((activation*activation)**2).sum(axis=2))
        #activation = T.nnet.softmax(T.mean(activation, axis=2))
        #activation = relu_softmax(T.mean(activation, axis=2))
        #activation = T.nnet.softmax(T.max(activation, axis=2))
        #activation = relu_softmax(T.max(activation, axis=2))
        shp = activation.shape
        #activation = relu_softmax(activation.reshape((shp[0], shp[2])))
        return T.nnet.softmax(activation.reshape((shp[0], shp[2])))


class MLPModel(Model):
    reshape_data = False

    def __init__(
            self,
            data,
            parameters,
            batch_size,
            learning_rate,
            L1_reg=0.0,
            L2_reg=0.0001,
            zero_last_layer_params=False
    ):
        self.input = T.matrix('input', dtype=config.floatX)
        self.num_layers = len(parameters)

        super(MLPModel, self).__init__(
            parameters,
            batch_size,
            learning_rate,
            L1_reg,
            L2_reg,
            zero_last_layer_params
        )

    def _calc_num_units(self, layer_idx, params):
        units = [
            params['n_in']*params['n_units_per'],
            params['n_hids']*params['n_units_per']
        ]
        if layer_idx == 0:
            units[0] = params['n_in']
        elif layer_idx == (self.num_layers - 1):
            units[1] = params['n_hids']
        return tuple(units)

    def build_layers(self):
        layers = []
        for i, params in enumerate(self.parameters):
            units = self._calc_num_units(i, params)
            constructor_params = {
                'n_in': units[0],
                'n_out': units[1],
                'batch_size': batch_size,
                'k': params['k'],
                'activation': params['activation'],
                'name': 'l_layer_%d' % i
            }
            layers.append(HiddenLayer(**constructor_params))

        return layers

    def calculate_activation(self):
        activation = self.input
        for layer in self.layers:
            activation = layer.output(activation)
        return T.nnet.softmax(activation)


class EqualParametersModel(MLPModel):
    pass


class EqualComputationsModel(MLPModel):
    def _calc_num_units(self, layer_idx, params):
        units = super(EqualComputationsModel)._calc_num_units(
            layer_idx,
            params
        )
        return (units[0]*params['k'], units[1]*params['k'])


def train(
        model,
        train_model,
        test_model,
        validate_model,
        learning_rate,
        shared_learning_rate,
        n_epochs=1000
):
    def summarize_rates():
        print "Learning rate: ", learning_rate.rate

    # early-stopping parameters
    patience = 10000  # look as this many examples regardless
    patience_increase = 100  # wait this much longer when a new best is
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
    best_validation_loss = np.inf
    best_iter = 0
    test_score = 0.
    accum = 0
    epoch = 0
    done_looping = False

    ts = TS(['train', 'epoch', 'valid'])

    summarize_rates()

    ts.start()
    while (epoch < n_epochs) and (not done_looping):
        epoch = epoch + 1
        ts.start('epoch')
        for minibatch_index in xrange(data.n_train_batches):
            ts.start('train')
            minibatch_avg_cost = train_model(minibatch_index)
            ts.end('train')
            #print "0: ", model.layers[-5].in_idxs.get_value()
            #print "1: ", model.layers[-4].in_idxs.get_value()
            #print "2: ", model.layers[-3].in_idxs.get_value()
            #print "3: ", model.layers[-2].in_idxs.get_value()
            #print "4: ", model.layers[-1].in_idxs.get_value()

            minibatch_avg_cost = minibatch_avg_cost[0]
            accum = accum + minibatch_avg_cost

            # print (
            #     "minibatch_avg_cost: " + str(minibatch_avg_cost)
            #     + " minibatch_avg_cost: " + str(minibatch_avg_cost)
            # )
            # print (
            #     l_layers[0].W.get_value().sum()
            #     + ' ' + l_layers[1].W.get_value().sum()
            #     + ' '
            #     + layers[0].W.get_value().sum()
            #     + ' ' + layers[1].W.get_value().sum()
            # )
            # print (
            #     "A: " + np.max(np.abs(layers[0].W.get_value()))
            #     + ' ' + np.max(np.abs(layers[0].b.get_value()))
            #     + ' ' + np.max(np.abs(layers[1].W.get_value()))
            #     + ' ' + np.max(np.abs(layers[1].b.get_value()))
            # )
            # print (
            #     "B: " + np.abs(layers[0].W.get_value()).sum()
            #     + ' ' + np.abs(layers[0].b.get_value()).sum()
            #     + ' ' + np.abs(layers[1].W.get_value()).sum()
            #     + ' ' + np.abs(layers[1].b.get_value()).sum()
            # )
            # print (
            #     "C: " + np.abs(np.array(minibatch_avg_cost[1])).sum()
            #     + ' ' + np.abs(np.array(minibatch_avg_cost[2])).sum()
            #     + ' ' + np.abs(np.array(minibatch_avg_cost[3])).sum()
            #     + ' ' + np.abs(np.array(minibatch_avg_cost[4])).sum()
            # )

            # iteration number
            iter = (epoch - 1) * data.n_train_batches + minibatch_index

            if (iter + 1) % validation_frequency == 0:
                ts.end('epoch')
                ts.reset('epoch')

                ts.reset('train')
                accum = accum / validation_frequency
                summary = ("minibatch_avg_cost: %f, time: %f"
                           % (accum, ts.accumed['train'][-1][1]))
                accum = 0

                print "%s" % (summary)

                # compute zero-one loss on validation set
                summary = (
                    'epoch %i, minibatch %i/%i'
                    % (
                        epoch, minibatch_index + 1, data.n_train_batches
                    )
                )

                validation_losses = [validate_model(i) for i
                                     in xrange(data.n_valid_batches)]
                this_validation_loss = np.mean(validation_losses)
                #this_validation_loss = 0
                summary = ('big validation error %f %% '
                           % (this_validation_loss * 100.))

                print ("%s %s" % (summary, summary))
                #ipdb.set_trace()

                # if we got the best validation score until now
                this_validation_loss = this_validation_loss

                if this_validation_loss < best_validation_loss:
                    #improve patience if loss improvement is good enough
                    if this_validation_loss < best_validation_loss *  \
                       improvement_threshold:
                        patience = max(patience, iter * patience_increase)

                    best_validation_loss = this_validation_loss

                    best_iter = iter

                    # test it on the test set
                    test_losses = [test_model(i) for i
                                   in xrange(data.n_test_batches)]
                    test_score = np.mean(test_losses)
                    #test_score = 0
                    summary = 'big: %f' % (test_score * 100.)

                    print ('     epoch %i, minibatch %i/%i,'
                           ' test error of best model %s'
                           % (epoch, minibatch_index + 1,
                              data.n_train_batches, summary))

                learning_rate.update()

                shared_learning_rate.set_value(learning_rate.rate)

                summarize_rates()

            if patience <= iter:
                    done_looping = True
                    break

    ts.end()
    print('Optimization complete. Best validation score of %f %% '
          'obtained at iteration %i, with test performance %f %%' %
          (best_validation_loss * 100., best_iter + 1, test_score * 100.))
    print >> sys.stderr, ('The code for file ' +
                          os.path.split(__file__)[1] +
                          ' ran for %s' % ts)

    return ts.diffs['epoch']


if __name__ == '__main__':
    model_class = EqualParametersModel

    rng = np.random.RandomState()
    batch_size = 32
    n_epochs = 100000
    #learning_rate = LinearChangeRate(0.5, -0.01, 0.01, 'learning_rate')
    learning_rate = LinearChangeRate(0.21, -0.01, 0.2, 'learning_rate')
    #n_hids = (pow(10, y) for y in range(2, 2+n_hids_len))
    experiments = {
        0: {
            'n_hids': (25,),
            'n_units_per': 20,
            'k_pers': (1.,),
            'activations': (T.tanh, None)
        },
        1: {
            'n_hids': (25, 25),
            'n_units_per': 20,
            'k_pers': (1., 0.5),
            'activations': (T.tanh, T.tanh, None)
        },
        2: {
            'n_hids': (25, 100, 25),
            'n_units_per': 20,
            'k_pers': (1., 0.25, 1),
            'activations': (T.tanh, T.tanh, T.tanh, None)
        },
        3: {
            'n_hids': (50, 500, 10),
            'n_units_per': 20,
            'k_pers': (0.9, 0.05, 1),
            'activations': (T.tanh, T.tanh, T.tanh, None)
        },
        4: {
            'n_hids': (50, 500, 500, 10),
            'n_units_per': 20,
            'k_pers': (1, 0.05, 0.05, 1),
            'activations': (T.tanh, T.tanh, T.tanh, T.tanh, None)
        },
        5: {
            'n_hids': (50, 75, 100, 75, 50),
            'n_units_per': 32,
            'k_pers': (1., 0.1, 0.05, 0.1, 1),
            'activations': (T.tanh, T.tanh, T.tanh, T.tanh, T.tanh, None)
        }
    }

    print "Loading Data"
    print "... MNIST"
    data = MNIST(model_class.reshape_data)

    exps = Experiments(
        input_dim=data.train_set_x.shape[-1].eval(),
        num_classes=10
    )

    exps_to_run = [4]

    for idx in exps_to_run:
        exps.add(idx, experiments[idx])
        try:
            shared_learning_rate = shared(
                np.array(learning_rate.rate, dtype=config.floatX),
                name='learning_rate'
            )

            print "Building model: %s" % str(model_class)
            model = model_class(
                data=data,
                parameters=exps.get(idx),
                batch_size=batch_size,
                learning_rate=shared_learning_rate,
                #L1_reg=0.0001,
                L2_reg=0.0001
            )

            print "Training"
            epoch_time = train(
                model,
                learning_rate=learning_rate,
                shared_learning_rate=shared_learning_rate,
                n_epochs=n_epochs,
                **model.build_functions()
            )
        except MemoryError:
            epoch_time = -1

        print "epoch_time: %f" % (epoch_time)
        exps.save(idx, 'epoch_time', epoch_time)

    pkl.dump(exps, open('random_proj_experiments.pkl', 'wb'))

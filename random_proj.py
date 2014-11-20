#! /usr/bin/env python

import ipdb
import time
from datetime import datetime
import numpy as np
import sys
import os
import cPickle as pkl
from itertools import product
import gc

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

        self.experiments = {}
        self.results = {}

    def get_layers_definition(self, idx):
        layers_description = self.experiments[idx]['layers_description']
        parameters = self.experiments[idx]['parameters']

        # Shared variable used for always activating one block in a layer
        # as in the input and output layer
        self.one_block_idxs = shared(
            np.zeros((parameters['batch_size'], 1), dtype='int64'),
            name='one_block_idxs'
        )

        n_hids = layers_description['n_hids']
        n_units_per = layers_description['n_units_per']
        k_pers = layers_description['k_pers']
        activations = layers_description['activations']

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

        return new_exp

    def get_parameters(self, idx):
        return self.parameters[idx]

    def add(self, idx, layers_description, parameters):
        self.experiments[idx] = {
            'layers_description': layers_description,
            'parameters': parameters
        }
        if idx not in self.results.keys():
            self.results[idx] = {}

    def save(self, exp_id, model_name, k, v):
        if model_name not in self.results[exp_id].keys():
            self.results[exp_id][model_name] = {}
        self.results[exp_id][model_name][k] = v


class MNIST():
    def __init__(self, batch_size, reshape_data=False):
        assert(batch_size > 0)
        self.batch_size = batch_size

        self.reshape_data = reshape_data

        dataset = 'mnist.pkl.gz'
        datasets = load_data(dataset, reshape_data)

        self.train_set_x, self.train_set_y = datasets[0]
        self.valid_set_x, self.valid_set_y = datasets[1]
        self.test_set_x, self.test_set_y = datasets[2]

        self.n_train_batches = self.train_set_x.get_value(
            borrow=True
        ).shape[0] / self.batch_size
        self.n_valid_batches = self.valid_set_x.get_value(
            borrow=True
        ).shape[0] / self.batch_size
        self.n_test_batches = self.test_set_x.get_value(
            borrow=True
        ).shape[0] / self.batch_size


class Model(object):
    def __init__(
            self,
            data,
            layer_descriptions,
            batch_size,
            learning_rate,
            L1_reg=0.0,
            L2_reg=0.0001,
            zero_last_layer_params=False
    ):
        self.data = data

        self.layer_descriptions = layer_descriptions

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
                input: self.data.train_set_x[
                    index*self.batch_size:(index+1)*self.batch_size
                ],
                y: self.data.train_set_y[
                    index*self.batch_size:(index+1)*self.batch_size
                ]
            }
        )

        print "... Compiling test function"
        test_model = function(
            [index],
            error,
            givens={
                input: self.data.test_set_x[
                    index*self.batch_size:(index+1)*self.batch_size
                ],
                y: self.data.test_set_y[
                    index*self.batch_size:(index+1)*self.batch_size
                ]
            }
        )

        print "... Compiling validate function"
        validate_model = function(
            [index],
            error,
            givens={
                input: self.data.valid_set_x[
                    index*self.batch_size:(index+1)*self.batch_size
                ],
                y: self.data.valid_set_y[
                    index*self.batch_size:(index+1)*self.batch_size
                ]
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
            layer_descriptions,
            batch_size,
            learning_rate,
            L1_reg=0.0,
            L2_reg=0.0001,
            zero_last_layer_params=False
    ):
        self.input = T.tensor3('input', dtype=config.floatX)

        super(SparseBlockModel, self).__init__(
            data,
            layer_descriptions,
            batch_size,
            learning_rate,
            L1_reg,
            L2_reg,
            zero_last_layer_params
        )

    def build_layers(self):
        layers = []
        activation = self.input
        num_layers = len(self.layer_descriptions)
        for i, layer_desc in enumerate(self.layer_descriptions):
            constructor_params = {
                'x': activation,
                'n_in': (layer_desc['n_in'], layer_desc['n_units_per']),
                'n_out': (layer_desc['n_hids'], layer_desc['n_units_per']),
                'batch_size': self.batch_size,
                'k': layer_desc['k'],
                'activation': layer_desc['activation'],
                'name': 'b_layer_%d' % i
            }
            if i == 0:
                constructor_params['n_in'] = (1, layer_desc['n_in'])
            elif i == (num_layers - 1):
                constructor_params['n_out'] = (1, layer_desc['n_hids'])

            # The input indices should either be specified as in the first and
            # last layers or be the output indices of the previous layer.
            if 'in_idxs' in layer_desc.keys():
                constructor_params['in_idxs'] = layer_desc['in_idxs']
            else:
                constructor_params['in_idxs'] = layers[-1].out_idxs

            if 'out_idxs' in layer_desc.keys():
                constructor_params['out_idxs'] = layer_desc['out_idxs']

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
            layer_descriptions,
            batch_size,
            learning_rate,
            L1_reg=0.0,
            L2_reg=0.0001,
            zero_last_layer_params=False
    ):
        self.input = T.matrix('input', dtype=config.floatX)
        self.num_layers = len(layer_descriptions)

        super(MLPModel, self).__init__(
            data,
            layer_descriptions,
            batch_size,
            learning_rate,
            L1_reg,
            L2_reg,
            zero_last_layer_params
        )

    def _calc_num_units(self, layer_idx, params):
        raise NotImplementedError('_calc_num_units')

    def build_layers(self):
        layers = []
        for i, layer_desc in enumerate(self.layer_descriptions):
            units = self._calc_num_units(i)
            constructor_params = {
                'n_in': units[0],
                'n_out': units[1],
                'batch_size': self.batch_size,
                'k': layer_desc['k'],
                'activation': layer_desc['activation'],
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
    def _calc_num_units(self, layer_idx):
        params = self.layer_descriptions[layer_idx]
        units = [
            params['n_in']*params['n_units_per'],
            params['n_hids']*params['n_units_per']
        ]
        if layer_idx == 0:
            units[0] = params['n_in']
        elif layer_idx == (self.num_layers - 1):
            units[1] = params['n_hids']
        return tuple(units)


class EqualComputationsModel(MLPModel):
    def _calc_num_units(self, layer_idx):
        params = self.layer_descriptions[layer_idx]
        units = [
            params['n_in']*params['n_units_per'],
            params['n_hids']*params['n_units_per']
        ]
        if layer_idx == 0:
            units[0] = params['n_in']
        elif layer_idx == (self.num_layers - 1):
            units[1] = params['n_hids']

        units[0] *= self.layer_descriptions[layer_idx - 1]['k']
        units[1] *= params['k']

        return tuple(units)


def simple_train(
        model,
        train_model,
        test_model,
        validate_model,
        learning_rate,
        shared_learning_rate,
        n_epochs=1000
):
    ts = TS(['train', 'epoch'])
    epoch = 0
    minibatch_avg_cost_accum = 0
    while(epoch < n_epochs):
        ts.start('epoch')
        for minibatch_index in xrange(model.data.n_train_batches):
            if minibatch_index % 10 == 0:
                print '... minibatch_index: %d/%d\r' \
                    % (minibatch_index, model.data.n_train_batches),
                # Note the magic comma on the previous line prevents new lines
            ts.start('train')
            minibatch_avg_cost = train_model(minibatch_index)
            ts.end('train')

            minibatch_avg_cost_accum += minibatch_avg_cost[0]

        print '... minibatch_avg_cost_accum: %f' \
            % (minibatch_avg_cost_accum/float(model.data.n_train_batches))

        ts.end('epoch')
        epoch += 1

    return ts


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
                summary = ('validation error %f %% '
                           % (this_validation_loss * 100.))

                print ("%s" % (summary))

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
                    summary = 'test_score: %f' % (test_score * 100.)

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

    return ts


def run_experiments():
    model_class = SparseBlockModel
    #model_class = EqualParametersModel

    rng = np.random.RandomState()
    n_epochs = 1

    layer_descriptions = {
        0: {
            'n_hids': (25,),
            'n_units_per': 20,
            'k_pers': (1.,),
            'activations': (T.tanh, None),
        },
        1: {
            'n_hids': (25, 25),
            'n_units_per': 20,
            'k_pers': (1., 0.5),
            'activations': (T.tanh, T.tanh, None),
        },
        2: {
            'n_hids': (25, 100, 25),
            'n_units_per': 20,
            'k_pers': (1., 0.25, 1),
            'activations': (T.tanh, T.tanh, T.tanh, None),
        },
        3: {
            'n_hids': (50, 500, 10),
            'n_units_per': 20,
            'k_pers': (0.9, 0.05, 1),
            'activations': (T.tanh, T.tanh, T.tanh, None),
        },
        4: {
            'n_hids': (50, 500, 500, 10),
            'n_units_per': 20,
            'k_pers': (1, 0.05, 0.05, 1),
            'activations': (T.tanh, T.tanh, T.tanh, T.tanh, None),
        },
        5: {
            'n_hids': (50, 75, 100, 75, 50),
            'n_units_per': 32,
            'k_pers': (1., 0.1, 0.05, 0.1, 1),
            'activations': (T.tanh, T.tanh, T.tanh, T.tanh, T.tanh, None),
        },
        6: {
            'n_hids': (50, 500, 750, 750, 500, 10),
            'n_units_per': 32,
            'k_pers': (1, 0.1, 0.05, 0.05, 0.1, 1),
            'activations': (
                T.tanh, T.tanh, T.tanh, T.tanh, T.tanh, T.tanh, None
            ),
        },
    }

    parameter_configs = {
        0: {
            'batch_size': 32,
            'learning_rate': LinearChangeRate(
                0.21, -0.01, 0.2, 'learning_rate'
            ),
            'L1_reg': 0.0,
            'L2_reg': 0.0001
        },
        1: {
            'batch_size': 32,
            'learning_rate': LinearChangeRate(
                0.21, -0.01, 0.2, 'learning_rate'
            ),
            'L1_reg': 0.0,
            'L2_reg': 0.0001
        },
    }

    parameter_configs = {}
    for idx, batch_size in enumerate(range(0, 9)):
        parameter_configs[idx] = {
            'batch_size': 2**batch_size,
            'learning_rate': LinearChangeRate(
                0.21, -0.01, 0.2, 'learning_rate'
            ),
            'L1_reg': 0.0,
            'L2_reg': 0.0001
        }

    experiments = {}
    for i, (ld_idx, param_idx) in enumerate(product(
        range(len(layer_descriptions)),
        range(len(parameter_configs)),
    )):
        experiments[i] = {
            'layer_description': ld_idx,
            'parameters': param_idx,
        }

    exps = Experiments(
        input_dim=784,  # data.train_set_x.shape[-1].eval(),
        num_classes=10
    )

    exps_to_run = [0]
    exps_to_run = range(len(experiments))
    models = [EqualParametersModel, EqualComputationsModel, SparseBlockModel]

    data = None
    model = None
    timings = None
    for idx, model_class in product(exps_to_run, models):
        print "Experiment: %d, Model class: %s" % (idx, model_class)
        exp_config = experiments[idx]
        parameters = parameter_configs[exp_config['parameters']]
        exps.add(
            idx,
            layer_descriptions[exp_config['layer_description']],
            parameters
        )

        if (
                data is None
                or data.batch_size != parameters['batch_size']
                or data.reshape_data != model_class.reshape_data
        ):
            print "Loading Data"
            print "... MNIST"
            data = MNIST(parameters['batch_size'], model_class.reshape_data)
            gc.collect()

        try:
            shared_learning_rate = shared(
                np.array(
                    parameters['learning_rate'].rate,
                    dtype=config.floatX
                ),
                name='learning_rate'
            )

            print "Building model: %s" % str(model_class)
            model = model_class(
                data=data,
                layer_descriptions=exps.get_layers_definition(idx),
                batch_size=parameters['batch_size'],
                learning_rate=shared_learning_rate,
                L1_reg=parameters['L1_reg'],
                L2_reg=parameters['L2_reg'],
            )

            print "Training"
            timings = simple_train(
                model,
                learning_rate=parameters['learning_rate'],
                shared_learning_rate=shared_learning_rate,
                n_epochs=n_epochs,
                **model.build_functions()
            )

            model = None

        except MemoryError:
            epoch_time = -1

        if timings is not None:
            print "epoch_time: %s" % timings
            exps.save(idx, model_class.__name__, 'timings', timings)

        timings = None
        gc.collect()

        pkl.dump(exps, open('random_proj_experiments.pkl', 'wb'))


def plot_experiments():
    import matplotlib.pyplot as plt

    exps = pkl.load(open('random_proj_experiments.pkl', 'rb'))
    batch_sizes = [exp['parameters']['batch_size']
                   for exp_idx, exp in exps.experiments.iteritems()]
    timings = {model_name: np.zeros(len(batch_sizes))
               for model_name in exps.results[0].keys()}
    for exp_idx, results in exps.results.iteritems():
        for model_name, stats in results.iteritems():
            if model_name not in timings.keys():
                timings[model_name] = []
            timings[model_name][exp_idx] = stats['timings'].mean_difference('train')
    for model_name, timings in timings.iteritems():
        plt.plot(batch_sizes, timings, label=model_name)
    plt.legend()
    plt.show()


if __name__ == '__main__':
    run_experiments()

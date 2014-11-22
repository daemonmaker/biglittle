#! /usr/bin/env python

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

        self.layers_descriptions = {}
        self.parameters = {}
        self.experiments = {}
        self.results = {}

    def get_layers_definition(self, idx):
        layers_description = self.get_layers_description_by_exp_idx(idx)
        parameters = self.get_parameters_by_exp_idx(idx)

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
        index_selection_funcs = layers_description.get(
            'index_selection_funcs',
            (None,)*len(activations)
        )
        assert(len(activations) == len(index_selection_funcs))

        new_exp = []
        new_exp.append({
            'n_in': self.input_dim,
            'n_hids': n_hids[0],
            'n_units_per': n_units_per,
            'in_idxs': self.one_block_idxs,
            'k': k_pers[0],
            'activation': activations[0],
            'index_selection_func': index_selection_funcs[0]
        })
        for i in range(1, len(k_pers)):
            new_exp.append({
                'n_in': n_hids[i-1],
                'n_hids': n_hids[i],
                'n_units_per': n_units_per,
                'k': k_pers[i],
                'activation': activations[i],
                'index_selection_func': index_selection_funcs[i]
            })
        new_exp.append({
            'n_in': n_hids[-1],
            'n_hids': self.num_classes,
            'n_units_per': n_units_per,
            'out_idxs': self.one_block_idxs,
            'k': 1,
            'activation': activations[-1],
            'index_selection_func': index_selection_funcs[-1]
        })

        return new_exp

    def add_layers_description(self, idx, layers_description):
        self.layers_descriptions[idx] = layers_description

    def get_layers_description(self, idx):
        return self.layers_descriptions[idx]

    def get_layers_description_by_exp_idx(self, exp_idx):
        return self.get_layers_description(
            self.experiments[exp_idx]['layers_description_idx']
        )

    def add_parameters(self, idx, parameters):
        self.parameters[idx] = parameters

    def get_parameters(self, idx):
        return self.parameters[idx]

    def get_parameters_by_exp_idx(self, exp_idx):
        return self.get_parameters(self.experiments[exp_idx]['parameters_idx'])

    def get_table_idxs_by_exp_idxs(self, table, exp_idxs):
        result = set()
        for exp_idx in exp_idxs:
            result.add(self.experiments[exp_idx]['%s_idx' % table])
        return result

    def get_result_idxs_by_table_idx(self, table, idx):
        results = []
        for r_idx in self.results.keys():
            if self.experiments[r_idx]['%s_idx' % table] == idx:
                results.append(r_idx)
        return results

    def create_experiments(
        self, layers_descriptions_idxs=[], parameters_idxs=[]
    ):
        """
        Creates an experiment for each combination of layers and parameters
        """
        assert(type(layers_descriptions_idxs) == list)
        assert(type(parameters_idxs) == list)

        # Determine which layers and parameters to use
        if len(layers_descriptions_idxs) == 0:
            layers_descriptions_idxs = range(len(self.layers_descriptions))

        if len(parameters_idxs) == 0:
            parameters_idxs = range(len(self.parameters))

        # Create the experiments
        for idx, (ld_idx, p_idx) in enumerate(product(
            layers_descriptions_idxs,
            parameters_idxs
        )):
            self.experiments[idx] = {
                'layers_description_idx': ld_idx,
                'parameters_idx': p_idx
            }
            self.results[idx] = {}

    def get_idxs(self, table, filters=[], has_results=False):
        """
        Returns a list of idxs from the specified table matching the specified
        filters. Filters should be a dictionary where the key is the column
        name and the value is the required value.

        has_results determines whether there are results for a given
        experiment and is only relevant when search the experiments table.
        """
        assert(type(filters) == list)

        if table == 'experiments':
            source = self.experiments
        elif table == 'parameters':
            source = self.parameters
        elif table == 'layers_descriptions':
            source = self.layers_descriptions

        if len(filters) == 0:
            return source.keys()

        results = []
        for idx, values in source.iteritems():
            good = True

            # Determine whether the current record fits all the filters
            for k, v in filters:
                if values[k] != v:
                    good = False

            # Determine whether there are results for this experiment
            if has_results and idx not in self.results.keys():
                good = False

            if good:
                results.append(idx)
        return results

    def get_experiment_idxs(
            self, layers_description_idx=[], parameters_idx=[]
    ):
        assert(type(layers_description_idx) == list)
        assert(type(parameters_idx) == list)

        results = []
        for exp_idx, idxs in self.experiments.iteritems():
            good = True
            if (
                    len(layers_description_idx) > 0 and
                    idxs['layers_description_idx']
                    not in layers_description_idx
            ):
                good = False
            if (
                    len(parameters_idxs) > 0 and
                    idxs['parameters_idx'] not in parameters_idx
            ):
                good = False
            if good:
                results.append(exp_idx)
        return results

    def save(self, exp_id, model_name, k, v):
        if model_name not in self.results[exp_id].keys():
            self.results[exp_id][model_name] = {}
        self.results[exp_id][model_name][k] = v

    class ExperimentsIterator(object):
        def __init__(self, exps):
            self.exps = exps

            self.current_idx = 0
            self.stop_idx = len(self.exps.experiments) - 1

        def __iter__(self):
            return self

        def next(self):
            if self.current_idx > self.stop_idx:
                raise StopIteration

            self.current_idx += 1
            return self.current_idx - 1

    def __iter__(self):
        return self.ExperimentsIterator(self)


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


def all_same(idxs):
    return idxs[0, :].reshape((1, idxs.shape[1])).repeat(idxs.shape[0], axis=0)


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
            activation = layers[-1].output(
                activation,
                layer_desc.get('index_selection_func', None)
            )

        return layers

    def calculate_activation(self):
        print "... Calculating activation"
        top_active = []
        activation = self.input
        # TODO DWEBB change this to enumerate the layers instead of using an index
        for i in range(len(self.layers)):
            activation = self.layers[i].output(
                activation,
                self.layer_descriptions[i].get('index_selection_func', None)
            )
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
    timing_stats,
    n_epochs=1000
):
    timing_stats.add(['epoch', 'train'])
    epoch = 0
    minibatch_avg_cost_accum = 0
    while(epoch < n_epochs):
        timing_stats.start('epoch')
        for minibatch_index in xrange(model.data.n_train_batches):
            if minibatch_index % 10 == 0:
                print '... minibatch_index: %d/%d\r' \
                    % (minibatch_index, model.data.n_train_batches),
                # Note the magic comma on the previous line prevents new lines
            timing_stats.start('train')
            minibatch_avg_cost = train_model(minibatch_index)
            timing_stats.end('train')

            minibatch_avg_cost_accum += minibatch_avg_cost[0]

        print '... minibatch_avg_cost_accum: %f' \
            % (minibatch_avg_cost_accum/float(model.data.n_train_batches))

        timing_stats.end('epoch')
        epoch += 1


def train(
    model,
    train_model,
    test_model,
    validate_model,
    learning_rate,
    shared_learning_rate,
    timing_stats,
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

    timing_stats.add(['train', 'epoch', 'valid'])

    summarize_rates()

    timing_stats.start()
    while (epoch < n_epochs) and (not done_looping):
        epoch = epoch + 1
        timing_stats.start('epoch')
        for minibatch_index in xrange(data.n_train_batches):
            timing_stats.start('train')
            minibatch_avg_cost = train_model(minibatch_index)
            timing_stats.end('train')
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
                timing_stats.end('epoch')
                timing_stats.reset('epoch')

                timing_stats.reset('train')
                accum = accum / validation_frequency
                summary = ("minibatch_avg_cost: %f, time: %f"
                           % (accum, timing_stats.accumed['train'][-1][1]))
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

    timing_stats.end()
    print('Optimization complete. Best validation score of %f %% '
          'obtained at iteration %i, with test performance %f %%' %
          (best_validation_loss * 100., best_iter + 1, test_score * 100.))
    print >> sys.stderr, ('The code for file ' +
                          os.path.split(__file__)[1] +
                          ' ran for %s' % timing_stats)


def run_experiments(exps, models, rng=None):
    if rng is None:
        rng = np.random.RandomState()

    data = None
    model = None
    timings = None
    for idx, model_class in product(exps, models):
        print 'Experiment: %d, Model class: %s' % (idx, model_class)

        parameters = exps.get_parameters_by_exp_idx(idx)

        print 'Batch size: %d' % parameters['batch_size']

        if (
                data is None
                or data.batch_size != parameters['batch_size']
                or data.reshape_data != model_class.reshape_data
        ):
            print 'Loading Data'
            print '... MNIST'
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

            timings = TS(['build_model', 'build_functions', 'full_train'])

            print 'Building model: %s' % str(model_class)
            timings.start('build_model')
            layer_definitions = exps.get_layers_definition(idx)
            model = model_class(
                data=data,
                layer_descriptions=layer_definitions,
                batch_size=parameters['batch_size'],
                learning_rate=shared_learning_rate,
                L1_reg=parameters['L1_reg'],
                L2_reg=parameters['L2_reg'],
            )
            print '... time: %f' % timings.end('build_model')

            print 'Building functions'
            timings.start('build_functions')
            functions = model.build_functions()
            print '... time: %f' % timings.end('build_functions')

            print 'Training'
            timings.start('full_train')
            simple_train(
                model,
                learning_rate=parameters['learning_rate'],
                shared_learning_rate=shared_learning_rate,
                n_epochs=parameters['n_epochs'],
                timing_stats=timings,
                **functions
            )
            print 'Training time: %d' % timings.end('full_train')

            model = None

        except MemoryError:
            epoch_time = -1

        if timings is not None:
            print 'Timings: %s' % timings
            exps.save(idx, model_class.__name__, 'timings', timings)

        timings = None
        gc.collect()

        pkl.dump(exps, open('random_proj_experiments.pkl', 'wb'))


def plot_times_by_batch(database):
    import matplotlib.pyplot as plt

    # Load the database
    exps = pkl.load(open(database, 'rb'))

    # Find experiments that have results
    exp_idxs = exps.get_idxs('experiments', has_results=True)

    # Plot results for each experiment grouped by the layers_description
    layers_description_idxs = exps.get_table_idxs_by_exp_idxs(
        'layers_description',
        exp_idxs
    )

    for layers_description_idx in layers_description_idxs:
        result_idxs = exps.get_result_idxs_by_table_idx(
            'layers_description',
            layers_description_idx
        )
        batch_sizes = [exps.get_parameters_by_exp_idx(idx)['batch_size']
                       for idx in result_idxs]
        timings = {model_name: np.zeros(len(batch_sizes))
                   for model_name in exps.results[result_idxs[0]].keys()}
        for i, idx in enumerate(result_idxs):
            for model_name, stats in exps.results[idx].iteritems():
                timings[model_name][i] = stats[
                    'timings'
                ].mean_difference('train')/batch_sizes[i]

        for model_name, timings in timings.iteritems():
            plt.plot(batch_sizes, timings, marker='o', label=model_name,)
        plt.title('Train time per sample')
        layers_description = exps.get_layers_description(
            layers_description_idx
        )
        plt.suptitle('layers_description_idx: %d, n_units: %s,'
                     ' n_hids: %s, k_pers: %s' % (
            layers_description_idx,
            layers_description['n_hids'],
            layers_description['n_units_per'],
            layers_description['k_pers']
        ))
        plt.xlabel('Batch Size')
        plt.ylabel('Time (s)')
        plt.legend()
        plt.xticks(batch_sizes)
        plt.show()


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(
        description='Run random_proj experiments and plot results'
    )
    parser.add_argument(
        '-m', '--use_models',
        type=int, default=[], nargs='+',
        help='Identifier for which models to use in the experiments.'
    )
    parser.add_argument(
        '-b', '--batch_sizes',
        type=int, default=[1, 9], nargs='+',
        help='Range of batch sizes to test.'
    )
    parser.add_argument(
        '-n', '--number_of_epochs',
        type=int, default=1,
        help='Number of epochs to execute for each experiment.'
    )
    parser.add_argument(
        '-d', '--database',
        default='random_proj_experiments.pkl',
        help='Which database to use.'
    )
    parser.add_argument(
        '-l', '--load_database',
        default=False, action='store_true',
        help='Whether to load an existing database.'
    )
    parser.add_argument(
        '-p', '--plot',
        default=False,
        action='store_true',
        help='Plot results instaed of execute experiments.'
    )
    args = parser.parse_args()

    if args.plot:
        plot_times_by_batch(args.database)
    else:
        if args.load_database:
            exps = pkl.load(open(args.database))
        else:
            ## Create experiments
            exps = Experiments(
                input_dim=784,  # data.train_set_x.shape[-1].eval(),
                num_classes=10
            )

            # Add descriptions of models
            exps.add_layers_description(
                0,
                {
                    'n_hids': (25,),
                    'n_units_per': 32,
                    'k_pers': (1.,),
                    'activations': (T.tanh, None),
                }
            )
            exps.add_layers_description(
                1,
                {
                    'n_hids': (25, 25),
                    'n_units_per': 32,
                    'k_pers': (1., 0.5),
                    'activations': (T.tanh, T.tanh, None),
                }
            )
            exps.add_layers_description(
                2,
                {
                    'n_hids': (25, 100, 25),
                    'n_units_per': 32,
                    'k_pers': (1., 0.25, 1),
                    'activations': (T.tanh, T.tanh, T.tanh, None),
                }
            )
            exps.add_layers_description(
                3,
                {
                    'n_hids': (25, 100, 25),
                    'n_units_per': 32,
                    'k_pers': (1., 0.25, 1),
                    'activations': (T.tanh, T.tanh, T.tanh, None),
                    'index_selection_funcs': (
                        all_same, all_same, all_same, None
                    )
                }
            )
            exps.add_layers_description(
                4,
                {
                    'n_hids': (50, 100, 10),
                    'n_units_per': 32,
                    'k_pers': (1, 0.05, 1),
                    'activations': (T.tanh, T.tanh, T.tanh, None),
                },
            )
            exps.add_layers_description(
                5,
                {
                    'n_hids': (50, 100, 10),
                    'n_units_per': 32,
                    'k_pers': (1, 0.05, 1),
                    'activations': (T.tanh, T.tanh, T.tanh, None),
                    'index_selection_funcs': (
                        all_same, all_same, all_same, None
                    )
                },
            )
            exps.add_layers_description(
                6,
                {
                    'n_hids': (50, 100, 100, 10),
                    'n_units_per': 32,
                    'k_pers': (1, 0.05, 0.05, 1),
                    'activations': (T.tanh, T.tanh, T.tanh, T.tanh, None),
                }
            )
            exps.add_layers_description(
                7,
                {
                    'n_hids': (50, 100, 100, 10),
                    'n_units_per': 32,
                    'k_pers': (1, 0.05, 0.05, 1),
                    'activations': (T.tanh, T.tanh, T.tanh, T.tanh, None),
                    'index_selection_funcs': (
                        all_same, all_same, all_same, all_same, None
                    )
                }
            )
            exps.add_layers_description(
                8,
                {
                    'n_hids': (50, 75, 100, 75, 50),
                    'n_units_per': 32,
                    'k_pers': (1., 0.1, 0.05, 0.1, 1),
                    'activations': (
                        T.tanh, T.tanh, T.tanh, T.tanh, T.tanh, None
                    ),
                }
            )
            exps.add_layers_description(
                9,
                {
                    'n_hids': (50, 75, 100, 75, 50),
                    'n_units_per': 32,
                    'k_pers': (1., 0.1, 0.05, 0.1, 1),
                    'activations': (
                        T.tanh, T.tanh, T.tanh, T.tanh, T.tanh, None
                    ),
                    'index_selection_funcs': (
                        all_same, all_same, all_same, all_same,
                        all_same, None
                    )
                }
            )
            exps.add_layers_description(
                10,
                {
                    'n_hids': (50, 500, 500, 500, 500, 10),
                    'n_units_per': 32,
                    'k_pers': (1, 0.1, 0.05, 0.05, 0.1, 1),
                    'activations': (
                        T.tanh, T.tanh, T.tanh, T.tanh, T.tanh, T.tanh, None
                    ),
                },
            )
            exps.add_layers_description(
                11,
                {
                    'n_hids': (50, 500, 500, 500, 500, 10),
                    'n_units_per': 32,
                    'k_pers': (1, 0.1, 0.05, 0.05, 0.1, 1),
                    'activations': (
                        T.tanh, T.tanh, T.tanh, T.tanh, T.tanh, T.tanh, None
                    ),
                    'index_selection_funcs': (
                        all_same, all_same, all_same, all_same,
                        all_same, all_same, None
                    )
                }
            )

            # Add parameter combinations
            for idx, batch_size in enumerate(args.batch_sizes):
                exps.add_parameters(
                    idx,
                    {
                        'n_epochs': args.number_of_epochs,
                        'batch_size': batch_size,
                        'learning_rate': LinearChangeRate(
                            0.21, -0.01, 0.2, 'learning_rate'
                        ),
                        'L1_reg': 0.0,
                        'L2_reg': 0.0001
                    }
                )

            if len(args.use_models) > 0:
                print 'Executing experiments %s' % args.use_models
                exps.create_experiments(args.use_models)
            else:
                exps.create_experiments()

        run_experiments(
            exps,
            models=[
                EqualParametersModel,
                EqualComputationsModel,
                SparseBlockModel
            ]
        )

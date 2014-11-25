#! /usr/bin/env python

import time
from datetime import datetime
import numpy as np
import sys
import os
import os.path as op
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
from experiments import Experiments
from layer import HiddenLayer, HiddenBlockLayer, HiddenRandomBlockLayer
from timing_stats import TimingStats as TS
from models import (
    EqualParametersModel,
    EqualComputationsModel,
    SparseBlockModel,
    all_same
)


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
        print "Epoch %d" % epoch
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
        plt.title('Train time per sample', fontsize=12)
        layers_description = exps.get_layers_description(
            layers_description_idx
        )
        plt.suptitle(
            'layers_description_idx: %d, n_units: %s, n_hids: %s,\n'
            'k_pers: %s, all same: %r' % (
                layers_description_idx,
                layers_description['n_hids'],
                layers_description['n_units_per'],
                layers_description['k_pers'],
                'index_selection_funcs' in layers_description.keys()
            ),
            y=0.99,
            fontsize=10
        )
        plt.xlabel('Batch Size')
        plt.ylabel('Time (s)')
        plt.legend()
        plt.xticks(batch_sizes)
        figs_dir = 'figs'
        if not op.exists(figs_dir):
            os.mkdir(figs_dir)
        plt.savefig(
            op.join(
                figs_dir,
                'layers_description_%d.png' % layers_description_idx
            ),
            format='png'
        )
        plt.show()


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(
        description='Run random_proj experiments and plot results'
    )
    parser.add_argument(
        '-m', '--use_layers',
        type=int, default=[], nargs='+',
        help='Identifier for which models to use in the experiments.'
    )
    parser.add_argument(
        '-c', '--layer_class',
        default='HiddenRandomBlockLayer',
        help='The type of layer to use in the block sparse model.'
    )
    parser.add_argument(
        '-b', '--batch_sizes',
        type=int, default=[32], nargs='+',
        help='Range of batch sizes to test.'
    )
    parser.add_argument(
        '-n', '--number_of_epochs',
        type=int, default=1,
        help='Number of epochs to execute for each experiment.'
    )
    parser.add_argument(
        '-u', '--units_per_block',
        type=int, default=32,
        help='Number of units per block in the sparse block models.'
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
            ## Determine the type of sparsity layer to use
            if args.layer_class == 'HiddenRandomBlockLayer':
                layer_class = HiddenRandomBlockLayer
            else:
                layer_class = HiddenBlockLayer

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
                    'n_units_per': args.units_per_block,
                    'k_pers': (1, 1),
                    'activations': (T.tanh, None),
                    'layer_classes': [
                        HiddenBlockLayer,
                        HiddenBlockLayer,
                    ],
                }
            )
            exps.add_layers_description(
                1,
                {
                    'n_hids': (25, 25),
                    'n_units_per': args.units_per_block,
                    'k_pers': (1, 0.5, 1),
                    'activations': (T.tanh, T.tanh, None),
                    'layer_classes': [
                        HiddenBlockLayer,
                        layer_class,
                        HiddenBlockLayer,
                    ],
                }
            )
            exps.add_layers_description(
                2,
                {
                    'n_hids': (25, 100, 25),
                    'n_units_per': args.units_per_block,
                    'k_pers': (1, 0.25, 0.25, 1),
                    'activations': (T.tanh, T.tanh, T.tanh, None),
                    'layer_classes': [
                        HiddenBlockLayer,
                        layer_class,
                        layer_class,
                        HiddenBlockLayer,
                    ],
                }
            )
            exps.add_layers_description(
                3,
                {
                    'n_hids': (25, 100, 25),
                    'n_units_per': args.units_per_block,
                    'k_pers': (1., 0.25, 0.25, 1),
                    'activations': (T.tanh, T.tanh, T.tanh, None),
                    'layer_classes': [
                        HiddenBlockLayer,
                        layer_class,
                        layer_class,
                        HiddenBlockLayer,
                    ],
                    'index_selection_funcs': (
                        None, all_same, all_same, None
                    )
                }
            )
            exps.add_layers_description(
                4,
                {
                    'n_hids': (50, 100, 20),
                    'n_units_per': args.units_per_block,
                    'k_pers': (1, 0.05, 0.2, 1),
                    'activations': (T.tanh, T.tanh, T.tanh, None),
                    'layer_classes': [
                        HiddenBlockLayer,
                        layer_class,
                        layer_class,
                        HiddenBlockLayer,
                    ],
                },
            )
            exps.add_layers_description(
                5,
                {
                    'n_hids': (50, 100, 20),
                    'n_units_per': args.units_per_block,
                    'k_pers': (1, 0.05, 0.05, 1),
                    'activations': (T.tanh, T.tanh, T.tanh, None),
                    'layer_classes': [
                        HiddenBlockLayer,
                        layer_class,
                        layer_class,
                        HiddenBlockLayer,
                    ],
                    'index_selection_funcs': (
                        None, all_same, all_same, None
                    )
                },
            )
            exps.add_layers_description(
                6,
                {
                    'n_hids': (25, 100, 100, 25),
                    'n_units_per': args.units_per_block,
                    'k_pers': (1, 0.05, 0.05, 1, 1),
                    'activations': (T.tanh, T.tanh, T.tanh, T.tanh, None),
                    'layer_classes': [
                        HiddenBlockLayer,
                        layer_class,
                        layer_class,
                        layer_class,
                        HiddenBlockLayer,
                    ],
                }
            )
            exps.add_layers_description(
                7,
                {
                    'n_hids': (25, 100, 100, 25),
                    'n_units_per': args.units_per_block,
                    'k_pers': (1, 0.05, 0.05, 1, 1),
                    'activations': (T.tanh, T.tanh, T.tanh, T.tanh, None),
                    'layer_classes': [
                        HiddenBlockLayer,
                        layer_class,
                        layer_class,
                        layer_class,
                        HiddenBlockLayer,
                    ],
                    'index_selection_funcs': (
                        None, all_same, all_same, all_same, None
                    )
                }
            )
            exps.add_layers_description(
                8,
                {
                    'n_hids': (50, 200, 500, 200, 50),
                    'n_units_per': args.units_per_block,
                    'k_pers': (1., 0.1, 0.02, 0.02, 0.1, 1),
                    'activations': (
                        None, T.tanh, T.tanh, T.tanh, T.tanh, None
                    ),
                    'layer_classes': [
                        HiddenBlockLayer,
                        layer_class,
                        layer_class,
                        layer_class,
                        layer_class,
                        HiddenBlockLayer,
                    ],
                }
            )
            exps.add_layers_description(
                9,
                {
                    'n_hids': (50, 75, 200, 75, 50),
                    'n_units_per': args.units_per_block,
                    'k_pers': (1., 0.1, 0.05, 0.05, 0.1, 1),
                    'activations': (
                        T.tanh, T.tanh, T.tanh, T.tanh, T.tanh, None
                    ),
                    'layer_classes': [
                        HiddenBlockLayer,
                        layer_class,
                        layer_class,
                        layer_class,
                        layer_class,
                        HiddenBlockLayer,
                    ],
                    'index_selection_funcs': (
                        None, all_same, all_same, all_same,
                        all_same, None
                    )
                }
            )
            exps.add_layers_description(
                10,
                {
                    'n_hids': (50, 500, 500, 500, 500, 20),
                    'n_units_per': args.units_per_block,
                    'k_pers': (1, 0.07, 0.03, 0.02, 0.01, 0.15, 1),
                    'activations': (
                        T.tanh, T.tanh, T.tanh, T.tanh, T.tanh, T.tanh, None
                    ),
                    'layer_classes': [
                        HiddenBlockLayer,
                        layer_class,
                        layer_class,
                        layer_class,
                        layer_class,
                        layer_class,
                        HiddenBlockLayer,
                    ],
                },
            )
            exps.add_layers_description(
                11,
                {
                    'n_hids': (50, 500, 500, 500, 500, 20),
                    'n_units_per': args.units_per_block,
                    'k_pers': (1, 0.07, 0.03, 0.02, 0.01, 0.15, 1),
                    'activations': (
                        T.tanh, T.tanh, T.tanh, T.tanh, T.tanh, T.tanh, None
                    ),
                    'layer_classes': [
                        HiddenBlockLayer,
                        layer_class,
                        layer_class,
                        layer_class,
                        layer_class,
                        layer_class,
                        HiddenBlockLayer,
                    ],
                    'index_selection_funcs': (
                        None, all_same, all_same, all_same,
                        all_same, all_same, None
                    )
                }
            )
            exps.add_layers_description(
                12,
                {
                    'n_hids': (50, 100, 500, 500, 500, 500, 500, 100, 20),
                    'n_units_per': args.units_per_block,
                    'k_pers': (1, 0.1, 0.05, 0.01, 0.01, 0.01, 0.01, 0.05, 0.1, 1),
                    'activations': (
                        None, T.tanh, T.tanh, T.tanh, T.tanh,
                        T.tanh, T.tanh, T.tanh, T.tanh, None
                    ),
                    'layer_classes': [
                        HiddenBlockLayer,
                        layer_class,
                        layer_class,
                        layer_class,
                        layer_class,
                        layer_class,
                        layer_class,
                        layer_class,
                        layer_class,
                        HiddenBlockLayer,
                    ],
                },
            )
            exps.add_layers_description(
                13,
                {
                    'n_hids': (50, 100, 500, 500, 500, 500, 500, 100, 20),
                    'n_units_per': args.units_per_block,
                    'k_pers': (1, 0.1, 0.05, 0.01, 0.01, 0.01, 0.1, 0.5, 0.1, 1),
                    'activations': (
                        None, T.tanh, T.tanh, T.tanh, T.tanh,
                        T.tanh, T.tanh, T.tanh, T.tanh, None
                    ),
                    'layer_classes': [
                        HiddenBlockLayer,
                        layer_class,
                        layer_class,
                        layer_class,
                        layer_class,
                        layer_class,
                        layer_class,
                        layer_class,
                        layer_class,
                        HiddenBlockLayer,
                    ],
                    'index_selection_funcs': (
                        None, all_same, all_same, all_same,
                        all_same, all_same, None
                    )
                }
            )
            exps.add_layers_description(
                14,
                {
                    'n_hids': (50, 100, 20),
                    'n_units_per': args.units_per_block,
                    'k_pers': (1, 0.05, 0.05, 1),
                    'activations': (T.tanh, T.tanh, T.tanh, None),
                    'layer_classes': [
                        layer_class,
                        layer_class,
                        layer_class,
                        layer_class,
                    ],
                },
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

            if len(args.use_layers) > 0:
                print 'Executing experiments for layers %s' % args.use_layers
                exps.create_experiments(args.use_layers)
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

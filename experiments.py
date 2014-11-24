import numpy as np
from itertools import product

from theano import shared


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
        layer_classes = layers_description['layer_classes']
        assert(len(activations) == len(index_selection_funcs))

        new_exp = []
        new_exp.append({
            'n_in': self.input_dim,
            'n_hids': n_hids[0],
            'n_units_per': n_units_per,
            'in_idxs': self.one_block_idxs,
            'k': k_pers[0],
            'activation': activations[0],
            'index_selection_func': index_selection_funcs[0],
            'layer_class': layer_classes[0]
        })
        for i in range(1, len(k_pers) - 1):
            new_exp.append({
                'n_in': n_hids[i-1],
                'n_hids': n_hids[i],
                'n_units_per': n_units_per,
                'k': k_pers[i],
                'activation': activations[i],
                'index_selection_func': index_selection_funcs[i],
                'layer_class': layer_classes[i]
            })
        new_exp.append({
            'n_in': n_hids[-1],
            'n_hids': self.num_classes,
            'n_units_per': n_units_per,
            'out_idxs': self.one_block_idxs,
            'k': k_pers[-1],
            'activation': activations[-1],
            'index_selection_func': index_selection_funcs[-1],
            'layer_class': layer_classes[-1]
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



from theano import function
from theano import tensor as T
from theano import config

from utils import *
from layer import (
    HiddenLayer,
    HiddenBlockLayer,
    HiddenRandomBlockLayer
)


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
    first_row = idxs[0, :].reshape((1, idxs.shape[1]))
    first_row = T.argsort(first_row)
    return first_row.repeat(idxs.shape[0], axis=0)


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

            #layers.append(HiddenRandomBlockLayer(**constructor_params))
            layers.append(layer_desc['layer_class'](**constructor_params))
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

import numpy as np

import theano
from theano import function
from theano import tensor as T
from theano import config
from theano import shared

from theano.sandbox.cuda.blocksparse import sparse_block_dot_SS

if theano.config.mode == 'FAST_COMPILE':
    mode_with_gpu = theano.compile.mode.get_mode('FAST_RUN').including('gpu')
else:
    mode_with_gpu = theano.compile.mode.get_default_mode().including('gpu')


class HiddenLayer(object):
    def __init__(
            self,
            n_in,
            n_out,
            batch_size,
            k=0.05,
            activation=T.tanh,
            name='HiddenLayer',
            rng=None
    ):
        assert(
            n_in > 0
            and n_out > 0
        )

        self.n_in = n_in
        self.n_out = n_out

        assert(batch_size > 0)
        self.batch_size = batch_size

        self.activation = activation
        self.name = name

        if rng is None:
            self.rng = np.random.RandomState()
            self.rng.seed(0)
        else:
            self.rng = rng

        # Setup parameters
        bound = np.sqrt(6. / (n_in + n_out))
        W_val = np.asarray(self.rng.uniform(
            low=-bound,
            high=bound,
            size=(
                self.n_in,
                self.n_out
            )
        ), dtype=config.floatX)
        #W_val = np.ones((self.n_in, self.n_out), dtype=config.floatX)

        b_val = np.zeros((self.n_out,)).astype(config.floatX)

        self._setup_parameters(W_val, b_val)

        assert(k >= 0. and k <= 1.)
        self.k = int(k*n_out)
        if self.k > 0.:
            name = 'top_active'
            if name is not None:
                name = self.name + '_' + name

            self.top_active = shared(
                np.repeat(np.arange(self.k).reshape(1, self.k), self.batch_size, axis=0),
                name=name,
            )

    def _setup_parameters(self, W_val, b_val):
        self.W = shared(
            W_val,
            name='W_'+str(self.name)
        )

        self.b = shared(
            b_val,
            name='b_'+str(self.name)
        )

        self.params = [self.W, self.b]

    def set_parameters(self, W, b):
        self.W.set_value(W)
        self.b.set_value(b)

    def most_active(self, x):
        return function([], updates=(x, T.argsort(x)))

    def output(self, x):
        lin = T.dot(x, self.W) + self.b
        return (lin if self.activation is None
                else self.activation(lin))

    def prediction(self, x):
        return T.argmax(x, axis=1)

    def cost(self, x, y):
        return -T.mean(T.log(x)[T.arange(y.shape[0]), y])

    def error(self, x, y):
        return T.mean(T.neq(self.prediction(x), y))


class HiddenBlockLayer(HiddenLayer):
    def __init__(
            self,
            n_in,
            n_out,
            in_idxs,
            out_idxs,
            batch_size,
            activation=T.tanh,
            name='HiddenBlockLayer',
            rng=None
    ):
        assert(
            type(n_in) == tuple
            and type(n_out) == tuple
        )

        super(
            HiddenBlockLayer,
            self
        ).__init__(
            n_in[0],
            n_out[0],
            batch_size,
            k=0,
            activation=activation,
            name=name,
            rng=rng
        )

        assert(
            n_in[1] > 0
            and n_out[1] > 0
        )

        self.n_units_per_in = n_in[1]
        self.n_units_per_out = n_out[1]

        # Setup parameters
        self.in_idxs = in_idxs
        self.out_idxs = out_idxs

        inputSize = self.n_in*self.n_units_per_in
        outputSize = self.n_out*self.n_units_per_out

        bound = np.sqrt(6. / (inputSize + outputSize))
        W_val = np.asarray(self.rng.uniform(
            low=-bound,
            high=bound,
            size=(
                self.n_in,
                self.n_out,
                self.n_units_per_in,
                self.n_units_per_out
            )
        ), dtype=config.floatX)
        #W_val = np.ones((self.n_in, self.n_out, self.n_units_per_in, self.n_units_per_out), dtype=config.floatX)

        b_val = np.zeros(
            outputSize
        ).reshape(
            self.n_out,
            self.n_units_per_out
        ).astype(config.floatX)

        self._setup_parameters(W_val, b_val)

    def output(self, x):
        sparse = sparse_block_dot_SS(
            self.W,
            x,
            self.in_idxs,
            self.b,
            self.out_idxs
        )
        return (sparse if self.activation is None
                else self.activation(sparse))

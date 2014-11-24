import os
import gzip
import cPickle
import numpy

import theano
import theano.tensor as T


def relu(x):
    return T.maximum(0, x)


def maxout(x):
    return T.max(x, axis=1)


def relu_softmax(x):
    return T.nnet.softmax(relu(x))


def restore_parameters(param_filename, layers):
    parameters = cnx.read_parameters(param_filename)
    for params, layer in zip(parameters, layers):
        layer.set_parameters(params[0], params[1])


def add_regularization(layers, cost, L1_reg, L2_reg):
    if L1_reg == 0 and L2_reg == 0:
        return cost

    for layer in layers:
        cost = cost + L1_reg*abs(layer.W).sum() + L2_reg*(layer.W**2).sum()

    return cost


class LearningRate(object):
    def __init__(self, init_rate, name):
        assert((init_rate >= 0) and (init_rate <= 1))
        #self.rate = shared(init_rate, name=name)
        self.rate = init_rate

    def update(self, name):
        #return []
        pass


class LinearChangeRate(LearningRate):
    def __init__(self, init_rate, change, final_rate, name):
        super(LinearChangeRate, self).__init__(init_rate, name)

        assert((final_rate >= 0) and (final_rate <= 1))
        self.final_rate = final_rate

        if init_rate < final_rate:
            assert(change > 0)
            self.increasing = True
        else:
            assert(change <= 0)
            self.increasing = False

        #self.change = shared(change, name + '_change')
        self.change = change

    def update(self):
        #update = self.rate + self.change
        #T.switch(
        #    self.increasing,
        #    
        #)
        #return [update]
        self.rate = self.rate + self.change

        if (self.change and
            (self.increasing and (self.rate > self.final_rate) or
             ((not self.increasing) and (self.rate < self.final_rate)))):
            self.rate = self.final_rate
            self.change = 0


def verify_cost(
        b_cost,
        b_layers,
        b_x,
        y,
        batch_size,
        train_set_x_b,
        train_set_y
):
    def f(W_0, W_1):
        index = 0
        d = {
            b_layers[0].W: T.patternbroadcast(
                W_0,
                (False, False, False, False)
            ),
            b_layers[1].W: T.patternbroadcast(
                W_1,
                (False, False, False, False)
            ),
            b_x: train_set_x_b[index*batch_size:(index+1)*batch_size],
            y: train_set_y[index*batch_size:(index+1)*batch_size]
        }
        return theano.clone(b_cost, d)
    return f


def verify_layer(expr, W):
    def f(W_real):
        index = 0
        d = {
            W: T.patternbroadcast(
                W_real,
                (False, False, False, False)
            )
        }
        return theano.clone(expr, d)
    return f


def verify_layers(
        batch_size,
        layers,
        train_set_x,
        train_set_y
):
    index = 0
    range_start = index*batch_size
    range_end = (index+1)*batch_size

    sample = train_set_x[range_start:range_end]
    layer_0_activation = layers[0].output(sample).eval()
    layer_1_activation = layers[1].output(layer_0_activation)

    layer_1_cost = layers[1].cost(
        T.nnet.softmax(T.mean(
            layer_1_activation,
            axis=2
        )),
        train_set_y[range_start:range_end]
    )

    layer_0_cost = layers[1].cost(
        T.nnet.softmax(T.mean(
            layers[1].output(layers[0].output(sample)),
            axis=2
        )),
        train_set_y[range_start:range_end]
    )

    temp = verify_layer(layer_1_cost, layers[1].W)
    T.verify_grad(
        temp,
        [layers[1].W.get_value()],
        rng=np.random.RandomState()
    )

    temp = verify_layer(layer_0_cost, layers[0].W)
    T.verify_grad(
        temp,
        [layers[0].W.get_value()],
        rng=np.RandomState()
    )


def load_data(dataset, reshape=False):
    ''' Loads the dataset

    :type dataset: string
    :param dataset: the path to the dataset (here MNIST)
    '''

    #############
    # LOAD DATA #
    #############

    # Download the MNIST dataset if it is not present
    data_dir, data_file = os.path.split(dataset)
    if data_dir == "" and not os.path.isfile(dataset):
        # Check if dataset is in the data directory.
        new_path = os.path.join(os.path.split(__file__)[0], "..", "data", dataset)
        if os.path.isfile(new_path) or data_file == 'mnist.pkl.gz':
            dataset = new_path

    if (not os.path.isfile(dataset)) and data_file == 'mnist.pkl.gz':
        import urllib
        origin = 'http://www.iro.umontreal.ca/~lisa/deep/data/mnist/mnist.pkl.gz'
        print 'Downloading data from %s' % origin
        urllib.urlretrieve(origin, dataset)

    print '... loading data'

    # Load the dataset
    f = gzip.open(dataset, 'rb')
    train_set, valid_set, test_set = cPickle.load(f)
    f.close()
    #train_set, valid_set, test_set format: tuple(input, target)
    #input is an numpy.ndarray of 2 dimensions (a matrix)
    #witch row's correspond to an example. target is a
    #numpy.ndarray of 1 dimensions (vector)) that have the same length as
    #the number of rows in the input. It should give the target
    #target to the example with the same index in the input.

    def shared_dataset(data_xy, borrow=True):
        """ Function that loads the dataset into shared variables

        The reason we store our dataset in shared variables is to allow
        Theano to copy it into the GPU memory (when code is run on GPU).
        Since copying data into the GPU is slow, copying a minibatch everytime
        is needed (the default behaviour if the data is not in a shared
        variable) would lead to a large decrease in performance.
        """
        data_x, data_y = data_xy
        if reshape:
            data_x = data_x.reshape(data_x.shape[0], 1, data_x.shape[1])
        shared_x = theano.shared(numpy.asarray(data_x,
                                               dtype=theano.config.floatX),
                                 borrow=borrow)
        shared_y = theano.shared(numpy.asarray(data_y,
                                               dtype=theano.config.floatX),
                                 borrow=borrow)
        # When storing data on the GPU it has to be stored as floats
        # therefore we will store the labels as ``floatX`` as well
        # (``shared_y`` does exactly that). But during our computations
        # we need them as ints (we use labels as index, and if they are
        # floats it doesn't make sense) therefore instead of returning
        # ``shared_y`` we will have to cast it to int. This little hack
        # lets ous get around this issue
        return shared_x, T.cast(shared_y, 'int32')

    test_set_x, test_set_y = shared_dataset(test_set)
    valid_set_x, valid_set_y = shared_dataset(valid_set)
    train_set_x, train_set_y = shared_dataset(train_set)

    rval = [(train_set_x, train_set_y), (valid_set_x, valid_set_y),
            (test_set_x, test_set_y)]
    return rval


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

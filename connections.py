import ipdb
from optparse import OptionParser as op
import cPickle as pkl
import os.path as path
import numpy as np

#from theano import config
#from theano import function
#import theano.tensor as T


def repeat(idxs, reps):
    return np.repeat(
        idxs.reshape(1, idxs.shape[0]),
        reps,
        axis=0
    )


def arange(k):
    return np.arange(k, dtype='int64')


def all_idxs(k, reps):
    return repeat(arange(k), reps)


class Extract(object):
    def __init__(self, k_percent):
        assert(0 < k_percent and 1 >= k_percent)
        self.k_percent = k_percent

    def extract(self, W, out_idxs):
        if out_idxs is not None:
            W = W[out_idxs]

        return self.f(W, out_idxs)

    def k_top(self, total_size):
        return int(self.k_percent*total_size)

    def outs_to_ins(self, W, out_idxs):
        return out_idxs if out_idxs is not None else np.arange(W.shape[0])


class ExtractTopWeightsFromOuts(Extract):
    """
    Extracts the top weights for a given layer with respect to the active
    output blocks from the previous layer.
    """
    def f(self, W, out_idxs):
        # Find kth max
        W_abs = np.abs(W)
        total_size = W.shape[0]*W.shape[1]
        k_top = self.k_top(total_size)
        w = np.sort(W_abs.reshape((total_size,)))
        max_val = w[-k_top]

        # Indices of all elements in matrix that are bigger
        idxs = np.where(W_abs > max_val)
        in_idxs = np.unique(idxs[0])
        out_idxs = np.unique(idxs[1])

        return in_idxs, out_idxs


class ExtractTopNeurons(Extract):
    def f(self, W, out_idxs):
        W_totals = np.abs(W).sum(axis=0)
        greatest_weight = np.argsort(W_totals)
        in_idxs = self.outs_to_ins(W, out_idxs)
        out_idxs = np.sort(greatest_weight[-self.k_top(W.shape[1]):])
        return in_idxs, out_idxs


class ExtractAssociations(Extract):
    def f(self, W, out_idxs):
        in_idxs = self.outs_to_ins(W, out_idxs)
        out_idxs = np.sort(
            np.argsort(W, axis=1)[:, -self.k_top(W.shape[1]):],
            axis=1
        )
        return in_idxs, out_idxs


def read_parameters(param_filename):
    f = open(param_filename, 'rb')

    layer_parameters = []
    while True:
        try:
            params = (pkl.load(f), pkl.load(f))
            layer_parameters.append(params)
        except EOFError:
            break

    f.close()

    return layer_parameters


def save(filename, in_idxs, out_idxs):
    f = open(filename, 'wb')

    for l_idx in range(len(in_idxs)):
        pkl.dump(in_idxs[l_idx], f)
        pkl.dump(out_idxs[l_idx], f)

    f.close()


def load(filename):
    f = open(filename, 'rb')

    in_idxs = []
    out_idxs = []

    while True:
        try:
            in_idxs.append(pkl.load(f))
            out_idxs.append(pkl.load(f))
        except EOFError:
            break

    f.close()

    return (in_idxs, out_idxs)


if __name__ == '__main__':
    # Parse the arguments
    usage = ('usage: %prog [options]'
             ' <parameter_file_name> <ins_and_outs_file_name>')
    pars = op(usage=usage)
    pars.add_option(
        '-k',
        '--k_percent',
        dest='k_percent',
        help='Percent k of top weights to extract.',
        metavar='K',
        default=0.1
    )
    pars.add_option(
        '-w',
        '--top_weights',
        help='Whether to identify neurons based on largest set of weights.',
        action='store_true',
        default=False
    )
    pars.add_option(
        '-n',
        '--top_neurons',
        help='Whether to identify neurons with largest incoming weights'
        ' instead of largest weights.',
        action='store_true',
        dest='top_neurons',
        default=False
    )
    pars.add_option(
        '-a',
        '--top_associations',
        help='Whether we just want the top k weights into a neuron.',
        action='store_true',
        dest='top_associations',
        default=False
    )
    pars.add_option(
        '-r',
        '--respect_outs',
        help='Whether to respect the outputs of the previous layer'
        'when deciding on current layer.',
        action='store_false',
        dest='ignore_outs',
        default=True
    )
    pars.add_option(
        '-i',
        '--first_layer_blocks',
        dest='n_first_layer_blocks',
        help='Number of blocks in first layer.',
        metavar='#',
        default=None
    )
    pars.add_option(
        '-o',
        '--last_layer_blocks',
        dest='n_last_layer_blocks',
        help='Number of blocks in last layer.',
        metavar='#',
        default=None
    )

    (options, args) = pars.parse_args()

    if ((len(args) < 2) or (
            (not path.exists(args[0]))
            and (not path.exists(args[1]))
    )):
        pars.print_usage()
        exit()

    param_filename = args[0]
    ins_and_outs_filename = args[1]

    k_percent = 1 if options.k_percent is None else float(options.k_percent)

    # Read the parameters
    # Assume stored as W followed by b for each layer
    layer_parameters = read_parameters(param_filename)

    # Identify the indices of the top weights
    #P = T.matrix('P')
    #k = T.iscalar('k')

    # Returns a matrix where each row is a list of indices for the neuron in
    # the next layer up with strong connection
    if options.top_weights:
        extractor = ExtractTopWeightsFromOuts(k_percent)
    elif options.top_associations:
        extractor = ExtractAssociations(k_percent)
    elif options.top_neurons:
        extractor = ExtractTopNeurons(k_percent)
    else:
        raise Exception('No extraction method specified.')

    in_idxs = []
    out_idxs = []
    previous_outs = None
    for (W, b) in layer_parameters:
        ins, outs = extractor.extract(W, previous_outs)

        if not options.ignore_outs:
            previous_outs = outs

        in_idxs.append(ins)
        out_idxs.append(outs)

    # Replace the number of input layer blocks as specified
    if options.n_first_layer_blocks is not None:
        in_idxs[0] = np.arange(int(options.n_first_layer_blocks))

    ipdb.set_trace()
    # Activate all output layer blocks unless otherwise specified
    if options.n_last_layer_blocks is not None:
        out_idxs[-1] = np.arange(int(options.n_last_layer_blocks))
    else:
        out_idxs[-1] = np.arange(layer_parameters[-1][0])

    if options.top_associations:
        out_idxs[-1] = repeat(out_idxs[-1])
    else:
        # Make sure that the inputs of a layer match the outputs of a the
        # previous layer
        for i in range(len(in_idxs))[1:]:
            out_idxs[i-1] = np.intersect1d(in_idxs[i], out_idxs[i-1])
            in_idxs[i] = out_idxs[i-1]

    # Save the indices of the top weights
    save(ins_and_outs_filename, in_idxs, out_idxs)

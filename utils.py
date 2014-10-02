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

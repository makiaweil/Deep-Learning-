"""
Source Code for Homework 3.b of ECBM E6040, Spring 2016, Columbia University

Instructor: Prof. Aurel A. Lazar

This code is based on
[1] http://deeplearning.net/tutorial/logreg.html
[2] http://deeplearning.net/tutorial/mlp.html
[3] http://deeplearning.net/tutorial/lenet.html
"""
import numpy

import theano
import theano.tensor as T
from theano.tensor.signal import downsample
floatX = theano.config.floatX

from hw3_utils import shared_dataset, load_data
from hw3_nn import LogisticRegression, HiddenLayer, myMLP, FConvLayer, train_nn, train_nn_alt
from hw3a import test_mlp


def Adam(cost, params, lr=0.0002, b1=0.1, b2=0.001, e=1e-8):
    updates = []
    grads = T.grad(cost, params)
    i = theano.shared(0.)
    i_t = i + 1.
    fix1 = 1. - (1. - b1)**i_t
    fix2 = 1. - (1. - b2)**i_t
    lr_t = lr * (T.sqrt(fix2) / fix1)
    for p, g in zip(params, grads):
        m = theano.shared(p.get_value() * 0.)
        v = theano.shared(p.get_value() * 0.)
        m_t = (b1 * g) + ((1. - b1) * m)
        v_t = (b2 * T.sqr(g)) + ((1. - b2) * v)
        g_t = m_t / (T.sqrt(v_t) + e)
        p_t = p - (lr_t * g_t)
        updates.append((m, m_t))
        updates.append((v, v_t))
        updates.append((p, p_t))
    updates.append((i, i_t))
    return updates

def AdamAlt(grads, params, lr=0.0002, b1=0.1, b2=0.001, e=1e-8):
    updates = []
    i = theano.shared(0.)
    i_t = i + 1.
    fix1 = 1. - (1. - b1)**i_t
    fix2 = 1. - (1. - b2)**i_t
    lr_t = lr * (T.sqrt(fix2) / fix1)
    for p, g in zip(params, grads):
        m = theano.shared(p.get_value() * 0.)#complex
        v = theano.shared(numpy.real(p.get_value()) * 0.)#real
        m_t = (b1 * g) + ((1. - b1) * m)
        v_t = (b2 * T.sqr(T.abs_(g))) + ((1. - b2) * v)#get square module
        g_t = m_t / (T.sqrt(v_t) + e)
        p_t = p - (lr_t * g_t)
        updates.append((m, m_t))
        updates.append((v, v_t))
        updates.append((p, p_t))
    updates.append((i, i_t))
    return updates

def test_lenet(learning_rate=0.1, n_epochs=1000, nkerns=[16, 512], n_hidden=500, batch_size=200, verbose=False):
    """
    Wrapper function for testing LeNet on SVHN dataset

    :type learning_rate: float
    :param learning_rate: learning rate used (factor for the stochastic
    gradient)

    :type n_epochs: int
    :param n_epochs: maximal number of epochs to run the optimizer

    :type nkerns: list of ints
    :param nkerns: number of kernels on each layer

    :type batch_size: int
    :param batch_szie: number of examples in minibatch.

    :type verbose: boolean
    :param verbose: to print out epoch summary or not to.

    """
    # learning_rate=0.1
    # n_epochs=1000
    # nkerns=[16, 512]
    # n_hidden=500
    # batch_size=200
    # verbose=True

    rng = numpy.random.RandomState(23455)

    datasets = load_data()

    train_set_x, train_set_y = datasets[0]
    valid_set_x, valid_set_y = datasets[1]
    test_set_x, test_set_y = datasets[2]

    # compute number of minibatches for training, validation and testing
    n_train_batches = train_set_x.get_value(borrow=True).shape[0]
    n_valid_batches = valid_set_x.get_value(borrow=True).shape[0]
    n_test_batches = test_set_x.get_value(borrow=True).shape[0]
    n_train_batches //= batch_size
    n_valid_batches //= batch_size
    n_test_batches //= batch_size

    # allocate symbolic variables for the data
    index = T.lscalar()  # index to a [mini]batch

    x = T.matrix('x')   # the data is presented as rasterized images
    y = T.ivector('y')  # the labels are presented as 1D vector of
                        # [int] labels

    ######################
    # BUILD ACTUAL MODEL #
    ######################
    print('... building the model')

    # Reshape matrix of rasterized images of shape (batch_size, 3 * 32 * 32)
    # to a 4D tensor, compatible with our LeNetConvPoolLayer
    layer0_input = x.reshape((batch_size, 3, 32, 32))

    # TODO: Construct the first convolutional pooling layer
    layer0 = FConvLayer(
       rng,
       input=layer0_input,
       image_shape=(batch_size,3,32,32),
       filter_shape=(nkerns[0],3,5,5),
    )

    # TODO: Construct the second convolutional pooling layer
    # layer1 = LeNetConvPoolLayer(
    #    rng,
    #    input=layer0.output,
    #    image_shape=(batch_size,nkerns[0],14,14),
    #    filter_shape=(nkerns[1],nkerns[0],5,5),
    # )

    # the HiddenLayer being fully-connected, it operates on 2D matrices of
    # shape (batch_size, num_pixels) (i.e matrix of rasterized images).
    layer2_input = layer0.output.flatten(2)

    # TODO: construct a fully-connected sigmoidal layer
    layer2 = HiddenLayer(
       rng,
       input=layer2_input,
       n_in=nkerns[0]*28*28,
       n_out=n_hidden,
    )

    # TODO: classify the values of the fully-connected sigmoidal layer
    layer3 = LogisticRegression(
         input=layer2.output,
         n_in=n_hidden,
    n_out=10)

    # the cost we minimize during training is the NLL of the model
    cost = layer3.negative_log_likelihood(y)

    # create a function to compute the mistakes that are made by the model
    test_model = theano.function(
        [index],
        layer3.errors(y),
        givens={
            x: test_set_x[index * batch_size: (index + 1) * batch_size],
            y: test_set_y[index * batch_size: (index + 1) * batch_size]
        }
    )

    validate_model = theano.function(
        [index],
        layer3.errors(y),
        givens={
            x: valid_set_x[index * batch_size: (index + 1) * batch_size],
            y: valid_set_y[index * batch_size: (index + 1) * batch_size]
        }
    )

    # TODO: create a list of all model parameters to be fit by gradient descent
    params = layer3.params + layer2.params + layer0.params
    paramsReal = layer0.paramsManualReal

    # create a list of gradients for all model parameters
    # grads = T.grad(cost, params)

    # train_model is a function that updates the model parameters by
    # SGD Since this model has many parameters, it would be tedious to
    # manually create an update rule for each model parameter. We thus
    # create the updates list by automatically looping over all
    # (params[i], grads[i]) pairs.
    # updates = [
    #     (param_i, param_i - learning_rate * grad_i)
    #     for param_i, grad_i in zip(params, grads)
    # ]

    updates = Adam(cost,params)
    manualGradients = T.grad(cost,paramsReal)#information extraction

    train_model = theano.function(
        [index],
        [cost]+manualGradients,
        updates=updates,
        givens={
            x: train_set_x[index * batch_size: (index + 1) * batch_size],
            y: train_set_y[index * batch_size: (index + 1) * batch_size]
        }
    )
    #bridge with numpy ffts to get freqGradient from manualGradients output
    paramsComplex = layer0.paramsManualComplex
    freqGradients = [T.ztensor4('complexGradients') for i in paramsComplex]
    manualUpdates = AdamAlt(freqGradients,paramsComplex)

    manual_train_model = theano.function(
        freqGradients,
        [],
        updates=manualUpdates
    )

    ###############
    # TRAIN MODEL #
    ###############
    print('... training')
    filters = (paramsReal,paramsComplex)
    train_nn_alt(train_model, manual_train_model, filters, validate_model, test_model,
        n_train_batches, n_valid_batches, n_test_batches, n_epochs, verbose)



"""
Source Code for Homework 3.a of ECBM E6040, Spring 2016, Columbia University

Instructor: Prof. Aurel A. Lazar

This code is based on
[1] http://deeplearning.net/tutorial/logreg.html
[2] http://deeplearning.net/tutorial/mlp.html
[3] http://deeplearning.net/tutorial/lenet.html
"""
import numpy

import theano
import theano.tensor as T
from theano.tensor.shared_randomstreams import RandomStreams

from hw3_utils import shared_dataset, load_data
from hw3_nn import LogisticRegression, HiddenLayer, myMLP, LeNetConvPoolLayer, train_nn
import matplotlib.pyplot as plt
# TODO
def noise_injection(train_set,noise):
    
   
    A = numpy.random.normal(0,noise,train_set.shape)
    train_set=train_set+A
    return train_set

def noise_injection_weight(W0,W,noise):
    if W0.name=='W':
        #print('W')
        rng=T.shared_randomstreams.RandomStreams()
        A=T.cast(rng.normal(size=W.shape,avg= 0,std=noise), theano.config.floatX)
    #print(W.name)
    #xx= T.TensorType(theano.config.floatX, broadcastable=())
    #tshape=theano.function(inputs=[xx], outputs=T.shape(xx))
    #print(T.shape(W[0]).eval())
    #A = numpy.random.normal(0,noise,tshape(W))
    #A_shared=theano.shared(A)
        W=W+A
    return W
    

# TODO
def translate_image(train_set,direction):
    N=train_set.shape[1]
    n=train_set.shape[0]
    if (direction=="right"):
        for i in range(N-1):
            train_set[:,N-1-i]=train_set[:,N-1-i-1]
        train_set[:,0]=numpy.zeros((n,))
    elif (direction=="left"):
        for i in range(N-1):
            train_set[:,i]=train_set[:,i+1]
        train_set[:,N-1]=numpy.zeros((n,))
    elif (direction=="up"):
        for i in range(n-1):
            train_set[i,:]=train_set[i+1,:]
        train_set[n-1,:]=numpy.zeros((1,N))    
    elif (direction=="down"):
        for i in range(n-1):
            train_set[n-1-i,:]=train_set[n-2-i,:]
        train_set[0,:]=numpy.zeros((1,N))       
    return train_set
    

# TODO
def test_mlp(learning_rate=0.01, L1_reg=0.00, L2_reg=0.0001, n_epochs=100,
             batch_size=128, n_hidden=500, n_hiddenLayers=3,
             verbose=False, smaller_set=True):
    """
    Wrapper function for training and testing MLP

    :type learning_rate: float
    :param learning_rate: learning rate used (factor for the stochastic
    gradient.

    :type L1_reg: float
    :param L1_reg: L1-norm's weight when added to the cost (see
    regularization).

    :type L2_reg: float
    :param L2_reg: L2-norm's weight when added to the cost (see
    regularization).

    :type n_epochs: int
    :param n_epochs: maximal number of epochs to run the optimizer.

    :type batch_size: int
    :param batch_szie: number of examples in minibatch.

    :type n_hidden: int or list of ints
    :param n_hidden: number of hidden units. If a list, it specifies the
    number of units in each hidden layers, and its length should equal to
    n_hiddenLayers.

    :type n_hiddenLayers: int
    :param n_hiddenLayers: number of hidden layers.

    :type verbose: boolean
    :param verbose: to print out epoch summary or not to.

    :type smaller_set: boolean
    :param smaller_set: to use the smaller dataset or not to.

    """

    # load the dataset; download the dataset if it is not present
    if smaller_set:
        datasets = load_data(ds_rate=5)
    else:
        datasets = load_data()

    train_set_x, train_set_y = datasets[0]
    valid_set_x, valid_set_y = datasets[1]
    test_set_x, test_set_y = datasets[2]

    # compute number of minibatches for training, validation and testing
    n_train_batches = train_set_x.get_value(borrow=True).shape[0] // batch_size
    n_valid_batches = valid_set_x.get_value(borrow=True).shape[0] // batch_size
    n_test_batches = test_set_x.get_value(borrow=True).shape[0] // batch_size

    ######################
    # BUILD ACTUAL MODEL #
    ######################
    print('... building the model')

    # allocate symbolic variables for the data
    index = T.lscalar()  # index to a [mini]batch
    x = T.matrix('x')  # the data is presented as rasterized images
    y = T.ivector('y')  # the labels are presented as 1D vector of
                        # [int] labels

    rng = numpy.random.RandomState(1234)

    # TODO: construct a neural network, either MLP or CNN.
    classifier = myMLP(
        rng=rng,
        input=x,
        n_in=32*32*3,
        n_hidden=n_hidden,
        n_hiddenLayers=n_hiddenLayers,
        n_out=10
    )

    # the cost we minimize during training is the negative log likelihood of
    # the model plus the regularization terms (L1 and L2); cost is expressed
    # here symbolically
    cost = (
        classifier.negative_log_likelihood(y)
        + L1_reg * classifier.L1
        + L2_reg * classifier.L2_sqr
    )

    # compiling a Theano function that computes the mistakes that are made
    # by the model on a minibatch
    test_model = theano.function(
        inputs=[index],
        outputs=classifier.errors(y),
        givens={
            x: test_set_x[index * batch_size:(index + 1) * batch_size],
            y: test_set_y[index * batch_size:(index + 1) * batch_size]
        }
    )

    validate_model = theano.function(
        inputs=[index],
        outputs=classifier.errors(y),
        givens={
            x: valid_set_x[index * batch_size:(index + 1) * batch_size],
            y: valid_set_y[index * batch_size:(index + 1) * batch_size]
        }
    )

    # compute the gradient of cost with respect to theta (sotred in params)
    # the resulting gradients will be stored in a list gparams
    gparams = [T.grad(cost, param) for param in classifier.params]

    # specify how to update the parameters of the model as a list of
    # (variable, update expression) pairs

    # given two lists of the same length, A = [a1, a2, a3, a4] and
    # B = [b1, b2, b3, b4], zip generates a list C of same size, where each
    # element is a pair formed from the two lists :
    #    C = [(a1, b1), (a2, b2), (a3, b3), (a4, b4)]
    updates = [
        (param, param - learning_rate * gparam)
        for param, gparam in zip(classifier.params, gparams)
    ]

    # compiling a Theano function `train_model` that returns the cost, but
    # in the same time updates the parameter of the model based on the rules
    # defined in `updates`
    train_model = theano.function(
        inputs=[index],
        outputs=cost,
        updates=updates,
        givens={
            x: train_set_x[index * batch_size: (index + 1) * batch_size],
            y: train_set_y[index * batch_size: (index + 1) * batch_size]
        }
    )

    ###############
    # TRAIN MODEL #
    ###############
    print('... training')

    train_nn(train_model, validate_model, test_model,
        n_train_batches, n_valid_batches, n_test_batches, n_epochs, verbose)

# TODO
def test_noise_inject_at_input(learning_rate=0.01, L1_reg=0.00,
             L2_reg=0.0001, n_epochs=100,
             batch_size=128, n_hidden=500, n_hiddenLayers=3,
             verbose=False,noise=1):
    """
    Wrapper function for experiment of noise injection at input

    :type learning_rate: float
    :param learning_rate: learning rate used (factor for the stochastic
    gradient.

    :type L1_reg: float
    :param L1_reg: L1-norm's weight when added to the cost (see
    regularization).

    :type L2_reg: float
    :param L2_reg: L2-norm's weight when added to the cost (see
    regularization).

    :type n_epochs: int
    :param n_epochs: maximal number of epochs to run the optimizer.

    :type batch_size: int
    :param batch_szie: number of examples in minibatch.

    :type n_hidden: int or list of ints
    :param n_hidden: number of hidden units. If a list, it specifies the
    number of units in each hidden layers, and its length should equal to
    n_hiddenLayers.

    :type n_hiddenLayers: int
    :param n_hiddenLayers: number of hidden layers.

    :type verbose: boolean
    :param verbose: to print out epoch summary or not to.

    :type smaller_set: boolean
    :param smaller_set: to use the smaller dataset or not to.

    """
    rng = numpy.random.RandomState(23455)

    # Load down-sampled dataset in raw format (numpy.darray, not Theano.shared)
    # train_set, valid_set, test_set format: tuple(input, target)
    # input is a numpy.ndarray of 2 dimensions (a matrix), where each row
    # corresponds to an example. target is a numpy.ndarray of 1 dimension
    # (vector) that has the same length as the number of rows in the input.

    # Load the smaller dataset in raw format, since we need to preprocess it
    train_set, valid_set, test_set = load_data(ds_rate=5, theano_shared=False)

    # Repeat the training set 5 times
    train_set[0] = numpy.tile(train_set[0], [5,1])
    train_set[1] = numpy.tile(train_set[1], 5)

    # TODO: injection noise to the training set
    # call noise_injection() ...
    
    train_set[0]=noise_injection(train_set[0],noise)
    
    
    
    # Convert raw dataset to Theano shared variables.
    test_set_x, test_set_y = shared_dataset(test_set)
    valid_set_x, valid_set_y = shared_dataset(valid_set)
    train_set_x, train_set_y = shared_dataset(train_set)

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

    # allocate symbolic variables for the data
    index = T.lscalar()  # index to a [mini]batch
    x = T.matrix('x')  # the data is presented as rasterized images
    y = T.ivector('y')  # the labels are presented as 1D vector of
                        # [int] labels

    rng = numpy.random.RandomState(1234)

    classifier = myMLP(
        rng=rng,
        input=x,
        n_in=32*32*3,
        n_hidden=n_hidden,
        n_hiddenLayers=n_hiddenLayers,
        n_out=10
    )

    # the cost we minimize during training is the negative log likelihood of
    # the model plus the regularization terms (L1 and L2); cost is expressed
    # here symbolically
    cost = (
        classifier.negative_log_likelihood(y)
        + L1_reg * classifier.L1
        + L2_reg * classifier.L2_sqr
    )

    # compiling a Theano function that computes the mistakes that are made
    # by the model on a minibatch
    test_model = theano.function(
        inputs=[index],
        outputs=classifier.errors(y),
        givens={
            x: test_set_x[index * batch_size:(index + 1) * batch_size],
            y: test_set_y[index * batch_size:(index + 1) * batch_size]
        }
    )

    validate_model = theano.function(
        inputs=[index],
        outputs=classifier.errors(y),
        givens={
            x: valid_set_x[index * batch_size:(index + 1) * batch_size],
            y: valid_set_y[index * batch_size:(index + 1) * batch_size]
        }
    )

    # compute the gradient of cost with respect to theta (sotred in params)
    # the resulting gradients will be stored in a list gparams
    gparams = [T.grad(cost, param) for param in classifier.params]

    # specify how to update the parameters of the model as a list of
    # (variable, update expression) pairs

    # given two lists of the same length, A = [a1, a2, a3, a4] and
    # B = [b1, b2, b3, b4], zip generates a list C of same size, where each
    # element is a pair formed from the two lists :
    #    C = [(a1, b1), (a2, b2), (a3, b3), (a4, b4)]
    updates = [
        (param, param - learning_rate * gparam)
        for param, gparam in zip(classifier.params, gparams)
    ]

    # compiling a Theano function `train_model` that returns the cost, but
    # in the same time updates the parameter of the model based on the rules
    # defined in `updates`
    train_model = theano.function(
        inputs=[index],
        outputs=cost,
        updates=updates,
        givens={
            x: train_set_x[index * batch_size: (index + 1) * batch_size],
            y: train_set_y[index * batch_size: (index + 1) * batch_size]
        }
    )

    ###############
    # TRAIN MODEL #
    ###############
    print('... training')

    train_nn(train_model, validate_model, test_model,
        n_train_batches, n_valid_batches, n_test_batches, n_epochs, verbose)

# TODO
def test_noise_injection_at_weight(learning_rate=0.01,
             L1_reg=0.00, L2_reg=0.0001, n_epochs=100,
             batch_size=128, n_hidden=500, n_hiddenLayers=3,
             verbose=False,noise=1):
    """
    Wrapper function for experiment of noise injection at weights

    :type learning_rate: float
    :param learning_rate: learning rate used (factor for the stochastic
    gradient.

    :type L1_reg: float
    :param L1_reg: L1-norm's weight when added to the cost (see
    regularization).

    :type L2_reg: float
    :param L2_reg: L2-norm's weight when added to the cost (see
    regularization).

    :type n_epochs: int
    :param n_epochs: maximal number of epochs to run the optimizer.

    :type batch_size: int
    :param batch_szie: number of examples in minibatch.

    :type n_hidden: int or list of ints
    :param n_hidden: number of hidden units. If a list, it specifies the
    number of units in each hidden layers, and its length should equal to
    n_hiddenLayers.

    :type n_hiddenLayers: int
    :param n_hiddenLayers: number of hidden layers.

    :type verbose: boolean
    :param verbose: to print out epoch summary or not to.

    :type smaller_set: boolean
    :param smaller_set: to use the smaller dataset or not to.

    """
    rng = numpy.random.RandomState(23455)

    # Load down-sampled dataset in raw format (numpy.darray, not Theano.shared)
    # train_set, valid_set, test_set format: tuple(input, target)
    # input is a numpy.ndarray of 2 dimensions (a matrix), where each row
    # corresponds to an example. target is a numpy.ndarray of 1 dimension
    # (vector) that has the same length as the number of rows in the input.

    # Load the smaller dataset
    datasets = load_data(ds_rate=5)
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

    # allocate symbolic variables for the data
    index = T.lscalar()  # index to a [mini]batch
    x = T.matrix('x')  # the data is presented as rasterized images
    y = T.ivector('y')  # the labels are presented as 1D vector of
                        # [int] labels

    rng = numpy.random.RandomState(1234)

    classifier = myMLP(
        rng=rng,
        input=x,
        n_in=32*32*3,
        n_hidden=n_hidden,
        n_hiddenLayers=n_hiddenLayers,
        n_out=10
    )

    # the cost we minimize during training is the negative log likelihood of
    # the model plus the regularization terms (L1 and L2); cost is expressed
    # here symbolically
    cost = (
        classifier.negative_log_likelihood(y)
        + L1_reg * classifier.L1
        + L2_reg * classifier.L2_sqr
    )

    # compiling a Theano function that computes the mistakes that are made
    # by the model on a minibatch
    test_model = theano.function(
        inputs=[index],
        outputs=classifier.errors(y),
        givens={
            x: test_set_x[index * batch_size:(index + 1) * batch_size],
            y: test_set_y[index * batch_size:(index + 1) * batch_size]
        }
    )

    validate_model = theano.function(
        inputs=[index],
        outputs=classifier.errors(y),
        givens={
            x: valid_set_x[index * batch_size:(index + 1) * batch_size],
            y: valid_set_y[index * batch_size:(index + 1) * batch_size]
        }
    )

    # compute the gradient of cost with respect to theta (sotred in params)
    # the resulting gradients will be stored in a list gparams
    gparams = [T.grad(cost, param) for param in classifier.params]

    # specify how to update the parameters of the model as a list of
    # (variable, update expression) pairs

    # given two lists of the same length, A = [a1, a2, a3, a4] and
    # B = [b1, b2, b3, b4], zip generates a list C of same size, where each
    # element is a pair formed from the two lists :
    #    C = [(a1, b1), (a2, b2), (a3, b3), (a4, b4)]

    # TODO: modify updates to inject noise to the weight
    updates = [
        (param, noise_injection_weight(param,param - learning_rate * gparam,noise)) 
        for param, gparam in zip(classifier.params, gparams)
    ]
    #print([param.name for param in classifier.params])
    #n=len(updates)/2
    #for j in range(n):
    #    updates[j*2][0]=noise_injection_weight(updates[j*2][0],noise)
        


    # compiling a Theano function `train_model` that returns the cost, but
    # in the same time updates the parameter of the model based on the rules
    # defined in `updates`
    train_model = theano.function(
        inputs=[index],
        outputs=cost,
        updates=updates,
        givens={
            x: train_set_x[index * batch_size: (index + 1) * batch_size],
            y: train_set_y[index * batch_size: (index + 1) * batch_size]
        }
    )

    ###############
    # TRAIN MODEL #
    ###############
    print('... training')

    train_nn(train_model, validate_model, test_model,
        n_train_batches, n_valid_batches, n_test_batches, n_epochs, verbose)

def test_data_augmentation(learning_rate=0.01,
             L1_reg=0.00, L2_reg=0.0001, n_epochs=100,
             batch_size=128, n_hidden=500, n_hiddenLayers=3,
             verbose=False):
    """
    Wrapper function for experiment of data augmentation

    :type learning_rate: float
    :param learning_rate: learning rate used (factor for the stochastic
    gradient.

    :type L1_reg: float
    :param L1_reg: L1-norm's weight when added to the cost (see
    regularization).

    :type L2_reg: float
    :param L2_reg: L2-norm's weight when added to the cost (see
    regularization).

    :type n_epochs: int
    :param n_epochs: maximal number of epochs to run the optimizer.

    :type batch_size: int
    :param batch_szie: number of examples in minibatch.

    :type n_hidden: int or list of ints
    :param n_hidden: number of hidden units. If a list, it specifies the
    number of units in each hidden layers, and its length should equal to
    n_hiddenLayers.

    :type n_hiddenLayers: int
    :param n_hiddenLayers: number of hidden layers.

    :type verbose: boolean
    :param verbose: to print out epoch summary or not to.

    :type smaller_set: boolean
    :param smaller_set: to use the smaller dataset or not to.

    """
    rng = numpy.random.RandomState(23455)

    # Load down-sampled dataset in raw format (numpy.darray, not Theano.shared)
    # train_set, valid_set, test_set format: tuple(input, target)
    # input is a numpy.ndarray of 2 dimensions (a matrix), where each row
    # corresponds to an example. target is a numpy.ndarray of 1 dimension
    # (vector) that has the same length as the number of rows in the input.

    # Load the smaller dataset in raw Format, since we need to preprocess it
    train_set, valid_set, test_set = load_data(ds_rate=5, theano_shared=False)

    # Repeat the training set 5 times
    train_set[1] = numpy.tile(train_set[1], 5)

    # TODO: translate the dataset
    train_set_x_u = translate_image(train_set[0],"up")
    train_set_x_d = translate_image(train_set[0],"down")
    train_set_x_r = translate_image(train_set[0],"right")
    train_set_x_l = translate_image(train_set[0],"left")

    # Stack the original dataset and the synthesized datasets
    train_set[0] = numpy.vstack((train_set[0],
                       train_set_x_u,
                       train_set_x_d,
                       train_set_x_r,
                       train_set_x_l))

    # Convert raw dataset to Theano shared variables.
    test_set_x, test_set_y = shared_dataset(test_set)
    valid_set_x, valid_set_y = shared_dataset(valid_set)
    train_set_x, train_set_y = shared_dataset(train_set)

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

    # allocate symbolic variables for the data
    index = T.lscalar()  # index to a [mini]batch
    x = T.matrix('x')  # the data is presented as rasterized images
    y = T.ivector('y')  # the labels are presented as 1D vector of
                        # [int] labels

    rng = numpy.random.RandomState(1234)

    classifier = myMLP(
        rng=rng,
        input=x,
        n_in=32*32*3,
        n_hidden=n_hidden,
        n_hiddenLayers=n_hiddenLayers,
        n_out=10
    )

    # the cost we minimize during training is the negative log likelihood of
    # the model plus the regularization terms (L1 and L2); cost is expressed
    # here symbolically
    cost = (
        classifier.negative_log_likelihood(y)
        + L1_reg * classifier.L1
        + L2_reg * classifier.L2_sqr
    )

    # compiling a Theano function that computes the mistakes that are made
    # by the model on a minibatch
    test_model = theano.function(
        inputs=[index],
        outputs=classifier.errors(y),
        givens={
            x: test_set_x[index * batch_size:(index + 1) * batch_size],
            y: test_set_y[index * batch_size:(index + 1) * batch_size]
        }
    )

    validate_model = theano.function(
        inputs=[index],
        outputs=classifier.errors(y),
        givens={
            x: valid_set_x[index * batch_size:(index + 1) * batch_size],
            y: valid_set_y[index * batch_size:(index + 1) * batch_size]
        }
    )

    # compute the gradient of cost with respect to theta (sotred in params)
    # the resulting gradients will be stored in a list gparams
    gparams = [T.grad(cost, param) for param in classifier.params]

    # specify how to update the parameters of the model as a list of
    # (variable, update expression) pairs

    # given two lists of the same length, A = [a1, a2, a3, a4] and
    # B = [b1, b2, b3, b4], zip generates a list C of same size, where each
    # element is a pair formed from the two lists :
    #    C = [(a1, b1), (a2, b2), (a3, b3), (a4, b4)]
    updates = [
        (param, param - learning_rate * gparam)
        for param, gparam in zip(classifier.params, gparams)
    ]

    # compiling a Theano function `train_model` that returns the cost, but
    # in the same time updates the parameter of the model based on the rules
    # defined in `updates`
    
    
    train_model = theano.function(
        inputs=[index],
        outputs=cost,
        updates=updates,
        givens={
            x: train_set_x[index * batch_size: (index + 1) * batch_size],
            y: train_set_y[index * batch_size: (index + 1) * batch_size]
        }
    )

    ###############
    # TRAIN MODEL #
    ###############
    print('... training')

    train_nn(train_model, validate_model, test_model,
        n_train_batches, n_valid_batches, n_test_batches, n_epochs, verbose)

# TODO
def test_adversarial_example(learning_rate=0.01, L1_reg=0.00, L2_reg=0.0001, n_epochs=100,
             batch_size=128, n_hidden=500, n_hiddenLayers=3,
             verbose=False, smaller_set=True):
    """
    Wrapper function for testing adversarial examples
    """
    
    
    
    
  
    # load the dataset; download the dataset if it is not present
    if smaller_set:
        datasets = load_data(ds_rate=5)
    else:
        datasets = load_data()

    train_set_x, train_set_y = datasets[0]
    valid_set_x, valid_set_y = datasets[1]
    test_set_x, test_set_y = datasets[2]

    # compute number of minibatches for training, validation and testing
    n_train_batches = train_set_x.get_value(borrow=True).shape[0] // batch_size
    n_valid_batches = valid_set_x.get_value(borrow=True).shape[0] // batch_size
    n_test_batches = test_set_x.get_value(borrow=True).shape[0] // batch_size

    ######################
    # BUILD ACTUAL MODEL #
    ######################
    print('... building the model')

    # allocate symbolic variables for the data
    index = T.lscalar()  # index to a [mini]batch
    x = T.matrix('x')  # the data is presented as rasterized images
    y = T.ivector('y')  # the labels are presented as 1D vector of
                        # [int] labels
    xx = T.matrix('xx')

    rng = numpy.random.RandomState(1234)

    # TODO: construct a neural network, either MLP or CNN.
    classifier = myMLP(
        rng=rng,
        input=x,
        n_in=32*32*3,
        n_hidden=n_hidden,
        n_hiddenLayers=n_hiddenLayers,
        n_out=10
    )

    # the cost we minimize during training is the negative log likelihood of
    # the model plus the regularization terms (L1 and L2); cost is expressed
    # here symbolically
    cost = (
        classifier.negative_log_likelihood(y)
        + L1_reg * classifier.L1
        + L2_reg * classifier.L2_sqr
    )

    # compiling a Theano function that computes the mistakes that are made
    # by the model on a minibatch
    test_model = theano.function(
        inputs=[index],
        outputs=classifier.errors(y),
        givens={
            x: test_set_x[index * batch_size:(index + 1) * batch_size],
            y: test_set_y[index * batch_size:(index + 1) * batch_size]
        }
    )

    
    #  Return "0" if good prediction "1" else, for the test set
    predict = theano.function(
        inputs=[index],
        outputs=classifier.errors(y),
        givens={
            x: test_set_x[index:index+1],
            y: test_set_y[index:index+1]
        }
    )
    #  Return the output (probabilities) of  example of the test set
    probs = theano.function(
        inputs=[index],
        outputs=classifier.logRegressionLayer.p_y_given_x,
        givens={
            x: test_set_x[index:index+1 ]
           
        }
    )
    
    
    
    
    
    
    validate_model = theano.function(
        inputs=[index],
        outputs=classifier.errors(y),
        givens={
            x: valid_set_x[index * batch_size:(index + 1) * batch_size],
            y: valid_set_y[index * batch_size:(index + 1) * batch_size]
        }
    )

    # compute the gradient of cost with respect to theta (sotred in params)
    # the resulting gradients will be stored in a list gparams
    gparams = [T.grad(cost, param) for param in classifier.params]

    # specify how to update the parameters of the model as a list of
    # (variable, update expression) pairs

    # given two lists of the same length, A = [a1, a2, a3, a4] and
    # B = [b1, b2, b3, b4], zip generates a list C of same size, where each
    # element is a pair formed from the two lists :
    #    C = [(a1, b1), (a2, b2), (a3, b3), (a4, b4)]
    updates = [
        (param, param - learning_rate * gparam)
        for param, gparam in zip(classifier.params, gparams)
    ]

    # compiling a Theano function `train_model` that returns the cost, but
    # in the same time updates the parameter of the model based on the rules
    # defined in `updates`
    train_model = theano.function(
        inputs=[index],
        outputs=cost,
        updates=updates,
        givens={
            x: train_set_x[index * batch_size: (index + 1) * batch_size],
            y: train_set_y[index * batch_size: (index + 1) * batch_size]
        }
    )

    ###############
    # TRAIN MODEL #
    ###############
    print('... training')

    train_nn(train_model, validate_model, test_model,
        n_train_batches, n_valid_batches, n_test_batches, n_epochs, verbose)
    
    
    
   
    
    
   
    # Adversarial x
    ginput=T.sgn(T.grad(cost, x))
    xx=x+0.001*ginput
    
    
    #  Return the adversarial test set
    xnoised=theano.function(
        inputs=[],
        outputs=xx,
        givens={
            x:test_set_x,
            y:test_set_y
            }
    )
    
    test_set_xn=xnoised()
    shared_xn = theano.shared(numpy.asarray(test_set_xn,
                                           dtype=theano.config.floatX),
                             borrow=True)
    
    
    #  Return "0" if good prediction "1" else, for the adversarial example
    predictn = theano.function(
        inputs=[index],
        outputs=classifier.errors(y),
        givens={
            x: shared_xn[index:index+1],
            y: test_set_y[index:index+1]
        }
    )
    
    
    #  Return the output (probabilities) of the adversarial example
    probsn = theano.function(
        inputs=[index],
        outputs=classifier.logRegressionLayer.p_y_given_x,
        givens={
            x: shared_xn[index:index+1 ]
           
        }
     )
     
        
    # Return the example    
    ret_example= theano.function(
        inputs=[index],
        outputs=x,
        givens={
            x: test_set_x[index:index+1 ]          
        }
     )
    
    #  At the first good example but bad adversarial example
    #  return the images and the probabilities
    for i in range(test_set_xn.shape[1]):
        if predict(i)==0.0 and predictn(i)==1.0:
            print("Being fooled at example number "+str(i))
            probs=probs(i)[0]
            probsadv=probsn(i)[0]
            im=ret_example(i).reshape((3,32,32))            
            imadv=test_set_xn[i,:].reshape((3,32,32))
            Im=numpy.array(im.transpose((1,2,0)),dtype=numpy.float32)
            Imadv=numpy.array(imadv.transpose((1,2,0)),dtype=numpy.float32)
            return(probs,probsadv,Im,Imadv)
            break
    
    
    
    
    
    
   
   
    
    

if __name__ == "__main__":
     test_mlp()
    # test_noise_inject_at_weight()
    # test_noise_inject_at_input()
    # test_data_augmentation()
    # test_adversarial_example()

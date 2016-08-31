import numpy

import theano
import theano.tensor as T

from project_nn import *
from project_utils import *

def network6(nb_labels,spectral,batch_size,n_epochs,verbose):
    """
    architecture 6

    :type nb_labels: int
    :param nb_labels: number of classes and determines dataset

    :type spectral: boolean
    :param spectral: use spectral filter parametrisation or not
    """

    # spectral=True
    # nb_labels=10
    # n_epochs=1000
    # batch_size=200
    # verbose=True
    # hyper params
    rng = numpy.random.RandomState(23455)

    if nb_labels == 10:
        datasets = load_data_cifar10()
    else:
        datasets = load_data_cifar100()

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
    image_shape = 32
    image_channels = 3
    layer0_input = x.reshape((batch_size, image_channels, image_shape, image_shape))
    #layers
    layer0 = FConvLayer(
       spectral, 
       rng,
       input=layer0_input,
       image_shape=(batch_size,image_channels,image_shape,image_shape),
       filter_shape=(96,image_channels,3,3)
    )
    image_shape = shapeConvo(image_shape,3)

    layer1 = MP(layer0.output)
    image_shape = shapeMax(image_shape,3,2)

    layer2 = FConvLayer(
       spectral,
       rng,
       input=layer1.output,
       image_shape=(batch_size,96,image_shape,image_shape),
       filter_shape=(192,96,3,3),
    )
    image_shape = shapeConvo(image_shape,3)

    layer3 = MP(layer2.output)
    image_shape = shapeMax(image_shape,3,2)
    ##### FLLAAATTTTT #######
    layer3_output = layer3.output.flatten(2)
    ##### FLLAAATTTTT #######
    layer4 = HiddenLayer(
       rng,
       input=layer3_output,
       n_in=192*(image_shape**2),
       n_out=1024,
    )
    layer5 = HiddenLayer(
       rng,
       input=layer4.output,
       n_in=1024,
       n_out=512,
    )
    layer6 = LogisticRegression(
        input=layer5.output,
        n_in=512,
    	n_out=nb_labels)
    param_layers = [layer0,layer2,layer4,layer5,layer6]
    convo_layers = [layer0,layer2]
    last_layer = layer6

    cost = last_layer.negative_log_likelihood(y)

    test_model = theano.function(
        [index],
        last_layer.errors(y),
        givens={
            x: test_set_x[index * batch_size: (index + 1) * batch_size],
            y: test_set_y[index * batch_size: (index + 1) * batch_size]
        }
    )

    validate_model = theano.function(
        [index],
        last_layer.errors(y),
        givens={
            x: valid_set_x[index * batch_size: (index + 1) * batch_size],
            y: valid_set_y[index * batch_size: (index + 1) * batch_size]
        }
    )


    if spectral:
        params = [par for l in param_layers for par in l.params]
        paramsReal = [par for l in convo_layers for par in l.paramsManualReal]
        #
        updates = Adam(cost,params)
        manualGradients = T.grad(cost,paramsReal)#information extraction
        #
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
        paramsComplex = [par for l in convo_layers for par in l.paramsManualComplex]
        freqGradients = [T.ztensor4('complexGradients') for i in paramsComplex]
        manualUpdates = AdamAlt(freqGradients,paramsComplex)
        #
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
        return train_nn_alt(train_model, manual_train_model, filters, validate_model, test_model,
        n_train_batches, n_valid_batches, n_test_batches, n_epochs, verbose)
    else :
        params = [par for l in param_layers for par in l.params]
        updates = Adam(cost,params)
        train_model = theano.function(
            [index],
            cost,
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
        return train_nn(train_model, validate_model, test_model,
        n_train_batches, n_valid_batches, n_test_batches, n_epochs, verbose)

def network7(nb_labels,spectral,batch_size,n_epochs,verbose):
    """
    architecture 7

    :type nb_labels: int
    :param nb_labels: number of classes and determines dataset

    :type spectral: boolean
    :param spectral: use spectral filter parametrisation or not
    """
    # spectral=True
    # nb_labels=10
    # n_epochs=2
    # batch_size=200
    # verbose=True
    # hyper params
    rng = numpy.random.RandomState(23455)

    if nb_labels == 10:
        datasets = load_data_cifar10()
    else:
        datasets = load_data_cifar100()

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
    image_shape = 32
    image_channels = 3
    layer0_input = x.reshape((batch_size, image_channels, image_shape, image_shape))
    #layers
    layer0 = FConvLayer(
       spectral, 
       rng,
       input=layer0_input,
       image_shape=(batch_size,image_channels,image_shape,image_shape),
       filter_shape=(96,image_channels,3,3)
    )
    image_shape = shapeConvo(image_shape,3)

    layer1 = FConvLayer(
       spectral, 
       rng,
       input=layer0.output,
       image_shape=(batch_size,96,image_shape,image_shape),
       filter_shape=(96,96,3,3)
    )
    image_shape = shapeConvo(image_shape,3)

    layer2 = MP(layer1.output)
    image_shape = shapeMax(image_shape,3,2)

    layer3 = FConvLayer(
       spectral,
       rng,
       input=layer2.output,
       image_shape=(batch_size,96,image_shape,image_shape),
       filter_shape=(192,96,3,3),
    )
    image_shape = shapeConvo(image_shape,3)

    layer4 = FConvLayer(
       spectral,
       rng,
       input=layer3.output,
       image_shape=(batch_size,192,image_shape,image_shape),
       filter_shape=(192,192,3,3),
    )
    image_shape = shapeConvo(image_shape,3)

    layer5 = FConvLayer(
       spectral,
       rng,
       input=layer4.output,
       image_shape=(batch_size,192,image_shape,image_shape),
       filter_shape=(192,192,3,3),
    )
    image_shape = shapeConvo(image_shape,3)

    layer6 = MP(layer5.output)
    image_shape = shapeMax(image_shape,3,2)

    layer7 = FConvLayer(
       spectral,
       rng,
       input=layer6.output,
       image_shape=(batch_size,192,image_shape,image_shape),
       filter_shape=(192,192,1,1),
    )
    image_shape = shapeConvo(image_shape,1)

    layer8 = FConvLayer(
       spectral,
       rng,
       input=layer7.output,
       image_shape=(batch_size,192,image_shape,image_shape),
       filter_shape=(100,192,1,1),
    )
    image_shape = shapeConvo(image_shape,1)# size 3 \times 3 
    ##### FLLAAATTTTT #######
    layer8_output = GA(layer8.output)
    ##### FLLAAATTTTT #######
    layer9 = LogisticRegression(
        input=layer8_output,
        n_in=100,
        n_out=nb_labels)

    param_layers = [layer0,layer1,layer3,layer4,layer5,layer7,layer8,layer9]
    convo_layers = [layer0,layer1,layer3,layer4,layer5,layer7,layer8]
    last_layer = layer9

    cost = last_layer.negative_log_likelihood(y)

    test_model = theano.function(
        [index],
        last_layer.errors(y),
        givens={
            x: test_set_x[index * batch_size: (index + 1) * batch_size],
            y: test_set_y[index * batch_size: (index + 1) * batch_size]
        }
    )

    validate_model = theano.function(
        [index],
        last_layer.errors(y),
        givens={
            x: valid_set_x[index * batch_size: (index + 1) * batch_size],
            y: valid_set_y[index * batch_size: (index + 1) * batch_size]
        }
    )


    if spectral:
        params = [par for l in param_layers for par in l.params]
        paramsReal = [par for l in convo_layers for par in l.paramsManualReal]
        #
        updates = Adam(cost,params)
        manualGradients = T.grad(cost,paramsReal)#information extraction
        #
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
        paramsComplex = [par for l in convo_layers for par in l.paramsManualComplex]
        freqGradients = [T.ztensor4('complexGradients') for i in paramsComplex]
        manualUpdates = AdamAlt(freqGradients,paramsComplex)
        #
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
        return train_nn_alt(train_model, manual_train_model, filters, validate_model, test_model,
        n_train_batches, n_valid_batches, n_test_batches, n_epochs, verbose)
    else :
        params = [par for l in param_layers for par in l.params]
        updates = Adam(cost,params)
        train_model = theano.function(
            [index],
            cost,
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
        return train_nn(train_model, validate_model, test_model,
        n_train_batches, n_valid_batches, n_test_batches, n_epochs, verbose)

def network7(nb_labels,spectral,batch_size,n_epochs,verbose):
    """
    architecture 7

    :type nb_labels: int
    :param nb_labels: number of classes and determines dataset

    :type spectral: boolean
    :param spectral: use spectral filter parametrisation or not
    """
    # spectral=False
    # nb_labels=10
    # n_epochs=2
    # batch_size=200
    # verbose=True
    # hyper params
    rng = numpy.random.RandomState(23455)

    if nb_labels == 10:
        datasets = load_data_cifar10()
    else:
        datasets = load_data_cifar100()

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
    image_shape = 32
    image_channels = 3
    layer0_input = x.reshape((batch_size, image_channels, image_shape, image_shape))
    #layers
    layer0 = FConvLayer(
       spectral, 
       rng,
       input=layer0_input,
       image_shape=(batch_size,image_channels,image_shape,image_shape),
       filter_shape=(96+32,image_channels,3,3)
    )
    image_shape = shapeConvo(image_shape,3)

    SP_output = SP(layer0.output,image_shape,0.85)
    image_shape = np.int(np.floor(image_shape*0.85))# reduction from SP

    layer1 = FConvLayer(
       spectral,
       rng,
       input=SP_output,
       image_shape=(batch_size,96+32,image_shape,image_shape),
       filter_shape=(96+32,96+32,1,1),
    )
    image_shape = shapeConvo(image_shape,1)

    layer2 = FConvLayer(
       spectral,
       rng,
       input=layer1.output,
       image_shape=(batch_size,96+32,image_shape,image_shape),
       filter_shape=(100,96+32,1,1),
    )
    image_shape = shapeConvo(image_shape,1) # 25
    ##### FLLAAATTTTT #######
    layer2_output = GA(layer2.output)
    ##### FLLAAATTTTT #######
    layer3 = LogisticRegression(
        input=layer2_output,
        n_in=100,
        n_out=nb_labels)

    param_layers = [layer0,layer1,layer2,layer3]
    convo_layers = [layer0,layer1,layer2]
    last_layer = layer3

    cost = last_layer.negative_log_likelihood(y)

    test_model = theano.function(
        [index],
        last_layer.errors(y),
        givens={
            x: test_set_x[index * batch_size: (index + 1) * batch_size],
            y: test_set_y[index * batch_size: (index + 1) * batch_size]
        }
    )

    validate_model = theano.function(
        [index],
        last_layer.errors(y),
        givens={
            x: valid_set_x[index * batch_size: (index + 1) * batch_size],
            y: valid_set_y[index * batch_size: (index + 1) * batch_size]
        }
    )


    if spectral:
        params = [par for l in param_layers for par in l.params]
        paramsReal = [par for l in convo_layers for par in l.paramsManualReal]
        #
        updates = Adam(cost,params)
        manualGradients = T.grad(cost,paramsReal)#information extraction
        #
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
        paramsComplex = [par for l in convo_layers for par in l.paramsManualComplex]
        freqGradients = [T.ztensor4('complexGradients') for i in paramsComplex]
        manualUpdates = AdamAlt(freqGradients,paramsComplex)
        #
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
        return train_nn_alt(train_model, manual_train_model, filters, validate_model, test_model,
        n_train_batches, n_valid_batches, n_test_batches, n_epochs, verbose)
    else :
        params = [par for l in param_layers for par in l.params]
        updates = Adam(cost,params)
        train_model = theano.function(
            [index],
            cost,
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
        return train_nn(train_model, validate_model, test_model,
        n_train_batches, n_valid_batches, n_test_batches, n_epochs, verbose)

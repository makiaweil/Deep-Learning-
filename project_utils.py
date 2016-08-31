"""
Source Code for Homework 3 of ECBM E6040, Spring 2016, Columbia University

This code contains implementation of several utility funtions for the homework.

Instructor: Prof. Aurel A. Lazar

This code is based on
[1] http://deeplearning.net/tutorial/logreg.html
"""
import os
import sys
import numpy
import scipy.io

import theano
import theano.tensor as T


def shared_dataset(data_xy, borrow=True):
    """ Function that loads the dataset into shared variables

    The reason we store our dataset in shared variables is to allow
    Theano to copy it into the GPU memory (when code is run on GPU).
    Since copying data into the GPU is slow, copying a minibatch everytime
    is needed (the default behaviour if the data is not in a shared
    variable) would lead to a large decrease in performance.
    """
    
    data_x, data_y = data_xy
    shared_x = theano.shared(numpy.asarray(data_x,
                                           dtype=theano.config.floatX), borrow=borrow)
    shared_y = theano.shared(numpy.asarray(data_y,
                                           dtype=theano.config.floatX), borrow=borrow)
    
    # When storing data on the GPU it has to be stored as floats
    # therefore we will store the labels as ``floatX`` as well
    # (``shared_y`` does exactly that). But during our computations
    # we need them as ints (we use labels as index, and if they are
    # floats it doesn't make sense) therefore instead of returning
    # ``shared_y`` we will have to cast it to int. This little hack
    # lets ous get around this issue
    return shared_x, T.cast(shared_y, 'int32')

def load_data_cifar10(ds_rate=None, theano_shared=True):
    ''' Loads the CIFAR10 dataset

    :type ds_rate: float
    :param ds_rate: downsample rate; should be larger than 1, if provided.

    :type theano_shared: boolean
    :param theano_shared: If true, the function returns the dataset as Theano
    shared variables. Otherwise, the function returns raw data.
    '''
    
    if ds_rate is not None:
        assert(ds_rate >= 1.)
    else:
        ds_rate = 1


    dataset_dir = 'cifar-10-batches-py/'
    for i in xrange(5):
        databatch = numpy.load(dataset_dir+'data_batch_'+str(i+1))
        if i == 0:
            train_set_x = numpy.asarray(databatch['data'])[::ds_rate]
            train_set_y = numpy.asarray(databatch['labels'])[::ds_rate]
        else:
            train_set_x = numpy.append(train_set_x, numpy.asarray(databatch['data']), axis=0)[::ds_rate]
            train_set_y = numpy.append(train_set_y, numpy.asarray(databatch['labels']), axis=0)[::ds_rate]
    
    
    databatch = numpy.load(dataset_dir+'test_batch')
    test_set_x = numpy.asarray(databatch['data'])[::ds_rate]
    test_set_y = numpy.asarray(databatch['labels'])[::ds_rate]
    
    
    train_set = [train_set_x, train_set_y]
    test_set = [test_set_x, test_set_y]


    # Downsample the training dataset if specified
    train_set_len = len(train_set[1])
    if ds_rate is not None:
        train_set_len = int(train_set_len // ds_rate)
        train_set = [x[:train_set_len] for x in train_set]

    # Extract validation dataset from train dataset
    valid_set = [x[-(train_set_len//10):] for x in train_set]
    train_set = [x[:-(train_set_len//10)] for x in train_set]

    # train_set, valid_set, test_set format: tuple(input, target)
    # input is a numpy.ndarray of 2 dimensions (a matrix)
    # where each row corresponds to an example. target is a
    # numpy.ndarray of 1 dimension (vector) that has the same length as
    # the number of rows in the input. It should give the target
    # to the example with the same index in the input.

    if theano_shared:
        test_set = shared_dataset(test_set)
        valid_set = shared_dataset(valid_set)
        train_set = shared_dataset(train_set)
    
    
    return [train_set, valid_set, test_set]


def load_data_cifar100(ds_rate=None, theano_shared=True):
    ''' Loads the CIFAR 100 dataset

    :type ds_rate: float
    :param ds_rate: downsample rate; should be larger than 1, if provided.

    :type theano_shared: boolean
    :param theano_shared: If true, the function returns the dataset as Theano
    shared variables. Otherwise, the function returns raw data.
    '''
    
    if ds_rate is not None:
        assert(ds_rate >= 1.)
    else:
        ds_rate = 1


    dataset_dir = 'cifar-100-python/'
    
    databatch = numpy.load(dataset_dir+'train')
    train_set_x = numpy.asarray(databatch['data'])[::ds_rate]
    train_set_y = numpy.asarray(databatch['fine_labels'])[::ds_rate]

    databatch = numpy.load(dataset_dir+'test')
    test_set_x = numpy.asarray(databatch['data'])[::ds_rate]
    test_set_y = numpy.asarray(databatch['fine_labels'])[::ds_rate]
    
    
    train_set = [train_set_x, train_set_y]
    test_set = [test_set_x, test_set_y]

    # Downsample the training dataset if specified
    train_set_len = len(train_set[1])
    if ds_rate is not None:
        train_set_len = int(train_set_len // ds_rate)
        train_set = [x[:train_set_len] for x in train_set]

    # Extract validation dataset from train dataset
    valid_set = [x[-(train_set_len//10):] for x in train_set]
    train_set = [x[:-(train_set_len//10)] for x in train_set]

    # train_set, valid_set, test_set format: tuple(input, target)
    # input is a numpy.ndarray of 2 dimensions (a matrix)
    # where each row corresponds to an example. target is a
    # numpy.ndarray of 1 dimension (vector) that has the same length as
    # the number of rows in the input. It should give the target
    # to the example with the same index in the input.

    if theano_shared:
        test_set = shared_dataset(test_set)
        valid_set = shared_dataset(valid_set)
        train_set = shared_dataset(train_set)
    
    
    return [train_set, valid_set, test_set]


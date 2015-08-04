__author__ = 'rahul'

import numpy as np
import cPickle

class LinClassify:
    x = np.random.rand(0,0)
    weight = np.random.rand(0,0)
    bias = np.random.rand(1,1)

    def __init__(self):
        # Read from cifar datasets - dataset -1
        filename = '/home/rahul/rahul/rahul_experiments/convnets/datasets/cifar-10/cifar-10-batches-py/data_batch_1'
        dictionary = cPickle.load(open(filename, 'rb'))

        # store Xi's and Yi's
        x = dictionary['data']
        y = dictionary['labels']

        K = 10   # Number of classes in cifar-10 , which is 10
        dim = x.shape[1]
        weight = np.random.rand(K,dim)
        bias = np.random.rand(K,1)


    # define all the operating funcs here..
    # 1 . Forward Propagate
    def computeFwd(self,inp, W, b):
        return inp * W + b

    #2. Error function
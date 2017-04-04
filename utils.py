''' 
This module provide numerous tools for image processing, metric computation,
network declaration, and data augmentation.
'''

import os
import numpy as np
import cv2 as cv
from keras import backend as K
from keras.layers.convolutional import Convolution2D
from keras.layers.advanced_activations import ELU
from keras.layers.core import Merge
from keras.engine.topology import Layer
from keras.initializations import get_fans


#%% Image Processing

# Image scaling
'''
It is better to have zero mean and one std data [ref needed]. Here some ways to
accomplish that. The first is a simple one that can by apllied on the fly, 
the second try to equalize as possible all the images, the third require to 
precompute the mean and std value. The question is : can we homogenize the
data in order to cope with different image settings at aquisition (wich 
could misslead the convolutional network) without loosing useful information.
Do we prefer to let the nework handle that ?  
'''

def scale_1(images):
    '''
    Scale the images by removing the mean and dividing by the standard
    deviation. Works by batch meaning that means and standard
    deviations of individual images are averaged over the batch.    
    '''
    images = images.astype(np.float32)
    mean = np.mean(images)
    std = np.std(images)
    return (images - mean) / std


def scale_2(images):
    '''
    Scale the image using Contrast Limited Adaptive Histogram Equalization.
    More information at :
    http://docs.opencv.org/3.1.0/d5/daf/tutorial_py_histogram_equalization.html
    '''
    clahe = cv.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    ada_images = np.empty_like(images)
    for i in range(len(images)):
        ada_images[i] = clahe.apply(images[i])
    ada_images = ada_images.astype(np.float32)
    mean = np.mean(ada_images, axis=(1, 2))[:, np.newaxis, np.newaxis]
    std = np.std(ada_images, axis=(1, 2))[:, np.newaxis, np.newaxis]
    return (ada_images - mean) / std

def scale_3(images):
    '''
    Scale the images according to user specified value. In my case I 
    precomputed the mean and the std on all the avaible images.
    '''
    images = images.astype(np.float32)
    mean = 99.6
    std = 55.3
    return (images - mean) / std

# Elastic deformation
'''
As sugested in "U-Net: Convolutional Networks for Biomedical Image 
Segmentation" here an implementation of elastic deformation
'''

def getElasticMap(gridShape, imageShape, sigma, rng):
    '''
    Return a mapping that describe how to copumpute the desired deformation. 
    In our case we generate smooth deformations using random displacement 
    vectors on a coarse grid which number of cell is determined by the
    gridshape variable. The displacements are sampled from a Gaussian 
    distribution with sigma pixels standard deviation. Reproductibility can
    be acheived by specifying a numpy generator rng. 
    '''
    sigma = np.array(sigma).reshape(1, 1, np.size(sigma)).astype(np.float32)
    shape = (gridShape[0], gridShape[1], 2)
    randomVectors = sigma * rng.randn(*shape).astype(np.float32).clip(-2, 2)
    displacementMap = np.moveaxis(cv.resize(randomVectors, imageShape[::-1],
                                            interpolation=cv.INTER_CUBIC),
                                  [0, 1, 2], [1, 2, 0])
    return displacementMap + np.indices(imageShape).astype(np.float32)


def warpElastic(image, elasticMap):
    ''' 
    Per-pixel displacements are computed from elasticMap using bicubic 
    interpolation
    '''
    return cv.remap(image, *elasticMap[::-1],
                    interpolation=cv.INTER_CUBIC,
                    borderMode=cv.BORDER_REFLECT_101)

#%% Metrics

def crossentropy(y_true, y_pred):
    '''
    Crossentropy declared in Keras format
    '''
    return K.mean(K.binary_crossentropy(y_pred, y_true), axis=(1,2,3))


#%% Keras network declaration utilities

def add_conv(model, *args, **kwargs):
    '''
    Add a convolution layer to a keras sequential model. Parameters can be 
    inputed but if not the following default settings are applied:
        - border_mode : same
        - init : the one set in the model
        - activation : ELU
    '''
    kwargs['border_mode'] = 'same'
    kwargs['init'] = model.weights_rng
    model.add(Convolution2D(*args, **kwargs))
    if 'activation' not in kwargs.keys():
        model.add(ELU())


def add_merge(model, branch):
    '''
    Merge the preceding layer of the model with a user specified branch. Don't
    ask me what all this code is for, a copy pasted it from an similar module 
    and just changed some details.
    '''
    layer = Merge(mode='concat', concat_axis=1)
    output_tensor = layer([model.outputs[0], branch])
    model.outputs = [output_tensor]
    model.inbound_nodes[0].output_tensors = model.outputs
    model.inbound_nodes[0].output_shapes = [model.outputs[0]._keras_shape]
    model.layers.append(layer)
    model.built = False
    model._flattened_layers = None


class SeedDropout(Layer):
    '''
    A seeded version of the one implemented in keras. I just added the seed 
    parameter in the initialization.
    '''
    def __init__(self, p, seed, **kwargs):
        self.p = p
        self.seed = seed
        if 0. < self.p < 1.:
            self.uses_learning_phase = True
        self.supports_masking = True
        super(SeedDropout, self).__init__(**kwargs)

    def call(self, x, mask=None):
        if 0. < self.p < 1.:
            x = K.in_train_phase(K.dropout(x, level=self.p, seed=self.seed), x)
        return x

    def get_config(self):
        config = {'p': self.p}
        base_config = super(SeedDropout, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class HeInit:
    '''
    A seeded version of the keras He initialization. I just added the seed 
    parameter in the initialization.   
    '''
    def __init__(self, seed, dim_ordering='th'):
        self.rng = np.random.RandomState(seed)
        self.dim_ordering = dim_ordering

    def __call__(self, shape, name=None):
        fan_in, fan_out = get_fans(shape, dim_ordering=self.dim_ordering)
        s = np.sqrt(2. / fan_in)
        return K.variable(self.rng.normal(0.0, s, shape),
                          dtype=np.float32, name=name)

#%% Data augmentation

SUBJECT_RANGE = range(1,48)

class Generator:
    '''
    The following keras Generator allows to genrate new randomly sorted and
    augmented data on the fly. As the data was too big for my RAM, i segmented
    it by subject. 
    Parameters : 
        - input_shape : the shape at which data is converted at the end
        - suffle_seed : integer for sorting reproductibility
        - scale : a fonction to scale images (as the ones defined in this 
        module)
        - train_dir : where the data is located 
        - clean_dir : where cleaned data is located. Can be set as train data 
        if we don't want to use cleaned data.
        - grid_shape : elastic deformation parameter
        - sigma : elastic deformation parameter
        - elastic_seed : elastic deformation parameter
        - only_present : Either to load all the data are only the one where the
        nerve is present.
    '''
    def __init__(self, input_shape, shuffle_seed, scale,
                 train_dir, clean_dir, grid_shape, sigma,
                 elastic_seed, only_present=False):

        self.input_shape = input_shape
        self.shuffle_rng = np.random.RandomState(shuffle_seed)

        self.scale = scale

        self.train_dir = train_dir
        self.clean_dir = clean_dir

        self.grid_shape = grid_shape
        self.sigma = sigma
        self.elastic_rng = np.random.RandomState(elastic_seed)

        self.only_present = only_present

    def get_training_set(self):
        '''
        To use during training. Augmentation is applied. Cleaned data can be 
        used.
        '''
        x_train = []
        y_train = []
        
        # Loop on data subsets (by subject)
        for i in SUBJECT_RANGE:
            # Load data
            x_i = np.load(os.path.join(self.train_dir, 'x_%d.npy' % i))
            y_i = np.load(os.path.join(self.clean_dir, 'y_%d.npy' % i))
            
            # Remove some dead pixels
            x_i = x_i[:,1:,1:]
            y_i = y_i[:,1:,1:]

            # Remove images with no nerve if required
            if self.only_present:
                index = np.max(y_i, axis=(1,2)) == 255
                x_i = x_i[index]
                y_i = y_i[index]
            
            # Apply a random elastic deformation to each image
            for j in range(len(x_i)):
                elasticMap = getElasticMap(self.grid_shape, x_i.shape[1:],
                                           self.sigma, self.elastic_rng)
                x_i[j] = warpElastic(x_i[j], elasticMap)
                y_i[j] = warpElastic(y_i[j], elasticMap)
            
            # Resize the image to required input_shape
            x_i = cv.resize(x_i.T, self.input_shape, cv.INTER_AREA).T
            y_i = cv.resize(y_i.T, self.input_shape, cv.INTER_AREA).T
            
            # Apply scaling
            x_i = self.scale(x_i).astype(np.float32)
            ret, y_i = cv.threshold(y_i, 127, 255, cv.THRESH_BINARY)
            y_i //=  255
            
            # Stack results in a list
            x_train += [np.expand_dims(x_i, axis=1)]
            y_train += [np.expand_dims(y_i, axis=1)]
        
        # Group all the genrated data in two numpy arrays
        x_train = np.concatenate(x_train)
        y_train = np.concatenate(y_train)

        # Randomly sort it
        p = self.shuffle_rng.permutation(len(x_train))

        # Et voilà !
        return x_train[p], y_train[p]

    def get_eval_set(self):
        '''
        To use during evaluation. Augmentation is not applied. We don't use
        cleaned data.
        '''
        x_train = []
        y_train = []
        
        # Loop on data subsets (by subject)
        for i in SUBJECT_RANGE:
            x_i = np.load(os.path.join(self.train_dir, 'x_%d.npy' % i))
            y_i = np.load(os.path.join(self.train_dir, 'y_%d.npy' % i))
            
            # Remove some dead pixels
            x_i = x_i[:,1:,1:]
            y_i = y_i[:,1:,1:]

            if self.only_present:
                index = np.max(y_i, axis=(1,2)) == 255
                x_i = x_i[index]
                y_i = y_i[index]
                
            # Apply a random elastic deformation to each image
            x_i = cv.resize(x_i.T, self.input_shape, cv.INTER_AREA).T
            y_i = cv.resize(y_i.T, self.input_shape, cv.INTER_AREA).T

            # Resize the image to required input_shape
            x_i = self.scale(x_i).astype(np.float32)
            ret, y_i = cv.threshold(y_i, 127, 255, cv.THRESH_BINARY)
            y_i //=  255

            # Stack results in a list
            x_train += [np.expand_dims(x_i, axis=1)]
            y_train += [np.expand_dims(y_i, axis=1)]

        # Group all the genrated data in two numpy arrays
        x_train = np.concatenate(x_train)
        y_train = np.concatenate(y_train)

        # Et voilà !
        return x_train, y_train


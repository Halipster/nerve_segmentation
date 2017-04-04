'''
This module contains a fonction intended to generate a U-Net convolutional 
network. Its architecture is adapted from the following article :
"U-Net: Convolutional Networks for Biomedical Image Segmentation" 
which can be found at : https://arxiv.org/abs/1505.04597
Major changes are :
    - less feature channels because we are dealing with black and white and 
    not color pictures (but mainly because i could'nt afford more computation)
    - use of padding in order to maintain the same output size
    - use of ELU activation
The user can specify the input shape of images and provide some seeding 
parameters for reproductibility.
'''

import numpy as np
from keras.models import Sequential
from keras.layers.convolutional import MaxPooling2D, UpSampling2D
from utils import SeedDropout, add_conv, add_merge, HeInit


def get_model(input_shape, dropout_seed, weight_seed):
    '''
    Return a U-Net according to the following parameters :
        - input_shape : the shape of the inputed images 
        - droupout_seed : a integer for seeding the droupouts generation
        - weight_seed : an integer for seeding the initial weights generation
    '''
    
    # initialization of the model and seeds
    dropout_rng = np.random.RandomState(dropout_seed)
    model = Sequential()
    model.weights_rng = HeInit(weight_seed)

    # U-Net architecture declaration    

    add_conv(model, 16, 3, 3, input_shape=(1, *input_shape))
    add_conv(model, 16, 3, 3)
    branch1 = model.output
    model.add(MaxPooling2D(pool_size=(2, 2)))

    add_conv(model, 32, 3, 3)
    add_conv(model, 32, 3, 3)
    branch2 = model.output
    model.add(MaxPooling2D(pool_size=(2, 2)))

    add_conv(model, 64, 3, 3)
    add_conv(model, 64, 3, 3)
    branch3 = model.output
    model.add(MaxPooling2D(pool_size=(2, 2)))

    add_conv(model, 128, 3, 3)
    add_conv(model, 128, 3, 3)
    branch4 = model.output
    model.add(MaxPooling2D(pool_size=(2, 2)))

    add_conv(model, 256, 3, 3)
    model.add(SeedDropout(0.5, dropout_rng.randint(-32768, 32767)))
    add_conv(model, 256, 3, 3)
    model.add(SeedDropout(0.5, dropout_rng.randint(-32768, 32767)))
    add_conv(model, 256, 3, 3)
    model.add(UpSampling2D(size=(2, 2)))
    
    add_merge(model, branch4)
    add_conv(model, 128, 3, 3)
    add_conv(model, 128, 3, 3)
    model.add(UpSampling2D(size=(2, 2)))

    add_merge(model, branch3)
    add_conv(model, 64, 3, 3)
    add_conv(model, 64, 3, 3)
    model.add(UpSampling2D(size=(2, 2)))

    add_merge(model, branch2)
    add_conv(model, 32, 3, 3)
    add_conv(model, 32, 3, 3)
    model.add(UpSampling2D(size=(2, 2)))

    add_merge(model, branch1)
    add_conv(model, 16, 3, 3)
    add_conv(model, 16, 3, 3)

    add_conv(model, 1, 1, 1, activation='sigmoid')

    return model

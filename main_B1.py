'''
This script train a U-NET on the data where nerves are present using scale_1 
and no cleaned data. 
'''

import numpy as np
from keras.optimizers import Adam
from utils import crossentropy, scale_1, Generator
from unet import get_model

input_shape = (128, 160)
batch_size = 4
nb_epoch = 20
seed = 95136
train_dir = '/media/hal/7ECE0648CE05F8E3/train'

model = get_model(input_shape=input_shape,
                  dropout_seed = seed,
                  weight_seed = seed)

#%%

model.compile(optimizer=Adam(),
              loss=crossentropy)

generator = Generator(input_shape=input_shape,
                      shuffle_seed=seed,
                      scale=scale_1,
                      train_dir=train_dir,
                      clean_dir=train_dir,
                      grid_shape=(4,4),
                      sigma=10,
                      elastic_seed=seed,
                      only_present=True)

loss = np.Inf

for epoch in range(nb_epoch):
    print('### EPOCH %d/%d ###' % (epoch + 1, nb_epoch))
    x_train, y_train = generator.get_training_set()
    hist = model.fit(x_train, y_train,
                    batch_size=batch_size, nb_epoch=1,
                    shuffle=False)
    if hist.history['loss'][0] < loss:
        model.save_weights('weights_B1.hdf5', overwrite=True)
        loss = hist.history['loss'][0]


#%%

import cv2 as cv

model.load_weights('weights_B1.hdf5')
x = np.load('x_test.npz')['x']
x = x[:,1:,1:]

N = len(x)
x_test = np.empty((N,*input_shape))

for i in range(N):
    x_test[i] = cv.resize(x[i].T, input_shape, cv.INTER_AREA).T

x_test = scale_1(x_test).astype(np.float32)
x_test = np.expand_dims(x_test, axis=1)

y_pred = model.predict(x_test, verbose=1)
np.save('y_pred_test_B1', y_pred)

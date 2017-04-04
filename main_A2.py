'''
This script train a U-NET on all the data using scale_2 and no cleaned data. 
'''

import numpy as np
from keras.optimizers import Adam
from utils import crossentropy, scale_2, Generator
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
                      scale=scale_2,
                      train_dir=train_dir,
                      clean_dir=train_dir,
                      grid_shape=(4,4),
                      sigma=10,
                      elastic_seed=seed)

loss = np.Inf

for epoch in range(nb_epoch):
    print('### EPOCH %d/%d ###' % (epoch + 1, nb_epoch))
    x_train, y_train = generator.get_training_set()
    hist = model.fit(x_train, y_train,
                    batch_size=batch_size, nb_epoch=1,
                    shuffle=False)
    if hist.history['loss'][0] < loss:
        model.save_weights('weights_A2.hdf5', overwrite=True)
        loss = hist.history['loss'][0]

model.load_weights('weights_A2.hdf5')
x_train, y_train = generator.get_eval_set()
y_pred = model.predict(x_train, verbose=1)
np.save('y_pred_A2', y_pred)

#%%

import cv2 as cv

model.load_weights('weights_A2.hdf5')
x = np.load('x_test.npz')['x']
x = x[:,1:,1:]

N = len(x)
x_test = np.empty((N,*input_shape))


clahe = cv.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
for i in range(N):
    x_test[i] = cv.resize(x[i].T, input_shape, cv.INTER_AREA).T
    x_test[i] = clahe.apply(x_test[i].astype(np.uint8))
    mean = np.mean(x_test[i])
    std = np.std(x_test[i])
    x_test[i] =  (x_test[i] - mean) / std

x_test = np.expand_dims(x_test, axis=1).astype(np.float32)

y_pred = model.predict(x_test, verbose=1)
np.save('y_pred_test_A2', y_pred)

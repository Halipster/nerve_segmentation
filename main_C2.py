'''
This script train a convolutional network on the data outputed from A2 in order
to decide if a nerve is present or not.
'''

import numpy as np
from utils import SeedDropout, HeInit
from keras.models import Sequential
from keras.layers.core import Dense, Flatten
from keras.layers.convolutional import MaxPooling2D
from keras.optimizers import Adam

seed = 16545
rng = np.random.RandomState(seed)
weights_rng = HeInit(seed)

#%% Load Data

def scale(x):
    return x - 0.025

x_train = scale(np.load('y_pred_A2.npy'))
y_train = np.max(np.load('y_true_A.npy'), axis=(1,2,3))

img_rows, img_cols = x_train.shape[2:]


#%% Define Model

model = Sequential()
model.add(MaxPooling2D(pool_size=(8, 8),input_shape=(1, img_rows, img_cols)))
model.add(Flatten())
model.add(Dense(256, activation='relu',W_regularizer='l2',b_regularizer='l2',init=weights_rng))
model.add(SeedDropout(0.5, rng.randint(-32768, 32767)))
model.add(Dense(1, activation='sigmoid',W_regularizer='l2',b_regularizer='l2',init=weights_rng))

#%%

model.compile(optimizer=Adam(), loss='binary_crossentropy', metrics=['acc'])

loss = np.Inf
nb_epoch = 20

for epoch in range(nb_epoch):
    print('### EPOCH %d/%d ###' % (epoch + 1, nb_epoch))
    p = rng.permutation(len(x_train))
    hist = model.fit(x_train[p], y_train[p],
                     batch_size=128, nb_epoch=1,
                     shuffle=False)
    if hist.history['loss'][0] < loss:
        model.save_weights('weights_C2.hdf5', overwrite=True)
        loss = hist.history['loss'][0]

model.load_weights('weights_C2.hdf5')
y_pred = model.predict(x_train, verbose=1)
np.save('y_pred_C2', y_pred)
np.save('y_true_C2', y_train)


#%%

model.load_weights('weights_C2.hdf5')

def scale(x):
    return x - 0.025

x_test = scale(np.load('y_pred_test_A2.npy'))

img_rows, img_cols = x_test.shape[2:]
is_present = model.predict(x_test, verbose=1)
np.save('y_pred_test_C2', is_present)


'''
This script synthesize the outputs of Bs and Cs to a unique submission
'''
import numpy as np
import cv2 as cv

nb_rows, nb_cols = 420, 580

present1 = np.load('y_pred_test_C1.npy') 
present2 = np.load('y_pred_test_C2.npy')
present3 = np.load('y_pred_test_C3.npy')

present = (present1 + present2 + present3) / 3
present = present > 0.35

y_pred1 = np.load('y_pred_test_B1.npy')[:, 0, :, :]
y_pred2 = np.load('y_pred_test_B2.npy')[:, 0, :, :]
y_pred3 = np.load('y_pred_test_B3.npy')[:, 0, :, :]
y_pred = (y_pred1 + y_pred2 + y_pred3) / 3
y_pred = np.round(y_pred * 255).astype(np.uint8)
N = len(y_pred)
y_class = np.empty((N, nb_rows, nb_cols), dtype=np.uint8)

for i in range(N):
    y_class[i] = (cv.resize(y_pred[i], (nb_cols, nb_rows)) > 128).astype(np.uint8)

def run_length_enc(label):
    from itertools import chain
    x = label.transpose().flatten()
    y = np.where(x > 0)[0]
    if len(y) == 0:  # consider as empty
        return ''
    z = np.where(np.diff(y) > 1)[0]
    start = np.insert(y[z+1], 0, y[0])
    end = np.append(y[z], y[-1])
    length = end - start
    res = [[s+1, l+1] for s, l in zip(list(start), list(length))]
    res = list(chain.from_iterable(res))
    return ' '.join([str(r) for r in res])

ids = []
rles = []
for i in range(N):
    if present[i]:
        rle = run_length_enc(y_class[i])
    else:
        rle = ''
    rles.append(rle)
    ids.append(str(i + 1))
    if i % 100 == 0:
        print('{}/{}'.format(i, N))
first_row = 'img,pixels'
file_name = 'submission.csv'
with open(file_name, 'w+') as f:
    f.write(first_row + '\n')
    for i in range(N):
        s = str(ids[i]) + ',' + rles[i]
        f.write(s + '\n')


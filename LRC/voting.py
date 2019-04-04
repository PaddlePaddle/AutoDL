import numpy as np
import cPickle as cp
import sys, os

#model_path = 'final_paddle-results'
model_path = 'paddle-results'
fl = os.listdir(model_path)
labels = np.load('labels.npz')['arr_0']
pred = np.zeros((10000, 10))
fl.sort()
i = 0
weight=1
for f in fl:
    print(f)
    if i == 1: weight=1.2
    if i == 2: weight=0.8
    if i == 3: weight=1.3
    if i == 4: weight=1.1
    if i == 5: weight=0.9
    pred += weight* cp.load(open(os.path.join(model_path, f)))
    print(np.mean(np.argmax(pred, axis=1) == labels))
    i += 1

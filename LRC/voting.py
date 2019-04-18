import numpy as np
try:
    import cPickle as pickle
except ImportError:
    import pickle
import sys, os

model_path = 'paddle_predict'
fl = os.listdir(model_path)
labels = np.load('labels.npz')['arr_0']
pred = np.zeros((10000, 10))
fl.sort()
i = 0
for f in fl:
    print(f)
    pred += pickle.load(open(os.path.join(model_path, f)))
    print(np.mean(np.argmax(pred, axis=1) == labels))
    i += 1

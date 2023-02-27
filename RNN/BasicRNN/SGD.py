import sys
sys.path.append("..")

import numpy as np 

class SGD:

    def __init__(self, lr=0.01):
        self.lr = lr
        self.THRESHOLD = 0.9

    def update(self, params, grads):
        for i in range(len(params)):

            if np.linalg.norm(grads[i]) > self.THRESHOLD:
                grads[i]= self.THRESHOLD * (grads[i] / np.linalg.norm(grads[i]))
            params[i] -= self.lr*grads[i]
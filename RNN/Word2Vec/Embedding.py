import sys
sys.path.append("..")

import numpy as np

class Embedding:
    def __init__(self, W):
        self.params = [W]
        self.grads = [np.zeros_like(W)]
        self.idx = None
        
    def forward(self, idx):
        W, = self.params
        self.idx = idx
        out = W[idx]
        return out
    
    def backward(self, dout):
        dW, = self.grads
        dW[...] = 0

        for i ,word_id in enumerate(self.idx):
            # 중복되는 index는 add로 해결 
            # (가중치의 크기만 변화, 가중치 방향은 변화 X 이니까)
            dW[word_id] += dout[i]
        
        return None
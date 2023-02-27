import sys
import numpy as np

sys.path.append(".")
from layers import MatMul
import os
print(os.getcwd())
import skip_gram_loss_layer

class SimpleSkipGram:
    def __init__(self, vocab_size, hidden_size):
        V, H = vocab_size, hidden_size
        
        # 가중치 초기화
        W_in = 0.01 * np.random.randn(V, H).astype('f')
        W_out = 0.01 * np.random.randn(H, V).astype('f')
        
        # 레이어 생성
        self.in_layer = MatMul(W_in)
        self.out_layer = MatMul(W_out)
        self.loss_layer = skip_gram_loss_layer.skip_gram_loss_layer()
        
        # 모든 가중치와 기울기를 리스트에 모은다.
        layers = [self.in_layer, self.out_layer]
        self.params, self.grads = [], []
        for layer in layers:
            self.params += layer.params
            self.grads += layer.grads
        
        # 인스턴스 변수에 단어의 분산표현을 저장한다.
        self.word_vecs1 = W_in
        self.word_vecs2 = W_out.T
        
    def forward(self, contexts, target):
        h = self.in_layer.forward(target)# 1 X D
        s = self.out_layer.forward(h)# 1 X V
        loss = self.loss_layer.forward(s, contexts) # 배열 하나에 윈도우 다 넣어주기
        
        return loss
    
    def backward(self, dout=1):
        dl = self.loss_layer.backward(dout)
        dh = self.out_layer.backward(dl)
        self.in_layer.backward(dh)
        return None
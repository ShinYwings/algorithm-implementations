import sys
sys.path.append('..')  # 부모 디렉터리의 파일을 가져올 수 있도록 설정
from Word2Vec import Embedding
from Word2Vec import negativeSampling
import numpy as np


class SkipGram:
    def __init__(self, vocab_size, hidden_size, window_size, corpus):
        V, H = vocab_size, hidden_size
        rn = np.random.randn

        # 가중치 초기화
        W_in = 0.01 * rn(V, H).astype('f')
        W_out = 0.01 * rn(V, H).astype('f')

        # 계층 생성
        self.in_layer = Embedding(W_in)
        self.loss_layers = []
        for i in range(2 * window_size):
            layer = negativeSampling(W_out, corpus, power=0.75, sample_size=5)
            self.loss_layers.append(layer)

        # 모든 가중치와 기울기를 리스트에 모은다.
        layers = [self.in_layer] + self.loss_layers
        self.params, self.grads = [], []
        for layer in layers:
            self.params += layer.params
            self.grads += layer.grads

        # 인스턴스 변수에 단어의 분산 표현을 저장한다.
        self.word_vecs = W_in # shallow copy (memory 주소 공유)

    def forward(self, contexts, target):
        h = self.in_layer.forward(target)
        loss = 0
        for i, layer in enumerate(self.loss_layers): 
            # 한 layer마다 loss가 negative sampling 개수 (k) 만큼 있는데 sum 으로 나옴   <= 이거를 contexts 개수 만큼 iteration 한게 전체 loss (그니까 )
            loss += layer.forward(h, contexts[:, i]) # [batch_size, contexts의 개수]  [batch_size = iteration 횟수, context의 개수 = 2 * window_size]
        return loss

    def backward(self, dout=1):
        dh = 0
        for i, layer in enumerate(self.loss_layers):
            dh += layer.backward(dout)
        self.in_layer.backward(dh)
        return None
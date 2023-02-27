import numpy as np 
import collections
from embeddingDot import embeddingDot
from unigramSampler import UnigramSampler
from sigmoidWithLoss import sigmoidWithLoss

class negativeSampling:

    def __init__(self, W, corpus, power = 0.75, sample_size =5):   
        self.sample_size = sample_size    # k negative samples (5~20이 적당)
        self.sampler = UnigramSampler(corpus, power, sample_size)
        self.loss_layers = [SigmoidWithLoss(W) for _ in range(sample_size+1)]
        self.embed_dot_layers = [EmbeddingDot(W) for _ in range(sample_size+1)]

        self.params, self.grads = [], []
        for layer in self.embed_dot_layers:
            self.params += layer.params
            self.grads += layer.grads

    def forward(self, h, target):
        batch_size = target.shape[0]
        negative_sample = self.sampler.get_negative_sample(target)
        
        # positive
        score = self.embed_dot_layers[0].forward(h,target)
        correct_label = np.ones(batch_size, dtype=np.int32)
        loss = self.loss_layers[0].forward(score, correct_label)
        
        # negative
        negative_label = np.zeros(batch_size, dtype=np.int32)
        for i in range(self.sample_size):
            negative_target = negative_sample[:,i]
            score = self.embed_dot_layers[1+i].forward(h, negative_target)
            loss += self.loss_layers[1+i].forward(score, negative_label)

        return loss

    def backward(self, dout=1):
        dh =0
        for l0, l1 in zip(self.loss_layers, self.embed_dot_layers):
            dscore = l0.backward(dout) # y-t
            dh += l1.backward(dscore) # 각 
        return dh
import sys

sys.path.append(".")
import os
# sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))))

import matplotlib.pyplot as plt
import numpy as np

from common.optimizer import Adam
from common.utils import convert_one_hot
from Word2Vec.simple_skip_gram import SimpleSkipGram
# from Word2Vec.originalskipgram import SimpleSkipGram
from dataset import ptb

def create_contexts_target(corpus, window_size=1):

    target = corpus[window_size:-window_size]
    contexts = []

    for idx in range(window_size, len(corpus)-window_size):
        cs = []
        for t in range(-window_size, window_size + 1):
            if t == 0:
                continue
            cs.append(corpus[idx + t])
        contexts.append(cs)

    return np.array(contexts), np.array(target)

lr = 0.02

# 학습 데이터 읽기(전체 중 1000개만)
# corpus = [0, 1, 2, 2, 3]
# word_to_id = {'a':0,'p':1,'l':2,'e':3, '<unk>':4, '<eos>':5}
# id_to_word = {0:'a',1:'p',2:'l',3:'e',4:'<unk>', 5:'<eos>'}
# corpus_size = 6
corpus, word_to_id, id_to_word = ptb.load_data('train')

hidden_size = 80 # RNN의 은닉 상태 벡터의 원소 수
corpus_size = 1000

corpus = corpus[:corpus_size]
vocab_size = int(max(corpus) + 1)

print('말뭉치 크기: %d, 어휘 수: %d' % (corpus_size, vocab_size))

one_hot = convert_one_hot(corpus, vocab_size)

# 모델 생성
model = SimpleSkipGram(vocab_size, hidden_size)
optimizer = Adam(lr)

contexts, target = create_contexts_target(one_hot, window_size=1)

loss =0
# 기울기를 구하여 매개변수 갱신
for i in range(15):
    loss = model.forward(contexts, target)
    model.backward()
    optimizer.update(model.params, model.grads)
    print("iter:",i, "loss",loss)

print(loss)
print(model.word_vecs1)
import numpy as np
import collections

class UnigramSampler:
    def __init__(self, corpus, power, sample_size):
        self.sample_size = sample_size
        self.vocab_size = None
        self.word_p = None
        
        counts = collections.Counter()
        for word_id in corpus:
            counts[word_id] += 1 # 빈도수
            
        vocab_size = len(counts)
        self.vocab_size = vocab_size
        
        self.word_p = np.zeros(vocab_size)
        for i in range(vocab_size):
            self.word_p[i] = counts[i]
            
        self.word_p = np.power(self.word_p, power)
        self.word_p /= np.sum(self.word_p)
        
    def get_negative_sample(self, target):
        batch_size = target.shape[0]
        
        negative_sample = np.zeros((batch_size, self.sample_size), dtype=np.int32)
        
        for i in range(batch_size):
            p = self.word_p.copy()
            target_idx = target[i]
            p[target_idx] = 0  # target이 뽑히지 않게 하기 위함
            p /= p.sum()  # 다시 정규화 해줌
            negative_sample[i, :] = np.random.choice(self.vocab_size,
                                                        size=self.sample_size,
                                                        replace=False, p=p)
            
            
        return negative_sample
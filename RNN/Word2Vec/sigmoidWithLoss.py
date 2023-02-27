import numpy as np

def cross_entropy_error(y, t):
    if y.ndim == 1:
        t = t.reshape(1, t.size)
        y = y.reshape(1, y.size)

    # 정답 데이터가 원핫 벡터일 경우 정답 레이블 인덱스로 변환
    if t.size == y.size:
        t = t.argmax(axis=1)
    
    batch_size = y.shape[0]

    # t가 0일때 1-y 선택 그리고 t가 1일때 y 선택
    cross_entropy = np.log(y[np.arange(batch_size), t] + 1e-7)
    loss = -np.sum(cross_entropy) / batch_size
    
    return loss

class sigmoidWithLoss:

    def __init__(self, W):

        self.params = [W]
        self.grads = []
        self.loss = None
        self.pred_y = None
        self.y= None 

    def forward(self, score, label):

        self.y = label
        self.pred_y = 1 / (1+np.exp(-score))
        
        self.loss = cross_entropy_error(np.c_[1-self.pred_y, self.pred_y], self.y)

        return self.loss

    def backward(self, dout=1):
        batch_size = self.y.shape[0]

        dx = (self.pred_y - self.y) * dout / batch_size

        return dx

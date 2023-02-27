import numpy as np

class skip_gram_loss_layer:

    def __init__(self):
        self.y = None  
        self.t = None

    def forward(self, x, t):
        self.t = t
        x_batch_size = x.shape[0]
        
        self.y = softmax(x)
        loss = list()
        for i in range(t.shape[1]): #윈도우 사이즈
            loss.append(cross_entropy_error(self.y, self.t[:,i]))
        loss = np.sum(loss)
        return loss

    def backward(self, dout=1):
        batch_size = self.t.shape[0]

        dx = self.y.copy()

        test = np.sum(self.t,axis=1)
        test2 = np.where(test==1)

        dx[test2[0], test2[1]] -= 1 

        dx *= dout
        dx = dx / batch_size

        return dx

def softmax(x):
    if x.ndim == 2:
        x = x - x.max(axis=1, keepdims=True)
        x = np.exp(x)
        x /= x.sum(axis=1, keepdims=True)
    elif x.ndim == 1:
        x = x - np.max(x)
        x = np.exp(x) / np.sum(np.exp(x))

    return x

def cross_entropy_error(y, t):
    
    if y.ndim == 1:
        t = t.reshape(1, t.size)
        y = y.reshape(1, y.size)
        
    if t.size == y.size:
        t = t.argmax(axis=1)
    batch_size = y.shape[0]

    return -np.sum(np.log(y[np.arange(batch_size), t] + 1e-7)) / batch_size

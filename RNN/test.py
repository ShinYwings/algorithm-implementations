import numpy as np
import tensorflow as tf 

def split_heads(inputs, batch_size):
    
    inputs = tf.reshape(inputs, shape=(batch_size, -1, 5,2 ))
    
    return inputs, tf.transpose(inputs, perm=[0, 2, 1, 3])

def sigmoid(x):
    return 1 / (1+np.exp(-x))

# case 1
precision = [[37.0, 37.1], [36.8, 36.7]]
recall = [73.0, 73.0, 72.9, 72.8, 73.0]
f1 = [49.2, 49.2, 48.9, 48.8, 49.1]
acc = [73.0, 73.0, 72.9, 72.8, 73.0]
auc = [0.803, 0.801, 0.802, 0.800, 0.804]

# case 2
precision2 = [23.0, 23.2, 22.9, 23.0, 23.2]
recall2 = [74.8, 74.9, 74.7, 74.8, 75.0]
f12 = [35.2, 35.5, 35.1, 35.2, 35.4]
acc2 = [74.8, 75.0, 74.7, 74.8, 74.9]
auc2 = [0.826, 0.826, 0.826, 0.827, 0.829]

x = [73.1, 72.9, 72.9, 72.6, 72.7]
y = [74.5, 74.1, 74.3, 74.2, 74.2]
z = [0.802, 0.801, 0.804, 0.797, 0.8]
i = [0.819, 0.819, 0.82, 0.814, 0.817]

whole = [x,y,z,i]

ans , ans2 = split_heads(whole, 2)

print(ans)

print(ans2)
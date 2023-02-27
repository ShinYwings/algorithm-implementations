import tensorflow as tf 

class BahdanauAttention(tf.keras.layers.Layer):
    def __init__(self, units):
        super( BahdanauAttention, self).__init__()
        self.W1= tf.keras.layers.Dense(units)  # encoder output
        self.W2= tf.keras.layers.Dense(units)  # Decoder hidden
        self.V= tf.keras.layers.Dense(1)
    
    def call(self, query, values):
        # query hidden state shape == (batch_size, hidden size)
        # query_with_time_axis shape == (batch_size, 1, hidden size)
        # values shape == (batch_size, max_len, hidden size)
        # we are doing this to broadcast addition along the time axis to calculate the score
        query_with_time_axis = tf.expand_dims(query, 1)

        #calculate the Attention score
        
        score= self.V(tf.nn.tanh(self.W1(query_with_time_axis) + self.W2(values)))
        
        # attention_weights shape == (batch_size, max_length, 1)
        attention_weights= tf.nn.softmax(score, axis=1)
        
        #context_vector
        context_vector= attention_weights * values
        
        #Computes the sum of elements across dimensions of a tensor
        context_vector = tf.reduce_sum(context_vector, axis=1)
        return context_vector, attention_weights
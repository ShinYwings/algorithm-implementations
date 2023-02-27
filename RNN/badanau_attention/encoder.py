import tensorflow as tf
import tensorflow.keras.backend as K

class Encoder(tf.keras.Model):
    def __init__(self, vocab_size, embedding_dim, encoder_units, batch_size):
        super(Encoder, self).__init__()
        self.batch_size= batch_size
        self.encoder_units=encoder_units
        self.embedding=tf.keras.layers.Embedding(vocab_size, embedding_dim)
        self.gru_forward= tf.keras.layers.GRU(encoder_units,
                                      return_sequences=True,
                                      return_state=True, recurrent_initializer='glorot_uniform')
        self.gru_backward= tf.keras.layers.GRU(encoder_units, 
                                        go_backwards=True,
                                      return_sequences=True,
                                      return_state=True, recurrent_initializer='glorot_uniform')

    def call(self, x, hidden):
        #pass the input x to the embedding layer
        x= self.embedding(x)
        # pass the embedding and the hidden state to GRU
        output_forward, state_forward = self.gru_forward(x, hidden)
        output_backward, state_backward = self.gru_backward(x, hidden)
        
        output = K.concatenate([output_forward, output_backward])
        
        return output, state_backward
    
    def initialize_hidden_state(self):
        return tf.zeros((self.batch_size, self.encoder_units))
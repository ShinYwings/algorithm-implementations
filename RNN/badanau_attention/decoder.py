import tensorflow as tf
from bahdanauAttention import BahdanauAttention

class Decoder(tf.keras.Model):
    def __init__(self, vocab_size, embedding_dim, decoder_units, batch_sz):
        super (Decoder,self).__init__()
        self.batch_sz= batch_sz
        self.decoder_units = decoder_units
        self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)
        self.gru= tf.keras.layers.GRU(decoder_units, 
                                      return_sequences= True,
                                      return_state=True,
                          recurrent_initializer='glorot_uniform')
        # Fully connected layer
        self.fc= tf.keras.layers.Dense(vocab_size)

        self.W= tf.keras.layers.Dense(decoder_units) # hidden init
        
        # attention
        self.attention = BahdanauAttention(self.decoder_units)
    
    def call(self, x, hidden, encoder_output):
        
        hidden = tf.nn.tanh(self.W(hidden))
        
        context_vector, attention_weights = self.attention(hidden, encoder_output)
        
        # pass output sequnece thru the input layers
        x= self.embedding(x)
        
        # concatenate context vector and embedding for output sequence
        x= tf.concat([tf.expand_dims(context_vector, 1), x], axis=-1)
        
        # passing the concatenated vector to the GRU
        
        output, state = self.gru(x, hidden)
        
        # output shape == (batch_size * 1, hidden_size)
        output= tf.reshape(output, (-1, output.shape[2]))
        
        # pass the output thru Fc layers
        x= self.fc(output)
        
        return x, state, attention_weights
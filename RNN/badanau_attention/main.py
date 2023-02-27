import numpy as np
import pandas as pd 
import tensorflow as tf 
from string import digits
import matplotlib.pyplot as plt
import tensorflow as tf
import matplotlib.ticker as ticker
from sklearn.model_selection import train_test_split
import re
import os
import io
import time
from encoder import Encoder
from decoder import Decoder
from bahdanauAttention import BahdanauAttention

def preprocess_sentence(sentence):
    
    num_digits= str.maketrans('','', digits)
    
    sentence= sentence.lower()
    sentence= re.sub(" +", " ", sentence)
    sentence= re.sub("'", '', sentence)
    sentence= sentence.translate(num_digits)
    sentence= re.sub(r"([?.!,¿])", r" \1 ", sentence)
    sentence = sentence.rstrip().strip()
    sentence=  'start_ ' + sentence + ' _end'
    
    return sentence

def create_dataset(path, num_examples):

    lines = io.open(path, encoding='UTF-8').read().strip().split('\n')
    
    word_pairs = [[preprocess_sentence(w) for w in l.split('\t')]  for l in lines[:num_examples]]
    
    return zip(*word_pairs)

def evaluate(sentence):
    attention_plot= np.zeros((max_target_length, max_source_length))
    #preprocess the sentnece
    sentence = preprocess_sentence(sentence)
    
    #convert the sentence to index based on word2index dictionary
    inputs= [source_sentence_tokenizer.word_index[i] for i in sentence.split(' ')]
    
    # pad the sequence 
    inputs = tf.keras.preprocessing.sequence.pad_sequences([inputs], maxlen=max_source_length, padding='post')
    
    #conver to tensors
    inputs = tf.convert_to_tensor(inputs)
    
    result= ''
    
    # creating encoder
    hidden = [tf.zeros((1, units))]
    encoder_output, encoder_hidden= encoder(inputs, hidden)
    
    # creating decoder
    decoder_hidden = encoder_hidden
    decoder_input = tf.expand_dims([target_sentence_tokenizer.word_index['start_']], 0)
    
    for t in range(max_target_length):
        predictions, decoder_hidden, attention_weights= decoder(decoder_input, decoder_hidden, encoder_output)
        
        # storing attention weight for plotting it
        attention_weights = tf.reshape(attention_weights, (-1,))
        attention_plot[t] = attention_weights.numpy()
        
        prediction_id= tf.argmax(predictions[0]).numpy()
        result += target_sentence_tokenizer.index_word[prediction_id] + ' '
        
        if target_sentence_tokenizer.index_word[prediction_id] == '_end':
            return result,sentence, attention_plot
        
        # predicted id is fed back to as input to the decoder
        decoder_input = tf.expand_dims([prediction_id], 0)
            
    return result,sentence, attention_plot

def plot_attention(attention, sentence, predicted_sentence):
    fig = plt.figure(figsize=(10,10))
    ax= fig.add_subplot(1,1,1)
    ax.matshow(attention, cmap='Greens')
    fontdict={'fontsize':10}
    
    ax.set_xticklabels([''] + sentence, fontdict=fontdict, rotation=90)
    ax.set_yticklabels([''] + predicted_sentence, fontdict=fontdict)
    ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(1))

    plt.show()

def translate(sentence):
    result, sentence, attention_plot = evaluate(sentence)
    
    print('Input : %s' % (sentence))
    print('predicted sentence :{}'.format(result))
    
    attention_plot= attention_plot[:len(result.split(' ')), :len(sentence.split(' '))]
    plot_attention(attention_plot, sentence.split(' '), result.split(' '))

with tf.device("GPU:1"):
    @tf.function
    def train_step(inp, targ, enc_hidden):
        
        loss = 0
        with tf.GradientTape() as tape:
            #create encoder
            enc_output, enc_hidden = encoder(inp, enc_hidden)
            dec_hidden = enc_hidden
            #first input to decode is start_
            dec_input = tf.expand_dims(
                [target_sentence_tokenizer.word_index['start_']] * BATCH_SIZE, 1)
            # Teacher forcing - feeding the target as the next input
            for t in range(1, targ.shape[1]):
                # passing enc_output to the decoder
                predictions, dec_hidden, _ = decoder(dec_input, dec_hidden, enc_output)
                # calculate loss based on predictions  
                loss += loss_function(targ[:, t], predictions)
                # using teacher forcing
                dec_input = tf.expand_dims(targ[:, t], 1)

        batch_loss = (loss / int(targ.shape[1]))
        variables = encoder.trainable_variables + decoder.trainable_variables
        gradients = tape.gradient(loss, variables)
        optimizer.apply_gradients(zip(gradients, variables))

        return batch_loss

if __name__ == "__main__":
    
    data_path = r"D:\NLP\kor-eng\kor.txt"

    #Read the data
    # lines_raw= pd.read_table(data_path,names=['source', 'target', 'comments'])
    # lines_raw.sample(5)

    sample_size=10000
    source, target, _ = create_dataset(data_path, sample_size)
    
    # create a tokenizer for source sentence
    source_sentence_tokenizer= tf.keras.preprocessing.text.Tokenizer(filters='')
    # Fit the source sentences to the source tokenizer
    source_sentence_tokenizer.fit_on_texts(source)
    #Transforms each text in texts to a sequence of integers.
    source_tensor = source_sentence_tokenizer.texts_to_sequences(source)
    source_tensor= tf.keras.preprocessing.sequence.pad_sequences(source_tensor,padding='post')
    
    # create the target sentence tokenizer
    target_sentence_tokenizer= tf.keras.preprocessing.text.Tokenizer(filters='')
    # Fit the source sentences to the source tokenizer
    target_sentence_tokenizer.fit_on_texts(target)
    #Transforms each text in texts to a sequence of integers.
    target_tensor = target_sentence_tokenizer.texts_to_sequences(target)
    target_tensor= tf.keras.preprocessing.sequence.pad_sequences(target_tensor,padding='post')

    source_train_tensor, source_test_tensor, target_train_tensor, target_test_tensor= train_test_split(source_tensor, target_tensor,test_size=0.2)

    BUFFER_SIZE = len(source_train_tensor)
    #setting the BATCH SIZE
    BATCH_SIZE = 8
    steps_per_epoch= len(source_train_tensor)//BATCH_SIZE
    embedding_dim=128
    units=512
    source_vocab_size= len(source_sentence_tokenizer.word_index)+1
    target_vocab_size= len(target_sentence_tokenizer.word_index)+1
    #Create data in memeory 
    dataset=tf.data.Dataset.from_tensor_slices((source_train_tensor, target_train_tensor)).shuffle(BATCH_SIZE)
    # shuffles the data in the batch
    dataset = dataset.batch(BATCH_SIZE, drop_remainder=True)
    #Creates an Iterator for enumerating the elements of this dataset.
    #Extract the next element from the dataset
    source_batch, target_batch = next(iter(dataset))
    print(source_batch.shape)

    encoder = Encoder(source_vocab_size, embedding_dim, units, BATCH_SIZE)
    # sample_hidden = encoder.initialize_hidden_state()
    # sample_output, sample_hidden= encoder(source_batch, sample_hidden)
    # print ('Encoder output shape: (batch size, sequence length, units) {}'.format(sample_output.shape))
    # print ('Encoder Hidden state shape: (batch size, units) {}'.format(sample_hidden.shape))

    # attention_layer= BahdanauAttention(10)  original/ 10일 수가 없어서 바꿈
    attention_layer= BahdanauAttention(units)
    # attention_result, attention_weights = attention_layer(sample_hidden, sample_output)

    decoder= Decoder(target_vocab_size, embedding_dim, units, BATCH_SIZE)
    # sample_decoder_output, _, _= decoder(tf.random.uniform((BATCH_SIZE,1)), sample_hidden, sample_output)

    optimizer = tf.keras.optimizers.Adam()

    loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction="none")

    def loss_function(real, pred):
        mask = tf.math.logical_not(tf.math.equal(real, 0))
        loss_ = loss_object(real, pred)
        
        mask = tf.cast(mask, dtype=loss_.dtype)
        loss_ *= mask
        
        return tf.reduce_mean(loss_)

    EPOCHS=20

    for epoch in range(EPOCHS):

        start = time.time()
        enc_hidden = encoder.initialize_hidden_state()
        total_loss = 0
        # train the model using data in bataches 

        for (batch, (inp, targ)) in enumerate(dataset.take(steps_per_epoch)):
            
            batch_loss = train_step(inp, targ, enc_hidden)
            total_loss += batch_loss

            if batch % 100 == 0:
                print('Epoch {} Batch {} Loss {}'.format(epoch + 1,
                                                            batch,                                                   
                                                    batch_loss.numpy()))
        print('Epoch {} Loss {}'.format(epoch + 1,
                                            total_loss / steps_per_epoch))
        print('Time taken for 1 epoch {} sec\n'.format(time.time() - start))

    #Calculating the max length of the source and target sentences
    max_target_length= max(len(t) for t in  target_tensor)
    max_source_length= max(len(t) for t in source_tensor)

    translate(u'I am going to work.')
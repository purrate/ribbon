import pandas as pd
import numpy as np
import sys, os, re, csv, codecs
import tensorflow as tf
from tensorflow import keras

from keras.preprocessing.sequence import pad_sequences
from keras.layers import Dense, Input, LSTM, Embedding, Conv1D, concatenate, Flatten, Concatenate
from keras.layers import  BatchNormalization, Dropout, Activation, SpatialDropout1D, MaxPooling1D, GlobalAveragePooling1D, GlobalMaxPooling1D
from keras.layers import Bidirectional, GlobalMaxPool1D
from keras.models import Model
from keras import initializers, regularizers, constraints, optimizers, layers
from flask import Flask, request, jsonify
global tokenizer
tokenizer = tf.keras.preprocessing.text.Tokenizer()
global metrics
metrics=[tf.keras.metrics.BinaryAccuracy()]
# import matplotlib.pyplot as plt


from sklearn.metrics import hamming_loss
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.metrics import f1_score 

def prob_to_class(arr):   # converts probailities to class labes based on 0.5 threshold
        global r,c 
        r,c = arr.shape
        global predict
        predict =np.zeros((r, c))
        for i in range(r):
            for j in range(c):
                if arr[i,j]>0.5:
                    predict[i,j] = 1
        return predict
  
app = Flask(__name__)
loaded_model = None
def load_model():
    global loaded_model
    multi_train = pd.read_csv('train.csv')

    multi_validation = pd.read_csv('dev.csv')

    multi_test = pd.read_csv('test.csv')


    #
    multi_train.head()

    #
    multi_train.rename(columns={"Ogling/Facial Expressions/Staring": "Staring", "Touching /Groping": "Touching"}, inplace=True)
    multi_validation.rename(columns={"Ogling/Facial Expressions/Staring": "Staring", "Touching /Groping": "Touching"},inplace=True)
    multi_test.rename(columns={"Ogling/Facial Expressions/Staring": "Staring", "Touching /Groping": "Touching"},inplace=True)
    #
    # removing the duplicates to avoid clash

    multi_train.drop_duplicates(keep = 'first', inplace=True)
    multi_validation.drop_duplicates(keep = 'first', inplace=True)
    multi_test.drop_duplicates(keep = 'first', inplace=True)

    #
    train = pd.concat([multi_train, multi_validation], axis=0 ) # concatinating the train and validation set 
    train.head()

    #
    test = multi_test  
    train.shape, test.shape

    #
    y_train = train[['Commenting', 'Staring', 'Touching']].values
    y_test = test[['Commenting', 'Staring', 'Touching']].values

    #
    # Text Data 
    description_train_text =list(train['Description'].values)
    description_test_text = list(test['Description'].values)

    # tokenizing
    # global tokenizer
    # tokenizer = tf.keras.preprocessing.text.Tokenizer()
    # fit on training data
    tokenizer.fit_on_texts(description_train_text)     
    train_description_sequences = tokenizer.texts_to_sequences(description_train_text)
    test_description_sequences = tokenizer.texts_to_sequences(description_test_text)


    vocab_size = len(tokenizer.word_index) + 1

    # padding

    train_description_padded = pad_sequences(
            train_description_sequences, maxlen=300, dtype='int32', padding='post',  
            truncating='post')                                                  

    test_description_padded = pad_sequences(
            test_description_sequences, maxlen=300, dtype='int32', padding='post', 
            truncating='post')         

    #---
    # Exact match ratio
    def exact_match_ratio(y_true, y_pred):
        global MR
        MR = np.all(y_pred == y_true, axis=1).mean()
        return MR

    #---
    # conver probabilities to label

    #---
    from gensim.models.fasttext import FastText
    # %matplotlib inline 
    import nltk
    #---
    embedding_size = 300  #  size of the embedding vector
    window_size = 20    # size of the number of words occurring before and after
    min_word = 5  # minimum frequency of a word 
    down_sampling = 1e-2   # most frequently occurring word will be randomly down sampled 

    #---
    word_punctuation_tokenizer = nltk.WordPunctTokenizer()
    word_tokenized_corpus = [word_punctuation_tokenizer.tokenize(sent) for sent in description_train_text]

    #---
    ##%%time shud have been there
    ft_model = FastText(word_tokenized_corpus,
                        vector_size=embedding_size,
                        window=window_size,
                        min_count=min_word,
                        sample=down_sampling,
                        sg=1,
                        epochs=100)

    #---
    embedding_matrix_fast_text = np.zeros((vocab_size, 300))
    for word, i in tokenizer.word_index.items():
        try:
            embedding_vector = ft_model.wv[word] 
        except:
            embedding_vector = np.zeros(300)

        if embedding_vector is not None:
           embedding_matrix_fast_text[i] = embedding_vector

    #---
    max_input=300
    inputs = Input(shape=(max_input,))  # input 
    embedding = Embedding(vocab_size, embedding_size, trainable=False) 

    embedding.build((None,))
    embedding.set_weights([embedding_matrix_fast_text])
    embeddings = embedding(inputs)

    x = SpatialDropout1D(0.35)(embeddings)

    x = Bidirectional(LSTM(128, return_sequences=True, dropout=0.15, recurrent_dropout=0.15))(x)
    x = Conv1D(64, kernel_size=3, padding='valid', kernel_initializer='glorot_uniform')(x)

    avg_pool = GlobalAveragePooling1D()(x)
    max_pool = GlobalMaxPooling1D()(x)  

    x = concatenate([avg_pool, max_pool])

    x = BatchNormalization()(x)

    x = Dropout(0.2)(x)   

    x = Dense(64, activation='relu')(x)

    x = Dropout(0.2)(x)   

    outputs = Dense(3, activation='sigmoid')(x)  # output

    model_4_fast_text = Model(inputs=inputs, outputs = outputs)  # model

    #---
    model_4_fast_text.compile(loss='binary_crossentropy', optimizer='adam', metrics=metrics)
    history = model_4_fast_text.fit(train_description_padded, y_train, batch_size=64, epochs=10, validation_data=(test_description_padded, y_test))

    #---
    score = model_4_fast_text.evaluate(test_description_padded, y_test, verbose=1)
    print("Accuracy:", score[1])

    #---
    import matplotlib.pyplot as plt
    plt.plot(history.history['binary_accuracy'])
    plt.plot(history.history['val_binary_accuracy'])

    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train','test'], loc='upper left')
    plt.show()

    #---
    y_pred = model_4_fast_text.predict(test_description_padded, batch_size=64)
    y_class = prob_to_class(y_pred)
    print(y_class)

    #---
    import tensorflow as tf

    # Defining hamming loss
    def f1_score(y_true, y_pred, threshold=0.5):
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.cast(tf.math.greater(y_pred, threshold), tf.float32)
        global true_positives
        true_positives = tf.reduce_sum(tf.cast(tf.math.logical_and(tf.equal(y_true, 1), tf.equal(y_pred, 1)), tf.float32), axis=0)
        global predicted_positives
        predicted_positives = tf.reduce_sum(y_pred, axis=0)
        global actual_positives
        actual_positives = tf.reduce_sum(y_true, axis=0)
        global precision
        precision = true_positives / (predicted_positives + tf.keras.backend.epsilon())
        global recall
        recall = true_positives / (actual_positives + tf.keras.backend.epsilon())
        global f1
        f1 = 2 * precision * recall / (precision + recall + tf.keras.backend.epsilon())
        return tf.reduce_mean(f1)

    def hamming_loss(y_true, y_pred, threshold=0.5):
        y_pred = tf.cast(tf.math.greater(y_pred, threshold), tf.float32)
        global hamming_losss
        hamming_losss = 1 - tf.reduce_mean(tf.reduce_mean(tf.cast(tf.equal(y_true, y_pred), tf.float32), axis=1))
        return hamming_losss

    # Assuming y_test and y_pred are defined
    # Compute F1 Score and Hamming Loss
    f1_score_value = f1_score(y_test, y_pred)
    hamming_loss_value = hamming_loss(y_test, y_pred)

    # Print the results
    print('Hamming Loss:', hamming_loss_value.numpy())
    print('hamming score:',1-hamming_loss_value.numpy())

    #---
    # Example of custom input
    custom_input_text = ["touching"]

    custom_input_sequences = tokenizer.texts_to_sequences(custom_input_text)
    custom_input_padded = pad_sequences(custom_input_sequences, maxlen=300, dtype='int32', padding='post', truncating='post')


    custom_predictions = model_4_fast_text.predict(custom_input_padded)

    custom_class_labels = prob_to_class(custom_predictions)

    print("Predicted probabilities for custom input:")
    print(custom_predictions)
    print("Predicted class labels for custom input:")
    print(custom_class_labels)
    
    loaded_model=model_4_fast_text

    #---
load_model()

print("loaded")
@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json(force=True)
        input_data = data['input_data']
        custom_input_sequences = tokenizer.texts_to_sequences(input_data)
        custom_input_padded = pad_sequences(custom_input_sequences, maxlen=300, dtype='int32', padding='post', truncating='post')
        # Use the global loaded_model for prediction
        predictions = loaded_model.predict(custom_input_padded)
        prediction = prob_to_class(predictions)
        print(prediction)
        # Return the prediction as JSON
        prediction_list = prediction.tolist()
        print(prediction_list)
        return jsonify({'output': prediction_list})

    except Exception as e:
        return jsonify({'error': str(e)})




#---
if __name__ == '__main__':
    app.run(port=5000)
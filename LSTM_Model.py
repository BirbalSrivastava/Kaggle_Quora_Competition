########################################
#<LSTM Kaggle Quora competiton> Birbal Srivastava </>
########################################
## import packages
########################################
import re
import csv
import codecs
import numpy as np
import pandas as pd

from nltk.corpus import stopwords
from keras.layers import Dense, Input, LSTM, GRU, Embedding, Dropout
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
from keras.layers.merge import concatenate
from keras.models import Model
from keras.layers.normalization import BatchNormalization
from keras.callbacks import EarlyStopping, ModelCheckpoint
from nltk.stem import SnowballStemmer
########################################
## parameters
########################################
home_dir = './'
validation_ratio = 0.1
training_file = home_dir + 'train.csv'
testing_file = home_dir + 'test.csv'
max_sentence_length = 30
vocabulary_size = 200000
embedding_dimensions = 100

number_of_lstms = 196
number_of_dense_layers = 131
drop_out_rate_lstm = 0.26
drop_out_rate_dense_layers = 0.33

act = 'relu'
weight_balancing = True

model_name = 'kaggle_LSTM_%d_%d_%.2f_%.2f' % (number_of_lstms, number_of_dense_layers, drop_out_rate_lstm, \
                                       drop_out_rate_dense_layers)

########################################
## Embedding
########################################
print('Kaggle LSTM Using Glove 100 embedding')
embedding_index = {}
glove_100 = "glove.6B.100d.txt"
with open(glove_100, encoding='utf-8') as embedding_file:
    for eachline in embedding_file:
        line_element = eachline.split(' ')
        word = line_element[0]
        embedding = np.asarray(line_element[1:], dtype='float32')
        embedding_index[word] = embedding

print('Words embeddings is : %d' % len(embedding_index))
########################################
## process texts in datasets
########################################
print('Processing text dataset')

def tokenize_text_to_list_of_words(text, remove_stopwords=False, stem_words=False):
    text = text.lower().split()
    # Optionally, remove stop words
    if remove_stopwords:
        stops = set(stopwords.words("english"))
        text = [w for w in text if not w in stops]

    text = " ".join(text)

    # Clean the text
    text = re.sub(r"[^A-Za-z0-9^,!.\/'+-=]", " ", text)
    text = re.sub(r"what's", "what is ", text)
    text = re.sub(r"\'s", " ", text)
    text = re.sub(r"\'ve", " have ", text)
    text = re.sub(r"can't", "cannot ", text)
    text = re.sub(r"n't", " not ", text)
    text = re.sub(r"i'm", "i am ", text)
    text = re.sub(r"\'re", " are ", text)
    text = re.sub(r"\'d", " would ", text)
    text = re.sub(r"\'ll", " will ", text)
    text = re.sub(r",", " ", text)
    text = re.sub(r"\.", " ", text)
    text = re.sub(r"!", " ! ", text)
    text = re.sub(r"\/", " ", text)
    text = re.sub(r"\^", " ^ ", text)
    text = re.sub(r"\+", " + ", text)
    text = re.sub(r"\-", " - ", text)
    text = re.sub(r"\=", " = ", text)
    text = re.sub(r"'", " ", text)
    text = re.sub(r"(\d+)(k)", r"\g<1>000", text)
    text = re.sub(r":", " : ", text)
    text = re.sub(r" e g ", " eg ", text)
    text = re.sub(r" b g ", " bg ", text)
    text = re.sub(r" u s ", " american ", text)
    text = re.sub(r"\0s", "0", text)
    text = re.sub(r" 9 11 ", "911", text)
    text = re.sub(r"e - mail", "email", text)
    text = re.sub(r"j k", "jk", text)
    text = re.sub(r"\s{2,}", " ", text)

    #shorten word to stems
    if stem_words:
        text = text.split()
        stemmer = SnowballStemmer('english')
        stemmed_words = [stemmer.stem(word) for word in text]
        text = " ".join(stemmed_words)

    # Return a list of words
    return (text)


question_one = []
question_two = []
is_duplicate_value = []
with codecs.open(training_file, encoding='utf-8') as embedding_file:
    reader = csv.reader(embedding_file, delimiter=',')
    header = next(reader)
    for line_element in reader:
        question_one.append(tokenize_text_to_list_of_words(line_element[3]))
        question_two.append(tokenize_text_to_list_of_words(line_element[4]))
        is_duplicate_value.append(int(line_element[5]))
print('Found %s texts in train.csv' % len(question_one))

question_one_in_testdataset = []
question_two_in_testdataset = []
test_ids = []
with codecs.open(testing_file, encoding='utf-8') as embedding_file:
    reader = csv.reader(embedding_file, delimiter=',')
    header = next(reader)
    for line_element in reader:
        question_one_in_testdataset.append(tokenize_text_to_list_of_words(line_element[1]))
        question_two_in_testdataset.append(tokenize_text_to_list_of_words(line_element[2]))
        test_ids.append(line_element[0])
print('Found %s texts in test.csv' % len(question_one_in_testdataset))

tokenizer = Tokenizer(num_words=vocabulary_size)
tokenizer.fit_on_texts(question_one + question_two + question_one_in_testdataset + question_two_in_testdataset)

question_one_sequence = tokenizer.texts_to_sequences(question_one)
question_two_sequence = tokenizer.texts_to_sequences(question_two)
test_sequences_1 = tokenizer.texts_to_sequences(question_one_in_testdataset)
test_sequences_2 = tokenizer.texts_to_sequences(question_two_in_testdataset)

word_index = tokenizer.word_index
print('Found %s unique tokens' % len(word_index))

padded_question_one = pad_sequences(question_one_sequence, maxlen=max_sentence_length)
padded_question_two = pad_sequences(question_two_sequence, maxlen=max_sentence_length)
is_duplicate_value = np.array(is_duplicate_value)
print('Shape of data tensor:', padded_question_one.shape)
print('Shape of label tensor:', is_duplicate_value.shape)

padded_question_one_testdata = pad_sequences(test_sequences_1, maxlen=max_sentence_length)
padded_question_two_testdata = pad_sequences(test_sequences_2, maxlen=max_sentence_length)
test_ids = np.array(test_ids)

########################################
## prepare embeddings
########################################
print('Preparing embedding matrix')

number_of_words = min(vocabulary_size, len(word_index)) + 1

embedding_matrix = np.zeros((number_of_words, embedding_dimensions))
for word, i in word_index.items():
    if i > vocabulary_size:
        continue
    embed_vec = embedding_index.get(word)
    if embed_vec is not None:
        embedding_matrix[i] = embed_vec
print('Number of null/empty word embeddings: %d' % np.sum(np.sum(embedding_matrix, axis=1) == 0))

########################################
## sample train/validation data
########################################
np.random.seed(1234)
randomly_permuted = np.random.permutation(len(padded_question_one))
training_index = randomly_permuted[:int(len(padded_question_one) * (1 - validation_ratio))]
validation_index = randomly_permuted[int(len(padded_question_one) * (1 - validation_ratio)):]

question_one_training = np.vstack((padded_question_one[training_index], padded_question_two[training_index]))
question_two_training = np.vstack((padded_question_two[training_index], padded_question_one[training_index]))
training_response_label = np.concatenate((is_duplicate_value[training_index], is_duplicate_value[training_index]))

question_one_validation = np.vstack((padded_question_one[validation_index], padded_question_two[validation_index]))
question_two_validation = np.vstack((padded_question_two[validation_index], padded_question_one[validation_index]))
validation_response_label = np.concatenate((is_duplicate_value[validation_index], is_duplicate_value[validation_index]))

weight_value = np.ones(len(validation_response_label))
if weight_balancing:
    weight_value *= 0.48
    weight_value[validation_response_label == 0] = 1.30

########################################
## Deep Network structure
########################################
embedding_layer = Embedding(number_of_words,
                            embedding_dimensions,
                            weights=[embedding_matrix],
                            input_length=max_sentence_length,
                            trainable=False)
lstm_layer = LSTM(number_of_lstms, dropout=drop_out_rate_lstm, recurrent_dropout=drop_out_rate_lstm)

sequence_1_input = Input(shape=(max_sentence_length,), dtype='int32')
embedded_sequences_1 = embedding_layer(sequence_1_input)
layer_one = lstm_layer(embedded_sequences_1)

sequence_2_input = Input(shape=(max_sentence_length,), dtype='int32')
embedded_sequences_2 = embedding_layer(sequence_2_input)
layer_two = lstm_layer(embedded_sequences_2)

merged = concatenate([layer_one, layer_two])
merged = Dropout(drop_out_rate_dense_layers)(merged)
merged = BatchNormalization()(merged)

merged = Dense(number_of_dense_layers, activation=act)(merged)
merged = Dropout(drop_out_rate_dense_layers)(merged)
merged = BatchNormalization()(merged)

predictions = Dense(1, activation='sigmoid')(merged)

########################################
## add class weight
########################################
if weight_balancing:
    class_weight = {0: 1.30, 1: 0.48}
else:
    class_weight = None

########################################
## Training starts now
########################################
model = Model(inputs=[sequence_1_input, sequence_2_input], \
              outputs=predictions)
model.compile(loss='binary_crossentropy',
              optimizer='nadam',
              metrics=['acc'])
# model.summary()
print(model_name)

early_stopping = EarlyStopping(monitor='val_loss', patience=5)
bst_model_path = model_name + '.h5'
model_checkpoint = ModelCheckpoint(bst_model_path, save_best_only=True, save_weights_only=True)

hist = model.fit([question_one_training, question_two_training], training_response_label, \
                 validation_data=([question_one_validation, question_two_validation], validation_response_label, weight_value), \
                 epochs=30, batch_size=2048, shuffle=True, \
                 class_weight=class_weight, callbacks=[early_stopping, model_checkpoint])

model.load_weights(bst_model_path)
best_validation_score = min(hist.history['val_loss'])

########################################
## The submissions
########################################

predictions = model.predict([padded_question_one_testdata, padded_question_two_testdata], batch_size=8192, verbose=1)
predictions += model.predict([padded_question_two_testdata, padded_question_one_testdata], batch_size=8192, verbose=1)
predictions /= 2

submission = pd.DataFrame({'test_id':test_ids, 'is_duplicate':predictions.ravel()})
submission.to_csv('%.4f_' % (best_validation_score) + model_name + '.csv', index=False)

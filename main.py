'''
LINKS ÚTEIS:
https://nlpforhackers.io/training-pos-tagger/
http://www.nltk.org/book/ch05.html
http://nilc.icmc.usp.br/macmorpho/
http://www.nltk.org/howto/portuguese_en.html
https://pythonprogramming.net/part-of-speech-tagging-nltk-tutorial/
'''

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import nltk
	#nltk.download("mac_morpho")
	#nltk.download('stopwords')
	#nltk.download('averaged_perceptron_tagger')
from nltk.corpus import mac_morpho, stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from keras import initializers
from keras.models import Sequential
from keras.layers import Dense, LSTM, InputLayer, Bidirectional, TimeDistributed, Embedding, Activation
from keras.backend import clear_session
from keras.preprocessing.sequence import pad_sequences

from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.feature_extraction.text import CountVectorizer
import tensorflow as tf
from graphs import graph_by_class, graph_by_window_size


def pre_process():
    current_dir = os.getcwd()
    train_text = open(current_dir+'/corpus/macmorpho-train.txt', 'r')
    dev_text = open(current_dir+'/corpus/macmorpho-dev.txt', 'r')
    test_text = open(current_dir+'/corpus/macmorpho-test.txt', 'r')

    sentences_train = []
    sentences_dev = []
    sentences_test = []
    #TRAIN DATA
    for line in train_text.readlines():
    	tudo = line.replace('\n', '').split(' ')
    	train = [nltk.tag.util.str2tuple(word, sep='_') for word in tudo]
    	sentences_train.append(train)
    	#print(train)
    MAX_LEGTH = 248


    classes = set([x[1] for x in train]) 
    #print(classes)

    #DEV DATA
    for line in dev_text.readlines():
    	tudo = line.replace('\n', '').split(' ')
    	dev = [nltk.tag.util.str2tuple(word, sep='_') for word in tudo]
    	sentences_dev.append(dev)
	#print(dev)

    #TEST DATA
    for line in test_text.readlines():
    	tudo = line.replace('\n', '').split(' ')
    	test = [nltk.tag.util.str2tuple(word, sep='_') for word in tudo]
    	sentences_test.append(test)
    	#print(test)

    print("Pre-processing done")
    return sentences_train, classes, sentences_dev,sentences_test
    #return sentences_train, train, classes, sentences_dev, dev, sentences_test, test

def splits_sentences(sentences_train, sentences_dev, sentences_test):
	#split words and tags
	train_sentence_words, train_sentence_tags =[], [] 
	for tagged_sentence in sentences_train:
	    sentence, tags = zip(*tagged_sentence)
	    train_sentence_words.append(np.array(sentence))
	    train_sentence_tags.append(np.array(tags))

	dev_sentence_words, dev_sentence_tags =[], [] 
	for tagged_sentence in sentences_dev:
	    sentence, tags = zip(*tagged_sentence)
	    dev_sentence_words.append(np.array(sentence))
	    dev_sentence_tags.append(np.array(tags))

	test_sentence_words, test_sentence_tags =[], [] 
	for tagged_sentence in sentences_test:
	    sentence, tags = zip(*tagged_sentence)
	    test_sentence_words.append(np.array(sentence))
	    test_sentence_tags.append(np.array(tags))

	#converts to numbers    
	return convert_to_numbers(train_sentence_words, train_sentence_tags, dev_sentence_words, dev_sentence_tags, test_sentence_words, test_sentence_tags)

def convert_to_numbers(train_sentence_words, train_sentence_tags, dev_sentence_words, dev_sentence_tags, test_sentence_words, test_sentence_tags):
	#converting to numbers
	words, tags = set([]), set([])
	 
	for s in train_sentence_words :
	    for w in s:
	        words.add(w.lower())
 
	for ts in train_sentence_tags:
	    for t in ts:
	        tags.add(t)

	word2index = {w: i + 2 for i, w in enumerate(list(words))}
	word2index['-PAD-'] = 0  # The special value used for padding
	word2index['-OOV-'] = 1  # The special value used for OOVs

	tag2index = {t: i + 1 for i, t in enumerate(list(tags))}
	tag2index['-PAD-'] = 0  # The special value used to padding

	#converting the word dataset to numbers

	train_sentences_X, val_sentences_X, test_sentences_X, train_tags_y, val_tags_y, test_tags_y = [], [], [], [], [], []

	#train	 
	for s in train_sentence_words:
	    s_int = []
	    for w in s:
	        try:
	            s_int.append(word2index[w.lower()])
	        except KeyError:
	            s_int.append(word2index['-OOV-'])
	 
	    train_sentences_X.append(s_int)

	#validation
	for s in dev_sentence_words:
	    s_int = []
	    for w in s:
	        try:
	            s_int.append(word2index[w.lower()])
	        except KeyError:
	            s_int.append(word2index['-OOV-'])
	 
	    val_sentences_X.append(s_int)
	#test 
	for s in test_sentence_words:
	    s_int = []
	    for w in s:
	        try:
	            s_int.append(word2index[w.lower()])
	        except KeyError:
	            s_int.append(word2index['-OOV-'])
	 
	    test_sentences_X.append(s_int)
	

	for s in train_sentence_tags:
	    train_tags_y.append([tag2index[t] for t in s])
	
	for s in dev_sentence_tags:
	    val_tags_y.append([tag2index[t] for t in s])
	 
	for s in test_sentence_tags:
	    test_tags_y.append([tag2index[t] for t in s])
	 
	print("Done number converting")

	return word2index, tag2index, train_sentences_X, val_sentences_X, test_sentences_X, train_tags_y, val_tags_y, test_tags_y

def to_categorical(sequences, categories):
    cat_sequences = []
    for s in sequences:
        cats = []
        for item in s:
            cats.append(np.zeros(categories))
            cats[-1][item] = 1.0
        cat_sequences.append(cats)
    return np.array(cat_sequences)

def create_model(window_size,train_sentences_X,train_tags_y,epochs,batch_size,val_sentences_X,val_tags_y, tag2index):
    
    model = Sequential()
    model.add(InputLayer(input_shape=(248,)))
    model.add(Embedding(len(word2index), 128))
    model.add(Bidirectional(LSTM(256, return_sequences=True)))
    model.add(TimeDistributed(Dense(len(tag2index))))
    model.add(Activation('softmax'))
    model.compile(loss='categorical_crossentropy', optimizer ='rmsprop',metrics=['accuracy'])
    print(model.summary())
    model.fit(train_sentences_X,train_tags_y,epochs=epochs, batch_size=batch_size,verbose=1,  validation_split=0.2)#validation_data=(val_sentences_X,val_tags_y))
    return model

def getOneHot(test_corpus): #returns one-hot index
    for i in range(0,len(test_corpus)):
        if(test_corpus[i]==1):
            return i

def main(window_size, epochs,batch_size, train, classes, dev, test):

    data_train,classes_train,vectorizer,corpus = return_training_data(train, window_size, epochs)
    
    data_val,classes_val,vectorizer2,corpus2 = return_validation_data(dev, window_size,epochs)

    model = create_model(window_size,data_train,classes_train,epochs,batch_size,data_val,classes_val)
    #generating test samples

    data_test = []
    classes_test = []
    data_test,classes_test,valor_test_por_classe,resultado_test_por_classe = return_testing_data(vectorizer, window_size, corpus, test)

    resultado = str(window_size) + '-' + str(epochs)


    #checa se a header existe
    if os.path.exists("results/total_accuracy.csv"):
        header_exists = True
    else:
        header_exists = False

    # if it does not exist, save the header
    with open("results/total_accuracy.csv", "a+") as f:
        if not header_exists:
            f.write("window_size,epochs,accuracy\n")
        f.write(str(window_size)+","+str(epochs)+","+str(model.evaluate(data_test,classes_test,batch_size=batch_size,verbose=2)[1])+"\n")


    with open("results/"+resultado+'.csv', "w") as f:
        f.write("index,accuracy\n")

    classes_list = vectorizer.get_feature_names()# will be used to return each class's accuracy, but without using an index

    for index in valor_test_por_classe:
        score = model.evaluate(valor_test_por_classe[index], resultado_test_por_classe[index], batch_size = batch_size, verbose = 2)
        with open("results/"+resultado+".csv","a") as f:
                f.write(str(classes_list[index])+","+str(score[1])+"\n")

    graph_by_class("results/"+resultado+".csv",window_size,epochs) # generating graphics
    del model


sentences_train, classes, sentences_dev,sentences_test = pre_process()
word2index, tag2index, train_sentences_X, val_sentences_X, test_sentences_X, train_tags_y, val_tags_y, test_tags_y = splits_sentences(sentences_train, sentences_dev,sentences_test)

window_size = 3
batch_size = 120
epochs=1

print(tag2index)

'''
train_sentences_X = pad_sequences(train_sentences_X, maxlen=248, padding='post')
test_sentences_X = pad_sequences(test_sentences_X, maxlen=248, padding='post')
val_sentences_X = pad_sequences(val_sentences_X, maxlen=248, padding='post')
val_sentences_X = pad_sequences(val_sentences_X, maxlen=248, padding='post')
train_tags_y = pad_sequences(train_tags_y, maxlen=248, padding='post')
test_tags_y = pad_sequences(test_tags_y, maxlen=248, padding='post')

#cat_train_tags_y = to_categorical(train_tags_y, len(tag2index))
#print(cat_train_tags_y)

#model = create_model(window_size,train_sentences_X,to_categorical(train_tags_y, len(tag2index)),epochs,batch_size,val_sentences_X,to_categorical(val_tags_y, len(tag2index)), tag2index)
'''
'''
text = [x[1] for x in train]
x = pd.Series(text)
x = x.value_counts(normalize=True)
print(x)



for ep in range (1,6):
    for window in range(2,6):
        #window_size, batches and epochs
        window_size = window
        epochs = ep
        batch_size = 48 #8192
        print("--------------------------------------------------")
        print("Doing for window_size = ", window_size," and epochs = ", epochs)
        main(window_size, epochs,batch_size, train, classes, dev, test)
        clear_session()
        tf.keras.backend.clear_session()
#the above can change, i'll make a function for that and etc
#graph_by_window_size("results/total_accuracy.csv")

'''
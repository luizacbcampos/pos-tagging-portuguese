'''
LINKS ÚTEIS:
https://nlpforhackers.io/training-pos-tagger/
http://www.nltk.org/book/ch05.html
http://nilc.icmc.usp.br/macmorpho/
http://www.nltk.org/howto/portuguese_en.html
https://pythonprogramming.net/part-of-speech-tagging-nltk-tutorial/
'''

import os
import numpy as np
import nltk
	#nltk.download("mac_morpho")
	#nltk.download('stopwords')
	#nltk.download('averaged_perceptron_tagger')
from nltk.corpus import mac_morpho, stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from keras import initializers
from keras.models import Sequential
from keras.layers import Dense, LSTM
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.feature_extraction.text import CountVectorizer
from graphs import graph_by_class, graph_by_window_size


def pre_process():
    current_dir = os.getcwd()
    train_text = open(current_dir+'/corpus/macmorpho-train.txt', 'r')
    dev_text = open(current_dir+'/corpus/macmorpho-dev.txt', 'r')
    test_text = open(current_dir+'/corpus/macmorpho-test.txt', 'r')



    #TRAIN DATA
    train_data = train_text.read().replace('\n', '').split(' ')
    train = [nltk.tag.util.str2tuple(word, sep='_') for word in train_data]
    #print(train)
    classes = set([x[1] for x in train]) 
    #print(classes)


    #DEV DATA
    dev_data = dev_text.read().replace('\n', '').split(' ')
    dev = [nltk.tag.util.str2tuple(word, sep='_') for word in dev_data]
    #print(dev)

    #TEST DATA
    test_data = test_text.read().replace('\n', '').split(' ')
    test = [nltk.tag.util.str2tuple(word, sep='_') for word in test_data]
    #print(test)

    print("Pre-processing done")
    return train, classes, dev, test


def return_training_data(train, window_size, epochs):
    data_train = []
    classes_train = []
    text = [x[1] for x in train] #text is only the classes from train (ou seja, todo x[1])
    vectorizer = CountVectorizer(lowercase=False, token_pattern='[A-Z;+;-]+')
    corpus = vectorizer.fit_transform(text)
    corpus = corpus.toarray()

    window_start=0 #sliding window
    window_end=window_size-1 #sliding window
    while(window_end<len(corpus)-1):
        window_end += 1
        data_train.append(corpus[window_start:window_end])
        classes_train.append(corpus[window_end])
        window_start += 1

    data_train = np.array(data_train)
    classes_train = np.array(classes_train)

    return data_train,classes_train,vectorizer,corpus

def create_model(window_size,data_train,classes_train,epochs,batch_size):
    np.random.seed(7)
    model = Sequential()
    model.add(LSTM(50,input_shape=(window_size,26)))
    model.add(Dense(25,activation='relu'))
    model.add(Dense(2, activation='sigmoid'))
    model.add(Dense(26, activation=lambda x: x))
    model.compile(loss='binary_crossentropy',optimizer ='adam',metrics=['accuracy'])
    model.fit(data_train,classes_train,epochs=epochs, batch_size=batch_size,validation_split=0.2,verbose=1)
    return model

def getOneHot(test_corpus): #returns one-hot index
    for i in range(0,len(test_corpus)):
        if(test_corpus[i]==1):
            return i

def return_testing_data(vectorizer, window_size, corpus, test):
    data_test = []
    classes_test = []
    test = [x[1] for x in test] #text is only the classes from test (ou seja, todo x[1])

    test_corpus = vectorizer.fit_transform(test)
    test_corpus = test_corpus.toarray()

    valor_test_por_classe = {} #o valor conhecido
    resultado_test_por_classe = {} #o valor previsto

    window_start=0 #sliding window
    window_end=window_size-1 #sliding window

    while(window_end<len(test_corpus)-1):
        index = getOneHot(test_corpus[(window_end)])
        if index not in valor_test_por_classe:
            valor_test_por_classe[index] = []

        if index not in resultado_test_por_classe:
            resultado_test_por_classe[index] = []

        window_end += 1
        valor_test_por_classe[index].append(test_corpus[window_start:window_end])
        resultado_test_por_classe[index].append(test_corpus[window_end])

        data_test.append(corpus[window_start:window_end])
        classes_test.append(corpus[window_end])

        window_start += 1

    for i in valor_test_por_classe:
        valor_test_por_classe[i] = np.array(valor_test_por_classe[i])
    for i in resultado_test_por_classe:
        resultado_test_por_classe[i] = np.array(resultado_test_por_classe[i])

    data_test = np.array(data_test)
    classes_test = np.array(classes_test)

    return data_test,classes_test,valor_test_por_classe,resultado_test_por_classe


train, classes, dev, test = pre_process()

#window_size, batches and epochs
window_size = 3
epochs = 1
batch_size = 8192
#the above can change, i'll make a function for that and etc

data_train,classes_train,vectorizer,corpus = return_training_data(train, window_size, epochs)
model = create_model(window_size,data_train,classes_train,epochs,batch_size)
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

with open("results/total_accuracy.csv", 'a+') as f:
    f.write("index,accuracy\n")

with open("results/"+resultado+'.csv', "w") as f:
    f.write("index,accuracy\n")

classes_list = vectorizer.get_feature_names()# will be used to return each class's accuracy, but without using an index

for index in valor_test_por_classe:
    score = model.evaluate(valor_test_por_classe[index], resultado_test_por_classe[index], batch_size = batch_size, verbose = 2)
    with open("../results/"+resultado+".csv","a") as f:
            f.write(str(classes_list[index])+","+str(score[1])+"\n")

graph_by_class("/results/"+result_file_name+".csv",window_size,epochs) # generating graphics
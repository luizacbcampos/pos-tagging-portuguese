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
#from graphs import graph_by_class, graph_by_window_size



current_dir = os.getcwd()
train_text = mac_morpho.raw(current_dir+'/corpus/macmorpho-train.txt')
dev_text = mac_morpho.raw(current_dir+'/corpus/macmorpho-dev.txt')
test_text = mac_morpho.raw(current_dir+'/corpus/macmorpho-test.txt')



train_data = word_tokenize(train_text)
train_toclean = [nltk.tag.util.str2tuple(word, sep='_') for word in train_data]
#print(train)

train = [ x for x in train_toclean if x[1] is not None] #cleans train from Not classified instances
classes = set([x[1] for x in train]) 
print(classes)


dev_data = word_tokenize(dev_text)
dev_toclean = [nltk.tag.util.str2tuple(word, sep='_') for word in dev_data]
#print(dev)

test_data = word_tokenize(test_text)
test_toclean = [nltk.tag.util.str2tuple(word, sep='_') for word in test_data]








'''

custom_sent_tokenizer = PunktSentenceTokenizer(train_text)
tokenized = custom_sent_tokenizer.tokenize(dev_text)

def process_content():
    try:
        for i in tokenized[:5]:
            words = nltk.word_tokenize(i)
            tagged = nltk.pos_tag(words)
            print(tagged)

    except Exception as e:
        print(str(e))


process_content() 

stop_words = stopwords.words("portuguese")
tagged_sentences = mac_morpho.tagged_sents()
'''
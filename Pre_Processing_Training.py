# -*- coding: utf-8 -*-
"""
Created on Thu Jan 30 21:02:50 2020

@author: mukmc
"""

#STANDARD IMPORT FUNCTIONS#
##################################

import numpy as np
import pandas as pd
from tqdm import tqdm
import string
import matplotlib.pyplot as plt
from sklearn.decomposition import NMF, LatentDirichletAllocation, TruncatedSVD
from sklearn.feature_extraction.text import CountVectorizer
import concurrent.futures
import time
import pyLDAvis.sklearn
from pylab import bone, pcolor, colorbar, plot, show, rcParams, savefig
import warnings
warnings.filterwarnings('ignore')

%matplotlib inline
import os
#print(os.listdir("../input"))
#print(os.listdir("../input/embeddings"))

# Plotly based imports for visualization
from plotly import tools
import plotly.plotly as py
from plotly.offline import init_notebook_mode, iplot
init_notebook_mode(connected=True)
import plotly.graph_objs as go
import plotly.figure_factory as ff

# spaCy based imports
import spacy
from spacy.lang.en.stop_words import STOP_WORDS
from spacy.lang.en import English

# Keras based imports
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils.vis_utils import plot_model
from keras.models import Model
from keras.layers import Input
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import Dropout
from keras.layers import Embedding
from keras.layers import LSTM, GRU, Bidirectional
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D
from keras.layers.merge import concatenate



####################################################
pos=[]
for root, dirs, files in os.walk("Data/train/pos"):
    for filename in files:
        pos.append(filename)
        

neg=[]
for root, dirs, files in os.walk("Data/train/neg"):
    for filename in files:
        neg.append(filename)
        
plist=[]
plab=[]

for i in pos:
    g=open("Data/train/pos/"+i,encoding="utf8")
    plist.append(g.read())
    g.close
    plab.append("Postitive")


nlist=[]

nlab=[]

for i in neg:
    g=open("Data/train/neg/"+i,encoding="utf8")
    nlist.append(g.read())
    g.close
    nlab.append("Negative")

#test
    
postest=[]
for root, dirs, files in os.walk("Data/test/pos"):
    for filename in files:
        postest.append(filename)
        

negtest=[]
for root, dirs, files in os.walk("Data/test/neg"):
    for filename in files:
        negtest.append(filename)
        

ptest = []
ptestlab=[]

ntest=[]
ntestlab=[]



for i in postest:
    g=open("Data/test/pos/"+i,encoding="utf8")
    ptest.append(g.read())
    g.close
    ptestlab.append("Postitive")
    
for i in negtest:
    g=open("Data/test/neg/"+i,encoding="utf8")
    ntest.append(g.read())
    g.close
    ntestlab.append("Negative")

#DataFrame Creation and saving for future usage
    
import pandas as pd

prevtrain = pd.DataFrame(plist,columns=['text'])
prevtrain['label']=plab
prevtrain.to_csv("Processed_Data/train_positive.csv")

nrevtrain = pd.DataFrame(nlist,columns=['text'])
nrevtrain['label']=nlab
nrevtrain.to_csv("Processed_Data/train_negative.csv")


prevtest = pd.DataFrame(ptest,columns=['text'])
prevtest['label']=ptestlab
prevtest.to_csv("Processed_Data/test_positive.csv")


nrevtest = pd.DataFrame(ntest,columns=['text'])
nrevtest['label']=ntestlab
nrevtest.to_csv("Processed_Data/test_negative.csv")

#############################################################

prevtrain = pd.read_csv("Processed_Data/train_positive.csv")
prevtrain=prevtrain.drop(["Unnamed: 0"],axis=1)

nrevtrain = pd.read_csv("Processed_Data/train_negative.csv")
nrevtrain=nrevtrain.drop(["Unnamed: 0"],axis=1)


prevtest = pd.read_csv("Processed_Data/test_positive.csv")
prevtest=prevtest.drop(["Unnamed: 0"],axis=1)


nrevtest = pd.read_csv("Processed_Data/test_negative.csv")
nrevtest=nrevtest.drop(["Unnamed: 0"],axis=1)


train = prevtrain.append(nrevtrain)

train = train.sample(frac=1).reset_index(drop=True)

test = prevtest.append(nrevtest)

test = test.sample(frac=1).reset_index(drop=True)


dataset = train.append(test)

dataset.reset_index(inplace=True,drop=True)

##############################################################
### PRE PROCESSING ###
##############################################################

punctuations = string.punctuation

def punct_remover(my_str):
    my_str = my_str.lower()
    no_punct = ""
    for char in my_str:
       if char not in punctuations:
           no_punct = no_punct + char
    return no_punct

punctuations


tqdm.pandas()
reviews_train = train["text"].progress_apply(punct_remover)
reviews_test = test['text'].progress_apply(punct_remover)


def create_tokenizer(lines):
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(lines)
    return tokenizer

# calculate the maximum document length
def max_length(lines):
    return max([len(s.split()) for s in lines])
 
# encode a list of lines
def encode_text(tokenizer, lines, length):
    # integer encode
    encoded = tokenizer.texts_to_sequences(lines)
    # pad encoded sequences
    padded = pad_sequences(encoded, maxlen=length, padding='post')
    return padded

##############################################################
##### GLOVE EMBEDDING ######
##############################################################
    
embeddings_index = dict()
f = open('Data/glove.840B.300d/glove.840B.300d.txt',encoding='utf8')
for line in f:
    values = line.split(" ")
    word = values[0]
    coefs = np.asarray(values[1:], dtype='float32')
    embeddings_index[word] = coefs
f.close()


##############################################################
##### Tokenize and load into Embedding #####
##############################################################

embed_token = create_tokenizer(reviews_train)
vocabulary_size = 90000

embedding_matrix = np.zeros((vocabulary_size, 300))
for word, index in embed_token.word_index.items():
    if index > vocabulary_size - 1:
        break
    else:
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            embedding_matrix[index] = embedding_vector
            
########################################################################
#### Complex Model that we did not have resources to go ahead with ####
########################################################################


def define_model(length, vocab_size):
    # channel 1
    inputs1 = Input(shape=(length,))
    embedding1 = Embedding(vocabulary_size, 300, weights=[embedding_matrix])(inputs1)
    conv1 = Conv1D(filters=16, kernel_size=4, activation='relu')(embedding1)
    drop1 = Dropout(0.5)(conv1)
    lstm1 = Bidirectional(LSTM(10, return_sequences = True))(drop1)
    gru1 = Bidirectional(GRU(10, return_sequences = True))(lstm1)
    pool1 = MaxPooling1D(pool_size=2)(gru1)
    flat1 = Flatten()(pool1)
    # channel 2
    inputs2 = Input(shape=(length,))
    embedding2 = Embedding(vocabulary_size, 300, weights=[embedding_matrix])(inputs2)
    conv2 = Conv1D(filters=16, kernel_size=6, activation='relu')(embedding2)
    drop2 = Dropout(0.5)(conv2)
    lstm2 = Bidirectional(LSTM(10, return_sequences = True))(drop2)
    gru2 = Bidirectional(LSTM(10, return_sequences = True))(lstm2)
    pool2 = MaxPooling1D(pool_size=2)(gru2)
    flat2 = Flatten()(pool2)
    # channel 3
    inputs3 = Input(shape=(length,))
    embedding3 = Embedding(vocabulary_size, 300, weights=[embedding_matrix])(inputs3)
    conv3 = Conv1D(filters=16, kernel_size=8, activation='relu')(embedding3)
    drop3 = Dropout(0.5)(conv3)
    lstm3 = Bidirectional(LSTM(10, return_sequences = True))(drop3)
    gru3 = Bidirectional(GRU(10, return_sequences = True))(lstm3)
    pool3 = MaxPooling1D(pool_size=2)(gru3)
    flat3 = Flatten()(pool3)
    merged = concatenate([flat1, flat2, flat3])
    # interpretation
    dense1 = Dense(10, activation='relu')(merged)
    outputs = Dense(1, activation='sigmoid')(dense1)
    model = Model(inputs=[inputs1, inputs2, inputs3], outputs=outputs)
    # compile
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    # summarize
    print(model.summary())
    #plot_model(model, show_shapes=True, to_file='multichannel.png')
    return model

################################################################################
############ Actual Model that we went ahead with ##############
################################################################################
    

def new_mod(length, vocab_size):
    inputs1=Input(shape=(length,))
    embedding1 = Embedding(vocab_size,300,weights=[embedding_matrix])(inputs1)
    conv1 = Conv1D(filters=16, kernel_size=4, activation='relu')(embedding1)
    drop1 = Dropout(0.5)(conv1)
    lstm1 = Bidirectional(LSTM(20, return_sequences = True))(drop1)
    gru1 = Bidirectional(GRU(20, return_sequences = True))(lstm1)
    pool1 = MaxPooling1D(pool_size=2)(gru1)
    flat1 = Flatten()(pool1)
    dense1 = Dense(30, activation='relu')(flat1)
    outputs = Dense(1, activation='sigmoid')(dense1)
    model = Model(inputs=inputs1, outputs=outputs)
    # compile
    model.compile(loss='binary_crossentropy', optimizer='nadam', metrics=['accuracy'])
    # summarize
    print(model.summary())
    #plot_model(model, show_shapes=True, to_file='multichannel.png')
    return model


# Preprocess data

# create tokenizer
tokenizer = create_tokenizer(reviews_train)
# calculate max document length
length = max_length(reviews_train)
# calculate vocabulary size
vocab_size = len(tokenizer.word_index) + 1
print('Max document length: %d' % length)
print('Vocabulary size: %d' % vocab_size)
vocab_size=90000
# encode data
length=800
trainX = encode_text(tokenizer, reviews_train, length)
testX = encode_text(tokenizer, reviews_test, length)
print(trainX.shape, testX.shape)

import pickle

# saving
with open('Saved_models/tokenizer.pickle', 'wb') as handle:
    pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)

with open('Saved_models/tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)    

##################################################
############ Training the Model ############
##################################################


model = new_mod(length, vocab_size)
model.summary()

import tensorflow as tf
run_opts = tf.RunOptions(report_tensor_allocations_upon_oom = True)

#from keras.models import load_model
#model=load_model("Saved_models/senti.h5")

model.fit(trainX, pd.get_dummies(train['label'])['Postitive'].values, epochs=2, batch_size=256)

model.evaluate(testX, pd.get_dummies(test['label'])['Postitive'].values, batch_size=256)

model.save("Saved_models/senti.h5")

model.predict(encode_text(tokenizer,['absolutely ridiclous product . pls avoid '],length))


# -*- coding: utf-8 -*-
"""
Created on Sat Jan 27 08:21:58 2018
@author: Rana Mahmoud
"""

import os
#os.environ['THEANO_FLAGS']='mode=FAST_RUN,device=gpu,floatX=float32'

# Parameters matplotlib
# ==================================================
import matplotlib.pyplot as plt
plt.rcParams['figure.figsize'] = (10,10) 
#%matplotlib inline
import seaborn as sns
plt.switch_backend('agg') 

# Parameters General
# ==================================================
import codecs
import csv
import keras
import sklearn
import gensim
import random
import scipy
import pydot
#import commands
import glob
import numpy as np
import pandas as pd
import re

# Parameters theano
# ==================================================
#import theano
#print( 'using : ',(theano.config.device))

# Parameters keras
# ==================================================
#from keras.utils.visualize_util import plot ## to print the model arch
from keras.utils.vis_utils import plot_model ## to print the model arch
from keras.preprocessing import sequence
from keras.preprocessing.text import Tokenizer
#from keras.models import Sequential , Graph
from keras.models import Sequential

from keras.optimizers import SGD
from keras.layers.embeddings import Embedding
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.regularizers import l2#, activity_l2
#from keras.layers.core import Dense , Dropout , Activation , Merge , Flatten
from keras.layers import Dense , Dropout , Activation , Merge , Flatten
from keras.layers.convolutional import Convolution1D, MaxPooling1D
from keras.layers import Embedding , LSTM

from keras.models import Model
from keras.layers import Input
from keras.layers import TimeDistributed
from keras.layers import Merge, merge
#Merge is for layers, merge is for tensors.
from keras.utils.vis_utils import plot_model

# Parameters sklearn
# ==================================================
import sklearn.metrics
from sklearn import preprocessing
from sklearn.base import BaseEstimator
from sklearn.svm import LinearSVC , SVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import confusion_matrix, classification_report #get F1 score

# Parameters gensim
# ==================================================
from gensim.models.word2vec import Word2Vec
import gensim.models.keyedvectors
from gensim.models.doc2vec import Doc2Vec , TaggedDocument
from gensim.corpora.dictionary import Dictionary
import gensim

import datetime

##new cnn
import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Convolution1D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers.embeddings import Embedding
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D
from keras.preprocessing import sequence
import numpy as np
from sklearn.naive_bayes import ComplementNB

# Model Hyperparameters
# ==================================================
#
print( "Model Hyperparameters :")

embeddings_dim = 300
print ("embeddings_dim = " ,embeddings_dim)

filter_sizes = [3, 5, 7]
print ("filter_sizes = ",filter_sizes)

dropout_prob = [0.5,0.5]
print ("dropout_prob = ",dropout_prob)

# maximum number of words to consider in the representations
max_features = 30000
print ("max_features = " ,max_features)

# percentage of the data used for model training
val_split = 0.75
print ("val_split = ",val_split)


# number of classes
num_classes = 2
print ("num_classes = " ,num_classes)


# Training parameters
# ==================================================
#
num_epochs = 30
print ("num_epochs = ",num_epochs)


    
# Reading pre-trained word embeddings
# ==================================================
# load the embeddings
print ("")
print ("Reading pre-trained word embeddings...")
embeddings = dict( )
#embeddings = Word2Vec.load_word2vec_format("E:\\data\\cbow\\(CBOW58)-ASA-3B-CBOW-window5-3iter-d300-vecotrs.bin", binary=True,encoding='utf8', unicode_errors='ignore')
embeddings = gensim.models.KeyedVectors.load_word2vec_format("E:\\data\\cbow\\(CBOW58)-ASA-3B-CBOW-window5-3iter-d300-vecotrs.bin", binary=True,encoding='utf8', unicode_errors='ignore')
#embeddings = gensim.models.Word2Vec.load('E:\\data\\cbow\\Twt-CBOW\\Twt-CBOW')

datasets = list()
datasets = {
#        ('ASTD','E:\\data\\cbow\\ASTD-unbalanced-not-linked.csv',40),
#            ('ATT','E:\\data\\cbow\\ATT-unbalanced-not-linked.csv',568),
#            ('HTL','E:\\data\\cbow\\HTL-unbalanced-not-linked.csv',1110),
            ('MOV','E:\\data\\cbow\\MOV-unbalanced-not-linked.csv',2335)
#            ,
#            ('PROD','E:\\data\\cbow\\PROD-unbalanced-not-linked.csv',234)#,
#            ('RES','E:\\data\\cbow\\RES-unbalanced-not-linked.csv',539)#,
#            ('LABR','E:\\data\\cbow\\LABR-unbalanced-not-linked.csv',882)        
        }

for dataset_name, file_name, max_sent_len in datasets:
    # Reading csv data
    # ==================================================
    print ("Reading text data for classification and building representations...")
    data = []
    data = [ ( row["text"] , row["polarity"]  ) for row in csv.DictReader(open(file_name, encoding="utf8"), delimiter=',', quoting=csv.QUOTE_NONE) ]
    
    random.shuffle( data )
    train_size = int(len(data) * val_split)
    train_texts = [ txt for ( txt, label ) in data[0:train_size] ]
    test_texts = [ txt for ( txt, label ) in data[train_size:-1] ]
    train_labels = [ label for ( txt , label ) in data[0:train_size] ]
    test_labels = [ label for ( txt , label ) in data[train_size:-1] ]
    num_classes = len( set( train_labels + test_labels ) )
    tokenizer = Tokenizer(nb_words=max_features, filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n', split=" ")
    tokenizer.fit_on_texts(train_texts)
    train_sequences = sequence.pad_sequences( tokenizer.texts_to_sequences( train_texts ) , maxlen=max_sent_len )
    test_sequences = sequence.pad_sequences( tokenizer.texts_to_sequences( test_texts ) , maxlen=max_sent_len )
    train_matrix = tokenizer.texts_to_matrix( train_texts )
    test_matrix = tokenizer.texts_to_matrix( test_texts )
    embedding_weights = np.zeros( ( max_features , embeddings_dim ) )
    for word,index in tokenizer.word_index.items():
        if index < max_features:
            try: embedding_weights[index,:] = embeddings[word]
            except: embedding_weights[index,:] = np.random.uniform(-0.25,0.25,embeddings_dim)
    le = preprocessing.LabelEncoder( )
    le.fit( train_labels + test_labels )
    train_labels = le.transform( train_labels )
    test_labels = le.transform( test_labels )
    print ("Classes : " + repr( le.classes_ ))

 
    #DNN
    input_dim = max_sent_len
    from sklearn.preprocessing import LabelEncoder, OneHotEncoder
    labelencoder_train_labels = LabelEncoder()
    train_labels = labelencoder_train_labels.fit_transform(train_labels)
    #train_labels = keras.utils.to_categorical(train_labels, 4)
    onehotencoder = OneHotEncoder(categorical_features = [0])
    train_labels = train_labels.reshape((-1, 1))
    train_labels = onehotencoder.fit_transform(train_labels).toarray()
    labelencoder_test_labels = LabelEncoder()
    test_labels = labelencoder_test_labels.fit_transform(test_labels)
    onehotencoder = OneHotEncoder(categorical_features = [0])
    test_labels = test_labels.reshape((-1, 1))
    test_labels = onehotencoder.fit_transform(test_labels).toarray()
    if len(train_labels[0]) > len(test_labels[0]):
        zero_col = np.zeros(len(test_labels))
        test_labels = np.c_[test_labels, zero_col]
    if len(train_labels[0]) < len(test_labels[0]):
        zero_col = np.zeros(len(train_labels))
        train_labels = np.c_[train_labels, zero_col]
    # Initialising the ANN
    model = Sequential()
    lay1 = model.add(Dense(output_dim = 200, init = 'uniform', activation = 'relu', 
                         input_dim = input_dim))  
    ## Adding the second hidden layer
    lay2 = model.add(Dense(output_dim = 100, init = 'uniform', activation = 'relu'))
    # Adding the third laye
    lay3 = model.add(Dense(output_dim = 150, init = 'uniform', activation = 'relu'))
    # Adding the forth laye
#    lay4 = model.add(Dense(output_dim = 100, init = 'uniform', activation = 'relu'))
    # Adding the output layer
    lay5 = model.add(Dense(output_dim = num_classes, init = 'uniform', activation = 'softmax'))
    
    outputs    = [layer.output for layer in model.layers] 
    inputs    = [layer.input for layer in model.layers]
    
    # Compiling the ANN
    model.compile(optimizer = 'adam', loss = 'categorical_crossentropy', 
                       metrics = ['accuracy'])
    model.fit(train_sequences, train_labels, batch_size = 128, nb_epoch = 50, verbose=1)
    score, acc  = model.evaluate(test_sequences, test_labels,batch_size=64)
    
    # Print model summary
    # ==================================================
    print(model.summary())
    
    #Visualize the model in a graph
    plot_model(model, to_file='E:/model_plot_dnn.png', show_shapes=True, show_layer_names=True)
    
    
    with open('E:\\data\\Results.txt', 'a') as the_file:
        the_file.write(dataset_name)
        the_file.write('\n---------------------------------------\n')
        the_file.write('\nDNN\n')
        the_file.write('\nStart_Time\n')
        the_file.write(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
        the_file.write('\n')

    ##fit with writing to a file
    # Fitting the ANN to the Training set
    pd.DataFrame(model.fit(train_sequences, train_labels, batch_size = 128, nb_epoch = 50, verbose=1).history).to_csv("E:\\data\\cbow\\DNN_Results\\history_"+dataset_name+".csv")
    
    with open('E:\\data\\Results.txt', 'a') as the_file:
        the_file.write('\nEnd_Time\n')
        the_file.write(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
        the_file.write('\n')

    # Evaluate model keras
    # ==================================================
    print("Evaluate...")
    score, acc  = model.evaluate(test_sequences, test_labels,batch_size=64)
    print('Test score:', score)
    print('Test accuracy:', acc)
    
    with open('E:\\data\\Results.txt', 'a') as the_file:
        the_file.write('\nTest score: ')
        the_file.write(str(score))
        the_file.write('\nTest accuracy: ')
        the_file.write(str(acc))
        the_file.write('\n')
    
    # Evaluate model sklearn
    # ==================================================
    results = np.array(model.predict(test_sequences, batch_size=32))
    if num_classes != 2: results = results.argmax(axis=-1)
    else: results = (results > 0.5).astype('int32')
    print ("Accuracy-sklearn = " + repr( sklearn.metrics.accuracy_score( test_labels , results )  ))
    print ('RMSE',np.sqrt(((results - test_labels) ** 2).mean(axis=0)).mean() ) # Printing RMSE
    print (sklearn.metrics.classification_report( test_labels , results ))
    
    with open('E:\\data\\Results.txt', 'a') as the_file:
        the_file.write('\nAccuracy-sklearn = ')
        the_file.write(str(repr(sklearn.metrics.accuracy_score(test_labels, results))))
        the_file.write('\nRMSE: ')
        the_file.write(str(np.sqrt(((results - test_labels) ** 2).mean(axis=0)).mean()))
        the_file.write('\n')
        the_file.write('\n-----------------------------------------------------------\n')

    
    ###Sequential CNN
    classifier = Sequential()
    classifier.add(Embedding(max_features, embeddings_dim, input_length=max_sent_len, mask_zero=False, weights=[embedding_weights] ))
    classifier.add(Conv1D(filters=100, kernel_size=4, padding='same', activation='relu'))
    classifier.add(MaxPooling1D(pool_size=2))
    classifier.add(Conv1D(filters=100, kernel_size=4, padding='same', activation='relu'))
    classifier.add(MaxPooling1D(pool_size=2))
    classifier.add(Flatten())
    classifier.add(Dense(250, activation='relu'))
    classifier.add(Dense(num_classes, activation='softmax'))
    classifier.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
#      print(classifier.summary())
#    classifier.fit(train_sequences, train_labels, validation_data=(test_sequences, test_labels), epochs=2, batch_size=128, verbose=2)
    # Final evaluation of the model
#    scores = classifier.evaluate(test_sequences, test_labels, verbose=0)

    with open('E:\\data\\Results.txt', 'a') as the_file:
        the_file.write(dataset_name)
        the_file.write('\n---------------------------------------\n')
        the_file.write('\nSequential CNN\n')
        the_file.write('\nStart_Time\n')
        the_file.write(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
        the_file.write('\n')

    ##fit with writing to a file
    # Fitting the ANN to the Training set
    pd.DataFrame(classifier.fit(train_sequences, train_labels, batch_size = 128, nb_epoch = 30, verbose=1).history).to_csv("E:\\data\\cbow\\DNN_Results\\history_"+dataset_name+".csv")
    
    with open('E:\\data\\Results.txt', 'a') as the_file:
        the_file.write('\nEnd_Time\n')
        the_file.write(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
        the_file.write('\n')

    # Evaluate model keras
    # ==================================================
    print("Evaluate...")
    score, acc  = classifier.evaluate(test_sequences, test_labels,batch_size=64)
    print('Test score:', score)
    print('Test accuracy:', acc)
    
    with open('E:\\data\\Results.txt', 'a') as the_file:
        the_file.write('\nTest score: ')
        the_file.write(str(score))
        the_file.write('\nTest accuracy: ')
        the_file.write(str(acc))
        the_file.write('\n')
    
    # Evaluate model sklearn
    # ==================================================
    results = np.array(classifier.predict(test_sequences, batch_size=32))
    if num_classes != 2: results = results.argmax(axis=-1)
    else: results = (results > 0.5).astype('int32')
    print ("Accuracy-sklearn = " + repr( sklearn.metrics.accuracy_score( test_labels , results )  ))
    print ('RMSE',np.sqrt(((results - test_labels) ** 2).mean(axis=0)).mean() ) # Printing RMSE
    print (sklearn.metrics.classification_report( test_labels , results ))
    
    with open('E:\\data\\Results.txt', 'a') as the_file:
        the_file.write('\nAccuracy-sklearn = ')
        the_file.write(str(repr(sklearn.metrics.accuracy_score(test_labels, results))))
        the_file.write('\nRMSE: ')
        the_file.write(str(np.sqrt(((results - test_labels) ** 2).mean(axis=0)).mean()))
        the_file.write('\n')
        the_file.write('\n-----------------------------------------------------------\n')




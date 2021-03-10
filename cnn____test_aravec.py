# -*- coding: utf-8 -*-
"""
Created on Sat Jan 13 08:21:58 2018

@author: PAVILION G4
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
#embeddings = Word2Vec.load_word2vec_format("E:\data\cbow\(CBOW58)-ASA-3B-CBOW-window5-3iter-d300-vecotrs.bin", binary=True,encoding='utf8', unicode_errors='ignore')
embeddings = gensim.models.Word2Vec.load('E:\\data\\aravec\\Twt-CBOW')

datasets = list()
datasets = {
#        ('ASTD','E:\\data\\cbow\\ASTD-unbalanced-not-linked.csv',40),
#            ('ATT','E:\\data\\cbow\\ATT-unbalanced-not-linked.csv',568),
##            ('HTL','E:\\data\\cbow\\HTL-unbalanced-not-linked.csv',1110),
            ('MOV','E:\\data\\cbow\\MOV-unbalanced-not-linked.csv',2335)
            #,
#            ('PROD','E:\\data\\cbow\\PROD-unbalanced-not-linked.csv',234),
##            ('RES','E:\\data\\cbow\\RES-unbalanced-not-linked.csv',539),
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

 
    # CNN
    # ===============================================================================================
    print ("Method = CNN for Arabic Sentiment Analysis'")
    model_variation = 'CNN-non-static'
    np.random.seed(0)
    nb_filter = embeddings_dim
    
    main_input = Input(shape=(max_sent_len,))
    embedding = Embedding(max_features, embeddings_dim, input_length=max_sent_len, mask_zero=False, weights=[embedding_weights] )(main_input)
    Drop1 = Dropout(dropout_prob[0])(embedding)
    i=0
    conv_name=["" for x in range(len(filter_sizes))]
    pool_name=["" for x in range(len(filter_sizes))]
    flat_name=["" for x in range(len(filter_sizes))]
    for n_gram in filter_sizes:
        conv_name[i] = str('conv_' + str(n_gram))
        conv_name[i] = Convolution1D(nb_filter=nb_filter, filter_length=n_gram, border_mode='valid', activation='relu', subsample_length=1, input_dim=embeddings_dim, input_length=max_sent_len)(Drop1)
        pool_name[i] = str('maxpool_' + str(n_gram))
        pool_name[i] = MaxPooling1D(pool_length=max_sent_len - n_gram + 1)(conv_name[i])
        flat_name[i] = str('flat_' + str(n_gram))
        flat_name[i] = Flatten()(pool_name[i])
        i+=1
    merged = merge([flat_name[0], flat_name[1], flat_name[2]], mode='concat')    
    droput_final = Dropout(dropout_prob[1])(merged)
    Dense_final = Dense(1, input_dim=nb_filter * len(filter_sizes))(droput_final)
    Out = Dense(1, activation = 'sigmoid')(Dense_final)
    model = Model(inputs=main_input, outputs=Out)
    
    
    # Print model summary
    # ==================================================
    print(model.summary())
    
    #Visualize the model in a graph
    #1st method
    #from IPython.display import SVG
    #from keras.utils.vis_utils import model_to_dot
    #from keras.utils import vis_utils
    #SVG(model_to_dot(model).create(prog='dot', format='svg'))
    #2nd method
    plot_model(model, to_file='E:/model_plot.png', show_shapes=True, show_layer_names=True)
    
    # model compilation
    # ==================================================
    if num_classes == 2: model.compile(loss='binary_crossentropy', optimizer='Adagrad', metrics=['accuracy']) 
    else: model.compile(loss='categorical_crossentropy', optimizer='Adagrad', metrics=['accuracy'])
    
    
    #  model early_stopping and checkpointer
    # ==================================================
    Round='round-1'
    Vectors = 'Mine-vec'
    Linked = 'Not-Linked'
    early_stopping = EarlyStopping(patience=20, verbose=1)
    checkpointer = ModelCheckpoint(filepath= 'E:\\data\\cbow\\Weights\\'+file_name+'_'+Round+'_'+Linked+'_'+model_variation+'_'+Vectors+'_weights_Ar_best.hdf5', verbose=1, save_best_only=False)
      
    #  model history
    # ==================================================
    #hist = model.fit(train_sequences, train_labels, batch_size=32, nb_epoch=30, verbose=2, callbacks=[early_stopping, checkpointer])
    #without chepointer
    #hist = model.fit(x=train_sequences, y=train_labels, batch_size=32, nb_epoch=30, verbose=2, callbacks= [early_stopping])
    #without early stoping
#    hist = model.fit(x=train_sequences, y=train_labels, batch_size=32, epochs=30, verbose=2)
    
    with open('E:\\data\\Results.txt', 'a') as the_file:
        the_file.write(dataset_name)
        the_file.write('\n---------------------------------------\n')
        the_file.write('\nStart_Time\n')
        the_file.write(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
        the_file.write('\n')

    ##fit with writing to a file
    pd.DataFrame(model.fit(x=train_sequences, y=train_labels, batch_size=32, epochs=2, verbose=2).history).to_csv("E:\\data\\history_"+dataset_name+".csv")

    with open('E:\\data\\Results.txt', 'a') as the_file:
        the_file.write('\nEnd_Time\n')
        the_file.write(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
        the_file.write('\n')
    
    # Evaluate model keras
    # ==================================================
    print("Evaluate...")
    score, acc  = model.evaluate(test_sequences, test_labels,batch_size=32)
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
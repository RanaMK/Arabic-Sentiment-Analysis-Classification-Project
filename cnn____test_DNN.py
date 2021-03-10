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
#   
print ("")
print ("Reading pre-trained word embeddings...")
embeddings = dict( )
#embeddings = Word2Vec.load_word2vec_format("E:\data\cbow\(CBOW58)-ASA-3B-CBOW-window5-3iter-d300-vecotrs.bin", binary=True,encoding='utf8', unicode_errors='ignore')
embeddings = gensim.models.KeyedVectors.load_word2vec_format("E:\data\cbow\(CBOW58)-ASA-3B-CBOW-window5-3iter-d300-vecotrs.bin", binary=True,encoding='utf8', unicode_errors='ignore')

#ASTD
file_name = 'E:\\data\\cbow\\ASTD-unbalanced-not-linked.csv'
max_sent_len = 40
#ATT
file_name = 'E:\\data\\cbow\\ATT-unbalanced-not-linked.csv'
max_sent_len = 568
#HTL
file_name = 'E:\\data\\cbow\\HTL-unbalanced-not-linked.csv'
max_sent_len = 1110
##MOV
file_name = 'E:\\data\\cbow\\MOV-unbalanced-not-linked.csv'
max_sent_len = 2335
#max_sent_len = 1000
##PROD
file_name = 'E:\\data\\cbow\\PROD-unbalanced-not-linked.csv'
max_sent_len = 234
##RES
file_name = 'E:\\data\\cbow\\RES-unbalanced-not-linked.csv'
max_sent_len = 539
##LABR
file_name = 'E:\\data\\cbow\\LABR-unbalanced-not-linked-800.csv'
max_sent_len = 882




# maximum length of a sentence
print ("max_sent_len = " ,max_sent_len)


# Reading csv data
# ==================================================
#
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
#Encode labels with value between 0 and n_class-1
le.fit( train_labels + test_labels )
train_labels = le.transform( train_labels )
test_labels = le.transform( test_labels )
print ("Classes : " + repr( le.classes_ ))

 
#DNN
input_dim = len(train_texts)
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
lay1 = model.add(Dense(output_dim = 8, init = 'uniform', activation = 'relu', 
                     input_dim = input_dim))  
## Adding the second hidden layer
lay2 = model.add(Dense(output_dim = 6, init = 'uniform', activation = 'relu'))
# Adding the output layer
lay3 = model.add(Dense(output_dim = num_classes, init = 'uniform', activation = 'softmax'))

outputs    = [layer.output for layer in model.layers] 
inputs    = [layer.input for layer in model.layers]

# Compiling the ANN
model.compile(optimizer = 'adam', loss = 'categorical_crossentropy', 
                   metrics = ['accuracy'])
# Fitting the ANN to the Training set
model.fit(train_texts.toarray(), train_labels, batch_size = 128, nb_epoch = 50, verbose=1)
#######Classifer Evaluation ######
score = model.evaluate(test_texts.toarray(), test_labels, batch_size=64, verbose=1)
accuracy = score[1]

# Print model summary
# ==================================================
# 
print(model.summary())

import datetime
datetime.datetime.now()
print(datetime.datetime.now())

from time import gmtime, strftime
strftime("%Y-%m-%d %H:%M:%S", gmtime())


loss_history = hist.history
numpy_loss_history = np.array(loss_history)
np.savetxt("loss_history.txt", numpy_loss_history, delimiter=",")

with open('E:\\data\\Results.txt', 'a') as the_file:
    the_file.write(strftime("%Y-%m-%d %H:%M:%S", gmtime()))
    the_file.write('\n')
    the_file.write(model.summary())
    the_file.write('\n')


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
# 
#if num_classes == 2: model.compile(loss='binary_crossentropy', optimizer='Adagrad') 
#else: model.compile(loss='categorical_crossentropy', optimizer='Adagrad')
if num_classes == 2: model.compile(loss='binary_crossentropy', optimizer='Adagrad', metrics=['accuracy']) 
else: model.compile(loss='categorical_crossentropy', optimizer='Adagrad', metrics=['accuracy'])


#  model early_stopping and checkpointer
# ==================================================
# 
Round='round-1'
Vectors = 'Mine-vec'
Linked = 'Not-Linked'
early_stopping = EarlyStopping(patience=20, verbose=1)
#Stop training when a monitored quantity has stopped improving.  
#patience: number of epochs with no improvement after which training will be stopped.
checkpointer = ModelCheckpoint(filepath= 'E:\\data\\cbow\\Weights\\'+'Mov'+'_'+Round+'_'+Linked+'_'+model_variation+'_'+Vectors+'_weights_Ar_best.hdf5', verbose=1, save_best_only=False)
#Save the model after every epoch.
checkpointer = ModelCheckpoint(filepath= 'E:\\data\\cbow\\Weights\\checkpoint_weights_Ar_best.hdf5', verbose=1, save_best_only=False)
  
#  model history
# ==================================================
# 
#  hist = model.fit({'input': train_sequences, 'output': train_labels}, batch_size=32, nb_epoch=30, verbose=2, show_accuracy=True,callbacks=[early_stopping, checkpointer])
#hist = model.fit(train_sequences, train_labels, batch_size=32, epochs=30, verbose=2, callbacks=[early_stopping, checkpointer])
hist = model.fit(train_sequences, train_labels, batch_size=32, epochs=5, verbose=2, validation_data=(test_sequences, test_labels), callbacks=[early_stopping, checkpointer])
#without chepointer
hist = model.fit(x=train_sequences, y=train_labels, batch_size=32, epochs=30, verbose=2, callbacks= [early_stopping])
hist = model.fit(x=train_sequences, y=train_labels, batch_size=32, epochs=5, verbose=2, callbacks= [early_stopping])
#without early stoping
hist = model.fit(x=train_sequences, y=train_labels, batch_size=32, epochs=30, verbose=2)
hist = model.fit(x=train_sequences, y=train_labels, batch_size=32, epochs=2, verbose=2)
#with only checkpoint
hist = model.fit(x=train_sequences, y=train_labels, batch_size=32, epochs=2, verbose=2, callbacks= [checkpointer])


#1 -- fakes
#hist_log = hist.history
#dataset_name='MOV'
#numpy_loss_history = np.array(hist_log).tofile('E:\\data\\Results'+dataset_name+'.txt',sep='\n')
##np.savetxt('E:\\data\\Results.txt', numpy_loss_history, delimiter="\n")

#2--fakes
#import pickle
#with open('E:\\data\\Results.txt', 'a') as file_pi:
#        pickle.dump(hist.history, file_pi)

#3
pd.DataFrame(model.fit(x=train_sequences, y=train_labels, batch_size=32, epochs=2, verbose=2).history).to_csv("E:\\data\\historyMOV.csv")

#4
#from keras.callbacks import CSVLogger
#keras.callbacks.CSVLogger('E:\\data\\Results.txt', separator='\n', append=True)
#csv_logger = CSVLogger('E:\\data\\Results.csv')
#hist = model.fit(x=train_sequences, y=train_labels, batch_size=32, epochs=1, verbose=2, callbacks=[csv_logger])

#5
#import sys
#oldStdout = sys.stdout
#file = open('logFile', 'w')
#sys.stdout = file
#model.fit(Xtrain, Ytrain)
#sys.stdout = oldStdout

#6
with open('E:\\data\\Results.txt', 'a') as the_file:
    the_file.write(strftime("%Y-%m-%d %H:%M:%S", gmtime()))
    the_file.write('\n')
    the_file.write(hist_log.)
    the_file.write('\n')




# summarize history for accuracy
plt.plot(hist.history['acc'])
plt.plot(hist.history['loss'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()


# Evaluate model keras
# ==================================================
#
print("Evaluate...")
#  score, acc = model.evaluate({'input': test_sequences, 'output': test_labels},show_accuracy=True,batch_size=32)
score, acc  = model.evaluate(test_sequences, test_labels,batch_size=32)
print('Test score:', score)
print('Test accuracy:', acc)


# Evaluate model sklearn
# ==================================================
#
#results = np.array(model.predict({'input': test_sequences}, batch_size=32)['output'])
results = np.array(model.predict(test_sequences, batch_size=32))
if num_classes != 2: results = results.argmax(axis=-1)
else: results = (results > 0.5).astype('int32')
print ("Accuracy-sklearn = " + repr( sklearn.metrics.accuracy_score( test_labels , results )  ))
print ('RMSE',np.sqrt(((results - test_labels) ** 2).mean(axis=0)).mean() ) # Printing RMSE
print (sklearn.metrics.classification_report( test_labels , results ))




###Here is how determinate a number of shapes of you Keras model (var model), and each shape unit occupies 4 bytes in memory:
#shapes_count = int(numpy.sum([numpy.prod(numpy.array([s if isinstance(s, int) else 1 for s in l.output_shape])) for l in model.layers]))
#memory = shapes_count * 4
####And here is how determinate a number of params of you Keras model (var model):
#from keras import backend as K
#trainable_count = int(numpy.sum([K.count_params(p) for p in set(model.trainable_weights)]))
#non_trainable_count = int(numpy.sum([K.count_params(p) for p in set(model.non_trainable_weights)]))
#####the memory required for various batch sizes?
# of parameters * 4 bits + size of batch * some constant.
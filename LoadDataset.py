# -*- coding: utf-8 -*-
"""
Created on Sat Oct  7 09:20:23 2017

@author: Rana Mahmoud
"""
import codecs
import numpy as np
import pandas as pd
import re

class LoadDataset:
    def __init__(self):
        self.file_name = "E:\\data\\Tweets.txt"
        
    def Load_data(self): 
        ####################### READ DATASET ####################
        reviews = codecs.open(self.file_name, 'r', 'utf-8').readlines()
        
        # remove comment lines and newlines
        reviews = [re.sub(("^\s+"), "", r) for r in reviews]
        reviews = [re.sub(("\s+$"), "", r) for r in reviews]
        # parse
        rating = list()
        body = list()
    
        for review in reviews:
            # split by <tab>
            parts = review.split(u"\t")
            # body is first part and rating is last part
            body.append(parts[0])
            rating.append(parts[1])
        
        return(body, rating)

#########################################################

    def split_train_validation_test(self, rating, percent_test, percent_valid):
        
        ## Splits the data-set into a training/validation/test sets  ##
        np.random.seed(1234)
        
        rating = np.array(rating)
        # length
        num_reviews = len(rating)
        review_ids = np.arange(0, num_reviews)
        
        review_ids_pos=review_ids[rating=='POS']
        review_ids_neg=review_ids[rating=='NEG']
        review_ids_neutral=review_ids[rating=='NEUTRAL']
        review_ids_obj=review_ids[rating=='OBJ']
        
        np.random.shuffle(review_ids_pos)
        np.random.shuffle(review_ids_neg)
        np.random.shuffle(review_ids_neutral)
        np.random.shuffle(review_ids_obj)
        
        percent_test = 0.2
        percent_valid = 0.2 
        
        ## Unbalanced dataset ###
        print(60 * "-")
        print("Loading data:", "un-balanced")
        ntest_pos = int(np.floor(len(review_ids_pos)*percent_test))
        ntest_neg = int(np.floor(len(review_ids_neg)*percent_test))
        ntest_neutral = int(np.floor(len(review_ids_neutral)*percent_test))
        ntest_obj = int(np.floor(len(review_ids_obj)*percent_test))
        
        nvalid_pos = int(np.floor(len(review_ids_pos)*percent_valid))
        nvalid_neg = int(np.floor(len(review_ids_neg)*percent_valid))
        nvalid_neutral = int(np.floor(len(review_ids_neutral)*percent_valid))
        nvalid_obj = int(np.floor(len(review_ids_obj)*percent_valid))
        
        test_ids_unbalanced = np.concatenate([review_ids_pos[0:ntest_pos] \
                               ,review_ids_neg[0:ntest_neg]\
                               ,review_ids_obj[0:ntest_obj]\
                               ,review_ids_neutral[0:ntest_neutral]])
        
        validation_ids_unbalanced = np.concatenate([review_ids_pos[ntest_pos:ntest_pos+nvalid_pos] \
                               ,review_ids_neg[ntest_neg:ntest_neg+nvalid_neg]\
                               ,review_ids_obj[ntest_obj:ntest_obj+nvalid_obj]\
                               ,review_ids_neutral[ntest_neutral:ntest_neutral+nvalid_neutral]])
        
        train_ids_unbalanced = np.concatenate([review_ids_pos[ntest_pos+nvalid_pos:] \
                               ,review_ids_neg[ntest_neg+nvalid_neg:]\
                               ,review_ids_obj[ntest_obj+nvalid_obj:]\
                               ,review_ids_neutral[ntest_neutral+nvalid_neutral:]])
        
        ## balanced dataset ###
        print(60 * "-")
        print("Loading data:", "balanced")
        sizes=[len(review_ids_pos),len(review_ids_neg),len(review_ids_neutral),len(review_ids_obj)]
        min_size = min(sizes)
        
        ntest = int(np.floor(min_size * percent_test))
        nvalid = int(np.floor(min_size * percent_valid))
        
        test_ids_balanced = np.concatenate([review_ids_pos[0:ntest] \
                               ,review_ids_neg[0:ntest]\
                               ,review_ids_obj[0:ntest]\
                               ,review_ids_neutral[0:ntest]])
        
        validation_ids_balanced = np.concatenate([review_ids_pos[ntest:ntest+nvalid] \
                               ,review_ids_neg[ntest:ntest+nvalid]\
                               ,review_ids_obj[ntest:ntest+nvalid]\
                               ,review_ids_neutral[ntest:ntest+nvalid]])
        
        train_ids_balanced = np.concatenate([review_ids_pos[ntest+nvalid:min_size] \
                               ,review_ids_neg[ntest+nvalid:min_size]\
                               ,review_ids_obj[ntest+nvalid:min_size]\
                               ,review_ids_neutral[ntest+nvalid:min_size]])
    
        return(test_ids_unbalanced, validation_ids_unbalanced, train_ids_unbalanced,
               test_ids_balanced, validation_ids_balanced, train_ids_balanced)
##############################################################################

##### get balanced train, test and validation sets ####
    def get_train_test_validation_balanced(self):

        balanced_train_y = [rating[i] for i in train_ids_balanced]
        balanced_test_y = [rating[i] for i in test_ids_balanced]
        balanced_valid_y = [rating[i] for i in validation_ids_balanced]

        balanced_train_x = [body[i] for i in train_ids_balanced]
        balanced_test_x = [body[i] for i in test_ids_balanced]
        balanced_valid_x = [body[i] for i in validation_ids_balanced]

#        return (np.array(balanced_train_x), np.array(balanced_train_y), np.array(balanced_test_x), np.array(balanced_test_y), np.array(balanced_valid_x), np.array(balanced_valid_y))
        return (balanced_train_x, balanced_train_y, balanced_test_x, balanced_test_y, balanced_valid_x, balanced_valid_y)

############################################################################

##### get unbalanced train, test and validation sets ####
    def get_train_test_validation_unbalanced(self):

        unbalanced_train_y = [rating[i] for i in train_ids_unbalanced]
        unbalanced_test_y = [rating[i] for i in test_ids_unbalanced]
        unbalanced_valid_y = [rating[i] for i in validation_ids_unbalanced]

        unbalanced_train_x = [body[i] for i in train_ids_unbalanced]
        unbalanced_test_x = [body[i] for i in test_ids_unbalanced]
        unbalanced_valid_x = [body[i] for i in validation_ids_unbalanced]

    #    return (np.array(unbalanced_train_x), np.array(unbalanced_train_y), np.array(unbalanced_test_x), np.array(unbalanced_test_y), np.array(unbalanced_valid_x), np.array(unbalanced_valid_y))
        return (unbalanced_train_x, unbalanced_train_y, unbalanced_test_x, unbalanced_test_y, unbalanced_valid_x, unbalanced_valid_y)
#######################################################################

LoadDataset=LoadDataset()
(body,rating)=LoadDataset.Load_data()
(test_ids_unbalanced, validation_ids_unbalanced, train_ids_unbalanced,
               test_ids_balanced, validation_ids_balanced, train_ids_balanced) = LoadDataset.split_train_validation_test(rating, 0.2, 0.2)
(balanced_train_x, balanced_train_y, balanced_test_x, balanced_test_y, balanced_valid_x, balanced_valid_y) = LoadDataset.get_train_test_validation_balanced()
(unbalanced_train_x, unbalanced_train_y, unbalanced_test_x, unbalanced_test_y, unbalanced_valid_x, unbalanced_valid_y) = LoadDataset.get_train_test_validation_unbalanced()





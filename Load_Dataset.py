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
        ####################### READ DATASET ####################
        self.file_name = "E:\\data\\Tweets.txt"
        reviews = codecs.open(file_name, 'r', 'utf-8').readlines()
        
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
        sizes=l=[len(review_ids_pos),len(review_ids_neg),len(review_ids_neutral),len(review_ids_obj)]
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


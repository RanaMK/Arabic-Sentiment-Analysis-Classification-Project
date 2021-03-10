# -*- coding: utf-8 -*-
"""
Created on Sat Oct  7 09:20:23 2017

@author: Rana Mahmoud
"""
import codecs
import numpy as np
import pandas as pd
import re
import nltk
from nltk.stem.isri import ISRIStemmer
import itertools
import pyarabic.arabrepr
from tashaphyne.stemming import ArabicLightStemmer
from pyarabic.named import *


class LoadDataset_General:
    def __init__(self):
        self.datasets = {'ASTD', 'LABR', 'BNN', 'SYR', 'HTL', 'MOV', 'ATT', 'PROD', 'RES', 'RES1', 'RES2'}
        
    def Load_Data(self, datasetName):
        ####################### READ DATASET ####################        
        body = list()
        rating=list()
        if datasetName == 'ASTD':
            file_name = "E:\\data\\Tweets.txt"    
            reviews = codecs.open(file_name, 'r', 'utf-8').readlines()
            reviews = [re.sub(("^\s+"), "", r) for r in reviews]
            reviews = [re.sub(("\s+$"), "", r) for r in reviews]
#            reviews = pd.DataFrame.drop_duplicates(reviews)
            for review in reviews:
                # split by <tab>
                parts = review.split(u"\t")
                # body is first part and rating is last part
                body.append(parts[0])
                rating.append(parts[1])
                        
        if datasetName == 'LABR':
            file_name = "E:\\data\\LABR\\reviews.tsv"    
            reviews = codecs.open(file_name, 'r', 'utf-8').readlines()
            reviews = [re.sub(("^\s+"), "", r) for r in reviews]
            reviews = [re.sub(("\s+$"), "", r) for r in reviews]
#            reviews = pd.DataFrame.drop_duplicates(reviews)
            for review in reviews:
                # split by <tab>
                parts = review.split(u"\t")
                # body is first part and rating is last part
                body.append(parts[4])
                rating.append(parts[0])
                        
        if datasetName == 'BBN':
            file_name = "E:\\data\\BBN\\bbn.txt"    
            reviews = codecs.open(file_name, 'r', 'utf-8').readlines()
            reviews = [re.sub(("^\s+"), "", r) for r in reviews]
            reviews = [re.sub(("\s+$"), "", r) for r in reviews]
#                reviews = pd.DataFrame.drop_duplicates(reviews)
            for review in reviews:
                # split by <tab>
                parts = review.split(u"\t")
                # body is first part and rating is last part
                body.append(parts[0])
                rating.append(parts[1])
            
        if datasetName == 'SYR':
            file_name = "E:\\data\\Syrian_Tweets\\Syrian_tweets.txt"    
            reviews = codecs.open(file_name, 'r', 'utf-8').readlines()
            reviews = [re.sub(("^\s+"), "", r) for r in reviews]
            reviews = [re.sub(("\s+$"), "", r) for r in reviews]
#            reviews = pd.DataFrame.drop_duplicates(reviews)
            for review in reviews:
                # split by <tab>
                parts = review.split(u"\t")
                # body is first part and rating is last part
                body.append(parts[1])
                rating.append(parts[2])
            body = body [1:]
            rating = rating[1:]
            
        if datasetName == 'HTL':
            file_name = "E:\\data\\Large_SA\\HTL.csv"    
            reviews = pd.read_csv(file_name)
            reviews = pd.DataFrame.drop_duplicates(reviews)
            body = reviews.iloc[:,0].values.tolist()
            rating = reviews.iloc[:,1].values.tolist()
            
        if datasetName == 'MOV':
            file_name = "E:\\data\\Large_SA\\MOV.csv"    
            reviews = pd.read_csv(file_name)
            reviews = pd.DataFrame.drop_duplicates(reviews)
            body = reviews.iloc[:,0].values.tolist()
            rating = reviews.iloc[:,1].values.tolist()
            
        if datasetName == 'ATT':
            file_name = "E:\\data\\Large_SA\\ATT.csv"    
            reviews = pd.read_csv(file_name)
            reviews = pd.DataFrame.drop_duplicates(reviews)
            body = reviews.iloc[:,1].values.tolist()
            rating = reviews.iloc[:,2].values.tolist()
            
        if datasetName == 'PROD':
            file_name = "E:\\data\\Large_SA\\PROD.csv"    
            reviews = pd.read_csv(file_name)
            reviews = pd.DataFrame.drop_duplicates(reviews)
            body = reviews.iloc[:,0].values.tolist()
            rating = reviews.iloc[:,1].values.tolist()
            
        if datasetName == 'RES':
            file_name = "E:\\data\\Large_SA\\RES.csv"    
            reviews = pd.read_csv(file_name)
            reviews = pd.DataFrame.drop_duplicates(reviews)
            body = reviews.iloc[:,0].values.tolist()
            rating = reviews.iloc[:,1].values.tolist()
            
        if datasetName == 'RES1':
            file_name = "E:\\data\\Large_SA\\RES1.csv"    
            reviews = pd.read_csv(file_name)
            reviews = pd.DataFrame.drop_duplicates(reviews)
            body = reviews.iloc[:,1].values.tolist()
            rating = reviews.iloc[:,0].values.tolist()
            
        if datasetName == 'RES2':
            file_name = "E:\\data\\Large_SA\\RES2.csv"    
            reviews = pd.read_csv(file_name)
            reviews = pd.DataFrame.drop_duplicates(reviews)
            body = reviews.iloc[:,0].values.tolist()
            rating = reviews.iloc[:,1].values.tolist()   
            
        if datasetName == 'SemEval':
            file_name = "E:\\data\\SemEval\\SemEval2017-task4-train.subtask-A.arabic.txt"    
            reviews = codecs.open(file_name, 'r', 'utf-8').readlines()
            reviews = [re.sub(("^\s+"), "", r) for r in reviews]
            reviews = [re.sub(("\s+$"), "", r) for r in reviews]
            for review in reviews:
                # split by <tab>
                parts = review.split(u"\t")
                # body is last part and rating is 2nd part
                body.append(parts[2])
                rating.append(parts[1])
        if datasetName == 'SemEval_test':
            file_name = "E:\\data\\SemEval\\SemEval2017-task4-test.subtask-A.arabic.txt"    
            reviews = codecs.open(file_name, 'r', 'utf-8').readlines()
            reviews = [re.sub(("^\s+"), "", r) for r in reviews]
            reviews = [re.sub(("\s+$"), "", r) for r in reviews]
            for review in reviews:
                # split by <tab>
                parts = review.split(u"\t")
                # body is last part and rating is 2nd part
                body.append(parts[2])
                rating.append(parts[1])
                
        if datasetName == 'EG_NU':
            file_name = "E:\\Masters Thesis\\Final Datasets\\EG_NU_dataset\\NU_EG_Twitter_corpus_train.csv"    
            reviews = codecs.open(file_name, 'r', 'utf-8', ).readlines()
            reviews = [re.sub(("^\s+"), "", r) for r in reviews]
            reviews = [re.sub(("\s+$"), "", r) for r in reviews]
            #            reviews = pd.DataFrame.drop_duplicates(reviews)
            for review in reviews:
                # split by <tab>
                parts = review.split(u",")
                # body is first part and rating is last part
                body.append(parts[1])
                rating.append(parts[2])
            body = body [1:]
            rating = rating[1:]

        if datasetName == 'EG_NU_test':
            file_name = "E:\\Masters Thesis\\Final Datasets\\EG_NU_dataset\\NU_EG_Twitter_corpus_test.csv"
            reviews = codecs.open(file_name, 'r', 'utf-8', ).readlines()
            reviews = [re.sub(("^\s+"), "", r) for r in reviews]
            reviews = [re.sub(("\s+$"), "", r) for r in reviews]
            #            reviews = pd.DataFrame.drop_duplicates(reviews)
            for review in reviews:
                # split by <tab>
                parts = review.split(u",")
                # body is first part and rating is last part
                body.append(parts[1])
                rating.append(parts[2])
            body = body [1:]
            rating = rating[1:]


        
        for i in range(0, len(rating)):
            if type(rating[i]) == str:
                if rating[i].lower() == 'positive' or rating[i].upper() == 'POS':
                    rating[i] = 1
                elif rating[i].lower() == 'negative' or rating[i].upper() == 'NEG':
                    rating[i] = 2
                elif rating[i].lower() == 'neutral' or rating[i] == 'NEUTRAL' or rating[i] == 'neu':
                    rating[i] = 0
                else: rating[i] = 0
            else:
                if rating[i] == -1:
                    rating[i] = 2
                else:
                    rating[i] = int(rating[i])
        
        return body,rating

# Copied from the PyArabic package.
    def arabicrange(self):
        """return a list of arabic characteres .
        Return a list of characteres between \u060c to \u0652
        @return: list of arabic characteres.
        @rtype: unicode;
        """
        mylist = [];
        for i in range(0x0600, 0x00653):
            try :
                mylist.append(chr(i));
            except ValueError:
                pass;
        return mylist;

    # cleans a single review
    def clean_raw_review(self, body):
         # patterns to remove first
        pat = [\
            (u'http[s]?://[a-zA-Z0-9_\-./~\?=%&]+', u''),  # remove links
            (u'www[a-zA-Z0-9_\-?=%&/.~]+', u''),
#            u'\n+': u' ',                     # remove newlines
            (u'<br />', u' '),  # remove html line breaks
            (u'</?[^>]+>', u' '),  # remove html markup
#            u'http': u'',
            (u'[a-zA-Z]+\.org', u''),
            (u'[a-zA-Z]+\.com', u''),
            (u'://', u''),
            (u'&[^;]+;', u' '),
            (u':D', u':)'),
#            (u'[0-9/]+', u''),
            (u'[a-zA-Z.]+', u''),
#            u'[^0-9' + u''.join(self.arabicrange()) + \
#                u"!.,;:$%&*%'#(){}~`\[\]/\\\\\"" + \
#                u'\s^><\-_\u201D\u00AB=\u2026]+': u'',          # remove latin characters
#            (u'\s+', u' '),  # remove spaces
            (u'\.+', u'.'),  # multiple dots
            (u'[\u201C\u201D]', u'"'),  # “
            (u'[\u2665\u2764]', u''),  # heart symbol
            (u'[\u00BB\u00AB]', u'"'),
            (u'\u2013', u'-'),  # dash
            (u'\.', u' '),  # remove dots
            (u'\!', u' '),  # remove !
            (u'\.+', u' '),  # remove dots
            (u'\!+', u' '),  # remove !
            (u'\@',u''), #remove @
            (u'\_',u''), #remove underscore
        ]


#         patterns that disqualify a review
        remove_if_there = [\
            ( u'[^0-9' + u''.join(self.arabicrange()) + \
                u"!.,;:$%&*%'#(){}~`\[\]/\\\\\"" + \
                u'\s\^><\-_\u201D\u00AB=\u2026+|' + \
                u'\u0660-\u066D\u201C\u201D' + \
                u'\ufefb\ufef7\ufef5\ufef9]+', u''),  # non arabic characters
        ]
            
        # patterns that disqualify if empty after removing
        remove_if_empty_after = [\
            (u'[0-9a-zA-Z\-_]', u' '),  # alpha-numeric
            (u'[0-9' + u".,!;:$%&*%'#(){}~`\[\]/\\\\\"" + \
                u'\s\^><`\-=_+]+', u''),  # remove just punctuation
            (u'\s+', u' '),  # remove spaces
#            (u'\.+', u' '),  # remove dots
#            (u'\!+', u' '),  # remove !
        ]
            
        skip = False

        # if empty body, skip
        if body == u'': skip = True

        # do some subsitutions
        for k, v in pat:
            body = re.sub(k, v, body)

        # remove if exist
        for k, v in remove_if_there:
            if re.search(k, body):
                skip = True

        # remove if empty after replacing
        for k, v in remove_if_empty_after:
            temp = re.sub(k, v, body)
            if temp == u" " or temp == u"":
                skip = True

        # if empty string, skip
        if body == u'' or body == u' ':
            skip = True

        if not skip:
            return body
        else:
            return u""

############  NORMALIZATION ############
#Ref: https://maximromanov.github.io/2013/01-02.html
    def normalizeArabic(self, body):
        body = re.sub("[إأٱآا]", "ا", body)
        body = re.sub("ى", "ي", body)
        body = re.sub("ؤ", "ء", body)
        body = re.sub("ئ", "ء", body)
        return(body)

################ DENORMALIZATION ############
#    def deNormalize(text):
#        alifs           = '[إأٱآا]'
#        alifReg         = '[إأٱآا]'
#        # -------------------------------------
#        alifMaqsura     = '[يى]'
#        alifMaqsuraReg  = '[يى]'
#        # -------------------------------------
#        taMarbutas      = 'ة'
#        taMarbutasReg   = '[هة]'
#        # -------------------------------------
#        hamzas          = '[ؤئء]'
#        hamzasReg       = '[ؤئءوي]'
#        # Applying deNormalization
#        text = re.sub(alifs, alifReg, text)
#        text = re.sub(alifMaqsura, alifMaqsuraReg, text)
#        text = re.sub(taMarbutas, taMarbutasReg, text)
#        text = re.sub(hamzas, hamzasReg, text)
#        return text


############# Noise Removal ##########
#This function removes short vowels and other symbols (harakat) that interfere with computational manipulations with Arabic texts.
    def deNoise(self, body):
        noise = re.compile(""" ّ    | # Tashdid
                                 َ    | # Fatha
                                 ً    | # Tanwin Fath
                                 ُ    | # Damma
                                 ٌ    | # Tanwin Damm
                                 ِ    | # Kasra
                                 ٍ    | # Tanwin Kasr
                                 ْ    | # Sukun
                                 ـ     # Tatwil/Kashida
                             """, re.VERBOSE)
        body = re.sub(noise, '', body)
        return body

############## ELONGATION ###########
    def Elong_remove(self, body):
        word = body.split(u" ")
        word_final = list()
        for w in word:
            w = ''.join(c[0] for c in itertools.groupby(w))
            word_final.append(w)
        body = " ".join(word_final) 
        return body

############### STEMMING #############
    def Stem_word(self, body):
        st = ISRIStemmer()
        word = body.split(u" ")
        word_stem = list()
        for w in word:
            word_stem.append(st.stem(w))
        body = " ".join(word_stem) 
        return body
    
    def Light_Stem_word(self, body):
        ArListem = ArabicLightStemmer()
        word = body.split(u" ")
        word_stem = list()
        for w in word:
            w_stem = ArListem.light_stem(w)
            word_stem.append(ArListem.get_stem())
#             print (ArListem.get_stem())
             # extract root
#             print (ArListem.get_root())
        body = " ".join(word_stem) 
        return body
     
    def Get_root_word(self, body):
        ArListem = ArabicLightStemmer()
        word = body.split(u" ")
        word_stem = list()
        for w in word:
            w_stem = ArListem.light_stem(w)
            word_stem.append(ArListem.get_root())
        body = " ".join(word_stem) 
        return body
    
    def Named_Entity_Recognition(self, body):
        ner = pyarabic.named
        word = body.split(u" ")
        word_ner = list()
        for w in word:
            w_ner= ner.extract_named(w)
            if w_ner == []:
                word_ner.append(w) 
            else:
                word_ner.append('NER')  
        body = " ".join(word_ner) 
        return body
    
    def Remove_Stopwords(self, body):
        stopWords=pd.read_csv('E:\\data\\Lexicons\\stop_words.txt')
        word = body.split(u" ")
        word_final = list()
        for w in word:
            if w in stopWords:
                word_final.append(u"")
            else:
                word_final.append(w)
        body_new = " ".join(word_final) 
        return body_new
    
    def Emoticon_detection(self, body):
        posEmojis=pd.read_csv('E:\\data\\Lexicons\\pos_emojis.txt').iloc[:,0].tolist()
        negEmojis=pd.read_csv('E:\\data\\Lexicons\\neg_emojis.txt').iloc[:,0].tolist()
#        posEmojis_Count = 0
#        negEmojis_Count = 0
        word_emj = body.split(u" ")
        word_list_emoj = list()
        for w in word_emj:
            if w in posEmojis:
#                posEmojis_Count+=1
                word_list_emoj.append("ايموشنموجب")
            elif w in negEmojis:
#                negEmojis_Count+=1
                word_list_emoj.append("ايموشنسالب")
            else:
                word_list_emoj.append(w)     
        body_final =" ".join(word_list_emoj)
        return body_final
    

#########################################################
#
#    def split_train_validation_test(self, rating, percent_test, percent_valid):
#        
#        
#    
#        return(test_ids_unbalanced, validation_ids_unbalanced, train_ids_unbalanced,
#               test_ids_balanced, validation_ids_balanced, train_ids_balanced)
        
##############################################################################


##### get balanced train, test and validation sets ####
    def get_train_test_validation_balanced(self, body, rating):

        ## Splits the data-set into a training/validation/test sets  ##
        np.random.seed(1234)
        
        rating = np.array(rating)
        # length
        num_reviews = len(rating)
        review_ids = np.arange(0, num_reviews)
        
        review_ids_pos=review_ids[rating==1]
        review_ids_neg=review_ids[rating==-1]
        review_ids_neutral=review_ids[rating==0]
        
        np.random.shuffle(review_ids_pos)
        np.random.shuffle(review_ids_neg)
        np.random.shuffle(review_ids_neutral)
        
        percent_test = 0.2
        percent_valid = 0.2 
        
        ## balanced dataset ###
        print(60 * "-")
        print("Loading data:", "balanced")
        sizes=[len(review_ids_pos),len(review_ids_neg),len(review_ids_neutral)]
        min_size = min(sizes)
        
        ntest = int(np.floor(min_size * percent_test))
        nvalid = int(np.floor(min_size * percent_valid))
        
        test_ids_balanced = np.concatenate([review_ids_pos[0:ntest] \
                               ,review_ids_neg[0:ntest]\
                               ,review_ids_neutral[0:ntest]])
        
        validation_ids_balanced = np.concatenate([review_ids_pos[ntest:ntest+nvalid] \
                               ,review_ids_neg[ntest:ntest+nvalid]\
                               ,review_ids_neutral[ntest:ntest+nvalid]])
        
        train_ids_balanced = np.concatenate([review_ids_pos[ntest+nvalid:min_size] \
                               ,review_ids_neg[ntest+nvalid:min_size]\
                               ,review_ids_neutral[ntest+nvalid:min_size]])
                
        balanced_train_y = [rating[i] for i in train_ids_balanced]
        balanced_test_y = [rating[i] for i in test_ids_balanced]
        balanced_valid_y = [rating[i] for i in validation_ids_balanced]

        balanced_train_x = [body[i] for i in train_ids_balanced]
        balanced_test_x = [body[i] for i in test_ids_balanced]
        balanced_valid_x = [body[i] for i in validation_ids_balanced]

#        return (np.array(balanced_train_x), np.array(balanced_train_y), np.array(balanced_test_x), np.array(balanced_test_y), np.array(balanced_valid_x), np.array(balanced_valid_y))
        return (balanced_train_x, balanced_train_y, balanced_test_x, balanced_test_y, balanced_valid_x, balanced_valid_y)

##### get unbalanced train, test and validation sets ####
    def get_train_test_validation_unbalanced(self, body, rating, datasetName):

        ## Splits the data-set into a training/validation/test sets  ##
        np.random.seed(1234)
        
        rating = np.array(rating)
        # length
        num_reviews = len(rating)
        review_ids = np.arange(0, num_reviews)
        
        review_ids_pos=review_ids[rating==1]
        review_ids_neg=review_ids[rating==-1]
        review_ids_neutral=review_ids[rating==0]
        
        np.random.shuffle(review_ids_pos)
        np.random.shuffle(review_ids_neg)
        np.random.shuffle(review_ids_neutral)
        
        percent_test = 0.2
        percent_valid = 0.2 
        
        ## Unbalanced dataset ###
        print(60 * "-")
        print("Loading data:", "un-balanced")
        ntest_pos = int(np.floor(len(review_ids_pos)*percent_test))
        ntest_neg = int(np.floor(len(review_ids_neg)*percent_test))
        ntest_neutral = int(np.floor(len(review_ids_neutral)*percent_test))
                
        nvalid_pos = int(np.floor(len(review_ids_pos)*percent_valid))
        nvalid_neg = int(np.floor(len(review_ids_neg)*percent_valid))
        nvalid_neutral = int(np.floor(len(review_ids_neutral)*percent_valid))
        
        test_ids_unbalanced = np.concatenate([review_ids_pos[0:ntest_pos] \
                               ,review_ids_neg[0:ntest_neg]\
                               ,review_ids_neutral[0:ntest_neutral]])
        
        validation_ids_unbalanced = np.concatenate([review_ids_pos[ntest_pos:ntest_pos+nvalid_pos] \
                               ,review_ids_neg[ntest_neg:ntest_neg+nvalid_neg]\
                               ,review_ids_neutral[ntest_neutral:ntest_neutral+nvalid_neutral]])
        
        train_ids_unbalanced = np.concatenate([review_ids_pos[ntest_pos+nvalid_pos:] \
                               ,review_ids_neg[ntest_neg+nvalid_neg:]\
                               ,review_ids_neutral[ntest_neutral+nvalid_neutral:]])
                
        unbalanced_train_y = [rating[i] for i in train_ids_unbalanced]
        unbalanced_test_y = [rating[i] for i in test_ids_unbalanced]
        unbalanced_valid_y = [rating[i] for i in validation_ids_unbalanced]

        unbalanced_train_x = [body[i] for i in train_ids_unbalanced]
        unbalanced_test_x = [body[i] for i in test_ids_unbalanced]
        unbalanced_valid_x = [body[i] for i in validation_ids_unbalanced]

    #    return (np.array(unbalanced_train_x), np.array(unbalanced_train_y), np.array(unbalanced_test_x), np.array(unbalanced_test_y), np.array(unbalanced_valid_x), np.array(unbalanced_valid_y))
        return (unbalanced_train_x, unbalanced_train_y, unbalanced_test_x, unbalanced_test_y, unbalanced_valid_x, unbalanced_valid_y)
#######################################################################

    
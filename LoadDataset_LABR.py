# -*- coding: utf-8 -*-
"""
Created on Sat Oct  7 09:20:23 2017

@author: Rana Mahmoud
"""
import codecs
import numpy as np
import pandas as pd
import re

#reviews = codecs.open("E:\\data\\LABR\\reviews.tsv", 'r', 'utf-8').readlines()
#reviews = [re.sub(("^\s+"), "", r) for r in reviews]
#arts = reviews.split(u"\t")

class LoadDataset_LABR:
    def __init__(self):
        self.file_name = "E:\\data\\LABR\\reviews.tsv"
        
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
            # body is first last and rating is first part
            body.append(parts[4])
            rating.append(parts[0])
        
        return(body, rating)
    
    def arabicrange(self):
        """return a list of arabic characteres .
        Return a list of characteres between \u060c to \u0652
        @return: list of arabic characteres.
        @rtype: unicode;
        """
        mylist = [];
        for i in range(0x0600, 0x00653):
            try :
                mylist.append(unichr(i));
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
#            u'[a-zA-Z.]+': u'',
#            u'[^0-9' + u''.join(self.arabicrange()) + \
#                u"!.,;:$%&*%'#(){}~`\[\]/\\\\\"" + \
#                u'\s^><\-_\u201D\u00AB=\u2026]+': u'',          # remove latin characters
            (u'\s+', u' '),  # remove spaces
            (u'\.+', u'.'),  # multiple dots
            (u'[\u201C\u201D]', u'"'),  # “
            (u'[\u2665\u2764]', u''),  # heart symbol
            (u'[\u00BB\u00AB]', u'"'),
            (u'\u2013', u'-'),  # dash
        ]

        # patterns that disqualify a review
        remove_if_there = [\
            (u'[^0-9' + u''.join(self.arabicrange()) + \
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
        ]

        # remove again
        # patterns to remove
        pat2 = [\
#            u'[^0-9' + u''.join(self.arabicrange()) + \
#                u"!.,;:$%&*%'#(){}~`\[\]/\\\\\"" + \
#                u'\s^><\-_\u201D\u00AB=\u2026]+': u'',          # remove latin characters
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

        # do some more subsitutions
        if not skip:
            for k, v in pat2:
                body = re.sub(k, v, body)

        # if empty string, skip
        if body == u'' or body == u' ':
            skip = True

        if not skip:
            return body
        else:
            return u""
        
#########################################################

    def split_train_validation_test(self, rating, percent_test, percent_valid):
        
        # clean body
#        body = self.clean_raw_review(body)
#        if body == u"": skip = True
        ## Splits the data-set into a training/validation/test sets  ##
        np.random.seed(1234)
        
        rating = np.array(rating)
        # length
        num_reviews = len(rating)
        review_ids = np.arange(0, num_reviews)
        
        review_ids_RATING1=review_ids[rating=='1']
        review_ids_RATING2=review_ids[rating=='2']
        review_ids_RATING3=review_ids[rating=='3']
        review_ids_RATING4=review_ids[rating=='4']
        review_ids_RATING5=review_ids[rating=='5']
        
        np.random.shuffle(review_ids_RATING1)
        np.random.shuffle(review_ids_RATING2)
        np.random.shuffle(review_ids_RATING3)
        np.random.shuffle(review_ids_RATING4)
        np.random.shuffle(review_ids_RATING5)
        
        percent_test = 0.2
        percent_valid = 0.2 
        
        ## Unbalanced dataset ###
        print(60 * "-")
        print("Loading data:", "un-balanced")
        ntest_RATING1 = int(np.floor(len(review_ids_RATING1)*percent_test))
        ntest_RATING2 = int(np.floor(len(review_ids_RATING2)*percent_test))
        ntest_RATING3 = int(np.floor(len(review_ids_RATING3)*percent_test))
        ntest_RATING4 = int(np.floor(len(review_ids_RATING4)*percent_test))
        ntest_RATING5 = int(np.floor(len(review_ids_RATING5)*percent_test))
        
        nvalid_RATING1 = int(np.floor(len(review_ids_RATING1)*percent_valid))
        nvalid_RATING2 = int(np.floor(len(review_ids_RATING2)*percent_valid))
        nvalid_RATING3 = int(np.floor(len(review_ids_RATING3)*percent_valid))
        nvalid_RATING4 = int(np.floor(len(review_ids_RATING4)*percent_valid))
        nvalid_RATING5 = int(np.floor(len(review_ids_RATING5)*percent_valid))
        
        test_ids_unbalanced = np.concatenate([review_ids_RATING1[0:ntest_RATING1] \
                               ,review_ids_RATING1[0:ntest_RATING2]\
                               ,review_ids_RATING3[0:ntest_RATING3]\
                               ,review_ids_RATING4[0:ntest_RATING4]\
                               ,review_ids_RATING5[0:ntest_RATING5]])
        
        validation_ids_unbalanced = np.concatenate([review_ids_RATING1[ntest_RATING1:ntest_RATING1+nvalid_RATING1] \
                               ,review_ids_RATING2[ntest_RATING2:ntest_RATING2+nvalid_RATING2]\
                               ,review_ids_RATING3[ntest_RATING3:ntest_RATING3+nvalid_RATING3]\
                               ,review_ids_RATING4[ntest_RATING4:ntest_RATING4+nvalid_RATING4]\
                               ,review_ids_RATING5[ntest_RATING5:ntest_RATING5+nvalid_RATING5]])
        
        train_ids_unbalanced = np.concatenate([review_ids_RATING1[ntest_RATING1+nvalid_RATING1:] \
                               ,review_ids_RATING2[ntest_RATING2+nvalid_RATING2:]\
                               ,review_ids_RATING3[ntest_RATING3+nvalid_RATING3:]\
                               ,review_ids_RATING4[ntest_RATING4+nvalid_RATING4:]\
                               ,review_ids_RATING5[ntest_RATING5+nvalid_RATING5:]])
        
        ## balanced dataset ###
        print(60 * "-")
        print("Loading data:", "balanced")
        sizes=[len(review_ids_RATING1),len(review_ids_RATING2),len(review_ids_RATING3),len(review_ids_RATING4),len(review_ids_RATING5)]
        min_size = min(sizes)
        
        ntest = int(np.floor(min_size * percent_test))
        nvalid = int(np.floor(min_size * percent_valid))
        
        test_ids_balanced = np.concatenate([review_ids_RATING1[0:ntest] \
                               ,review_ids_RATING2[0:ntest]\
                               ,review_ids_RATING3[0:ntest]\
                               ,review_ids_RATING4[0:ntest]\
                               ,review_ids_RATING5[0:ntest]])
        
        validation_ids_balanced = np.concatenate([review_ids_RATING1[ntest:ntest+nvalid] \
                               ,review_ids_RATING2[ntest:ntest+nvalid]\
                               ,review_ids_RATING3[ntest:ntest+nvalid]\
                               ,review_ids_RATING4[ntest:ntest+nvalid]\
                               ,review_ids_RATING5[ntest:ntest+nvalid]])
        
        train_ids_balanced = np.concatenate([review_ids_RATING1[ntest+nvalid:min_size] \
                               ,review_ids_RATING2[ntest+nvalid:min_size]\
                               ,review_ids_RATING3[ntest+nvalid:min_size]\
                               ,review_ids_RATING4[ntest+nvalid:min_size]\
                               ,review_ids_RATING5[ntest+nvalid:min_size]])
    
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





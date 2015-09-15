#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd

from gensim.models import Word2Vec
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import scale
    
from TextUtility import TextUtility


def makeFeatureVec(words, model, num_features):
    featureVec = np.zeros((num_features,), dtype="float32")
    nwords = 0

    index2word_set = set(model.index2word)
    for word in words:
        if word in index2word_set:
            nwords = nwords + 1
            featureVec = np.add(featureVec, model[word])

    if nwords != 0:
        featureVec /= nwords 
    return featureVec


def getAvgFeatureVecs(texts, model, num_features):
    counter = 0

    textFeatureVecs = np.zeros((len(texts), num_features), dtype="float32")

    for text in texts:
        textFeatureVecs[counter] = makeFeatureVec(text, model, num_features)
        counter = counter + 1
    return textFeatureVecs
    

def getClean(data):
    clean_data = []
    for text in data["text"]:
        clean_data.append(TextUtility.text_to_wordlist(text, True))
    return clean_data   


def run_we(train, test, f_we, n_dim, clf=LogisticRegression(class_weight="auto")):
    try:
        model = Word2Vec.load_word2vec_format(f_we, binary=False)
    except:
        print "Error in loading word embeddings"
        exit(1)
    
    X_train = scale(getAvgFeatureVecs(getClean(train), model, n_dim))
    X_test = scale(getAvgFeatureVecs(getClean(test), model, n_dim))
    y_train = train['label']
    
    clf.fit(X_train, y_train)
    
    y_pred = clf.predict(X_test)
    try:
        y_prob = clf.predict_proba(X_test)
    except:
        # for svm with probability output
        clf.set_params(probability=True)
        y_prob_pos = clf.predict(X_test)
        y_prob_neg = np.ones(X_test.shape[0]) - y_prob_pos
        y_prob = np.column_stack((y_prob_neg, y_prob_pos))
        
    return y_pred, y_prob
#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np

from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer

from TextUtility import TextUtility
  

def run_tfidf(train, test, grams='123', n_dim=40000, clf=LogisticRegression(class_weight="auto")):
    clean_train = []
    for text in train['text']:
        clean_train.append(" ".join(TextUtility.text_to_wordlist(text)))

    clean_test = []
    for text in test['text']:
        clean_test.append(" ".join(TextUtility.text_to_wordlist(text)))
    
    ngram_range = (int(grams[0]), int(grams[-1])) 
    vectorizer = TfidfVectorizer(max_features=n_dim, ngram_range=ngram_range, sublinear_tf=True)
    
    X_train = vectorizer.fit_transform(clean_train)
    X_test = vectorizer.transform(clean_test)
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

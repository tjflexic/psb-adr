#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np

from concept_matching import run_cm
from maxent_tfidf import run_tfidf
from maxent_nblcr import run_nblcr
from maxent_we import run_we

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import f1_score, classification_report


def find_best_weights(train, test, clf, thresh=0.5):
    y_pred_cm = run_cm(train, test, '../data/ADR-lexicon.txt')
    _, y_prob_tfidf = run_tfidf(train, test, grams='123', n_dim=40000, clf=clf)
    _, y_prob_nblcr = run_nblcr(train, test, '../data/nblcr', grams='123', clf=clf)
    _, y_prob_we = run_we(train, test, '../data/w2v_150.txt', 150, clf=clf)
    
    alphas = np.float32(np.linspace(0, 1, 21))
    max_f1 = 0
    best_weights = [0, 0, 0]
    
    for alpha1 in alphas:
        for alpha2 in alphas:
            for alpha3 in alphas:
                if alpha1 + alpha2 + alpha3 > 1: continue
                              
                y_pred = []
                for i in xrange(len(y_pred_cm)):
                    val = alpha1*y_pred_cm[i] + alpha2*y_prob_tfidf[i,1] + alpha3*y_prob_nblcr[i,1] + (1-alpha1-alpha2-alpha3)*y_prob_we[i,1]
                    if val >= thresh: y_pred.append(1)
                    else: y_pred.append(0)
                f1 = f1_score(test['label'], y_pred)
                if f1 > max_f1:
                    best_weights = [alpha1, alpha2, alpha3]
                    max_f1 = f1
                    
    return best_weights, max_f1


def run_ensemble(train, test, weights, clf, thresh=0.5):
    y_pred_cm = run_cm(train, test, '../data/ADR-lexicon.txt')
    _, y_prob_tfidf = run_tfidf(train, test, grams='123', n_dim=40000, clf=clf)
    _, y_prob_nblcr = run_nblcr(train, test, '../data/nblcr', grams='123', clf=clf)
    _, y_prob_we = run_we(train, test, '../data/w2v_150.txt', 150, clf=clf)
    
    y_pred = []
    
    for i in xrange(len(y_pred_cm)):
        val = weights[0]*y_pred_cm[i] + weights[1]*y_prob_tfidf[i,1] + weights[2]*y_prob_nblcr[i,1] + (1-weights[0]-weights[1]-weights[2])*y_prob_we[i,1]
        if val >= thresh: y_pred.append(1)
        else: y_pred.append(0)
        
    return y_pred

if __name__ == '__main__':
    print "Predict for validation data..."
    train = pd.read_csv('../data/tweets-train.txt', names=['id','label','text'], sep='\t', quotechar='\t')
    dev = pd.read_csv('../data/tweets-dev.txt', names=['id','label','text'], sep='\t', quotechar='\t')
    
    clf = LogisticRegression(class_weight="auto")
    #clf = SVC(kernel='linear', class_weight="auto", random_state=0)
    #clf = KNeighborsClassifier(n_neighbors=25, weights='distance')
    #clf = DecisionTreeClassifier(criterion='gini', class_weight="auto", max_features='sqrt', random_state=0)
    
    weights, f1 = find_best_weights(train, dev, clf)
    print "Best weights:", weights
    print "Max f1-score:", f1
     
    
    print "\nPredict for test data..."
     
    test = pd.read_csv('../data/tweets-test.txt', names=['id','label','text'], sep='\t', quotechar='\t')
    y_pred_test = run_ensemble(train, test, weights, clf)
     
    output = pd.DataFrame(data={"id":test["id"], "label":y_pred_test})
    output.to_csv('../data/test_result.txt', sep='\t', header=False, index=False, quoting=3)
     
    print "Save results for test data."

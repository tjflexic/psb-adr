#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
from collections import Counter

from sklearn.datasets import load_svmlight_files
from sklearn.linear_model import LogisticRegression

    
from TextUtility import TextUtility

def tokenize(sentence, grams):
    words = TextUtility.text_to_wordlist(sentence)
    tokens = []
    for gram in grams:
        for i in range(len(words) - gram + 1):
            tokens += ["_*_".join(words[i:i+gram])]
    return tokens


def build_dict(data, grams):
    dic = Counter()
    for token_list in data:
        dic.update(token_list)
    return dic


def compute_ratio(poscounts, negcounts, alpha=1):
    alltokens = list(set(poscounts.keys() + negcounts.keys()))
    dic = dict((t, i) for i, t in enumerate(alltokens))
    d = len(dic)
       
    p, q = np.ones(d) * alpha , np.ones(d) * alpha
    for t in alltokens:
        p[dic[t]] += poscounts[t]
        q[dic[t]] += negcounts[t]
    p /= abs(p).sum()
    q /= abs(q).sum()
    r = np.log(p/q)
    return dic, r


def generate_svmlight_file(data, dic, r, grams, fn):
    output = []
    for _, row in data.iterrows():
        tokens = tokenize(row['text'], grams)
        indexes = []
        for t in tokens:
            try:
                indexes += [dic[t]]
            except KeyError:
                pass
        indexes = list(set(indexes))
        indexes.sort()
        if 'label' in row:
            line = [str(row['label'])]
        else:
            line = ['0']
        for i in indexes:
            line += ["%i:%f" % (i + 1, r[i])]
        output += [" ".join(line)]
    
    out = open(fn, "w")    
    out.writelines("\n".join(output))
    out.close()
    
    
def run_nblcr(train, test, outfn, grams='123', clf=LogisticRegression(class_weight="auto")):
    f_train = outfn + '-train.txt'
    f_test = outfn + '-test.txt'
    
    ngram = [int(i) for i in grams]
    ptrain = []
    ntrain = []
        
    for _, row in train.iterrows():
        if row['label'] == 1:
            ptrain.append(tokenize(row['text'], ngram))
        elif row['label'] == 0:
            ntrain.append(tokenize(row['text'], ngram))
        
    pos_counts = build_dict(ptrain, ngram)
    neg_counts = build_dict(ntrain, ngram)
        
    dic, r = compute_ratio(pos_counts, neg_counts)
        
    generate_svmlight_file(train, dic, r, ngram, f_train)
    generate_svmlight_file(test, dic, r, ngram, f_test)
    
    X_train, y_train, X_test, _ = load_svmlight_files((f_train, f_test))
    
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
    
    
    
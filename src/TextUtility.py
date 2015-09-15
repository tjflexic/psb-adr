#!/usr/bin/env python
# -*- coding: utf-8 -*-

import re
from random import shuffle
from bs4 import BeautifulSoup
from nltk.corpus import stopwords
from nltk.data import load


class TextUtility(object):
    negators = [line.strip() for line in file('../data/negator.txt','r')]
    tokenizer = load('tokenizers/punkt/english.pickle')

    @staticmethod
    def text_to_wordlist(text, remove_stopwords=False):
        # Function to convert a text to a sequence of words,
        # optionally removing stop words.  Returns a list of words.
        # Remove HTML
        new_text = BeautifulSoup(text).get_text()
        # Extract special negator like n't
        new_text = re.sub('n\'t', ' not', new_text)
        # Remove non-letters
        new_text = re.sub("[^a-zA-Z]"," ", new_text)
        # Convert words to lower case and split them
        words = new_text.lower().split()
        # Optionally remove stop words excluding negators (false by default)
        if remove_stopwords:
            stops = set(stopwords.words("english"))
            words = [w for w in words if not w in stops or w in TextUtility.negators]

        return words


    @staticmethod
    def text_to_sentences(text, remove_stopwords=False):
        # Function to split a text into parsed sentences. Returns a
        # list of sentences, where each sentence is a list of words
        #
        # Use the NLTK tokenizer to split the paragraph into sentences
        raw_sentences = TextUtility.tokenizer.tokenize(text.decode('utf8').strip())
        # Loop over each sentence
        sentences = []
        for raw_sentence in raw_sentences:
            if len(raw_sentence) > 0:
                # Call review_to_wordlist to get a list of words
                sentences.append(TextUtility.text_to_wordlist(raw_sentence, remove_stopwords))

        return sentences
    

    @staticmethod
    def split_train_test(df, test_portion=0.3):
        # Function to split a DataFrame for training and test
        # Create random list of indices
        N = len(df)
        l = range(N)
        shuffle(l)
        # Get splitting indicies
        trainLen = int(N*(1-test_portion))
        # Get training and test sets
        train = df.ix[l[:trainLen]]
        test = df.ix[l[trainLen:]]
 
        return train, test
    

    @staticmethod
    def split_train_dev_test(df, dev_portion=0.2, test_portion=0.2):
        # Function to split a DataFrame for training development, and test
        # Create random list of indices
        N = len(df)
        l = range(N)
        shuffle(l)
        # Get splitting indicies
        trainLen = int(N*(1-dev_portion-test_portion))
        traindevLen = int(N*(1-test_portion))
        # Get training, development and test sets
        train = df.ix[l[:trainLen]]
        dev = df.ix[l[trainLen:traindevLen]]
        test = df.ix[l[traindevLen:]]
 
        return train, dev, test
#!/usr/bin/env python
# -*- coding: utf-8 -*-

from tweetnlp import tweet_tagger

def clean_tweet(tweet_token, tweet_tag):
    token_list = tweet_token.split()
    tag_list = tweet_tag.split()
    
    new_token_list = []
    for token, tag in zip(token_list, tag_list):
        if tag in ['U', '@']:
            continue
        elif tag == '#':
            new_token_list.append(token[1:].lower())
        else:
            new_token_list.append(token.lower())
    
    return ' '.join(new_token_list)


def format_train_dev(src, target):
    tids = []
    labels = []
    tweets = []
    for line in file(src):
        tids.append(line.strip().split('\t')[0])
        labels.append(line.strip().split('\t')[2])
        tweets.append(line.strip().split('\t')[3])
    
    tokens, tags = tweet_tagger.runtagger_parse(tweets)
    
    out = open(target, 'w')
    for tid, label, token, tag in zip(tids, labels, tokens, tags):
        out.write(tid + '\t' + label + '\t' + clean_tweet(token, tag) + '\n')
    out.close()


def format_test(src, target):
    tids = []
    tweets = []
    for line in file(src):
        tids.append(line.strip().split('\t')[0])
        tweets.append(line.strip().split('\t')[1])
    
    tokens, tags = tweet_tagger.runtagger_parse(tweets)
    
    out = open(target, 'w')
    for tid, token, tag in zip(tids, tokens, tags):
        # assume each test tweet is labeled the non-ADR class (0)
        out.write(tid + '\t0\t' + clean_tweet(token, tag) + '\n')
    out.close()


if __name__ == "__main__":
    format_train_dev('../data/ADR-tweets-train.txt', '../data/tweets-train.txt')
    format_train_dev('../data/ADR-tweets-dev.txt', '../data/tweets-dev.txt')
    format_test('../data/ADR-tweets-test.txt', '../data/tweets-test.txt')
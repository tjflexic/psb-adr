#!/usr/bin/env python
# -*- coding: utf-8 -*-
import subprocess
import os

f = os.path.join(os.path.dirname(__file__), "ark-tweet-nlp-0.3.2.jar")

RUN_TAGGER_CMD = "java -XX:ParallelGCThreads=8 -Xmx500m -jar %s" % f


def runtagger_parse(tweets):
    tmp_file = open('tmp.txt', 'w')
    for tweet in tweets:
        tmp_file.write(tweet + '\n')
    tmp_file.close()
    
    try:
        subprocess.call(RUN_TAGGER_CMD + ' tmp.txt > tmp_tagger.txt', shell=True)
    except:
        exit(1)
    
    tokens = []
    tags = []
    for line in file('tmp_tagger.txt'):
        tokens.append(line.strip().split('\t')[0])
        tags.append(line.strip().split('\t')[1])
    
    os.remove('tmp.txt')
    os.remove('tmp_tagger.txt')
    
    return tokens, tags


if __name__ == "__main__":
    pass


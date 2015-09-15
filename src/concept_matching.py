#!/usr/bin/env python
# -*- coding: utf-8 -*-

def run_cm(train, test, f_lex):
    try:
        adr_lex = {}
        for line in file(f_lex):
            concept_id = line.strip().split('\t')[0].strip()
            concept_name = line.strip().split('\t')[1].strip()
            adr_lex[concept_name] = concept_id
    except:
        print "Error in loading ADR lexicon"
        exit(1)
        
    y_pred = []

    for _, row in test.iterrows():
        text = row['text']
        is_adr = False
        for k in adr_lex:
            if k in text:
                is_adr = True
                break
        y_pred.append(int(is_adr))
    
    return y_pred

    

# psb-adr
Binary classification of adverse drug reactions (PSB 2016 social media mining shared task)

This is the source code of my submission for the PSB 2016 social media mining shared task on binary classification of adverse drug reactions (ADRs) (http://psb.stanford.edu/workshop/wkshp-smm). The system is ranked 2nd among all 20 participating among all 20 participants with ADR F-score of 41.82%.

The system is a weighted average ensemble of four classifiers: 

<p>(1) a concept-matching classifier based on ADR lexicon (see src/concept_matching.py)</p>
<p>(2) a maximum entropy (ME) classifier with n-gram features and TFIDF weighting scheme (see src/maxent_tfidf.py)</p>
<p>(3) a ME classifier based on n-grams using naive Bayes (NB) log-count ratios as feature values (see src/maxent_nblcr.py)</p>
<p>(4) a ME classifier with word embedding features (see src/maxent_we.py)</p>

# How to run #

* The code requires numpy, pandas, sklearn, bs4, nltk, and gensim.
* Firstly, preprocess the tweets :
```
python preprocess.py
```
* After that, run the following command to find best ensembel weights on validation set and generate the results on test set:
```
python ensemble.py
```

# Remark #

* Because of Twitter's privacy policies, we cannot publish all twitter data here, just list one example (see data/ADR-tweets-train.txt, data/ADR-tweets-dev.txt, data/ADR-tweets-test.txt).
* Two files (data/ADR-lexicon.txt and data/w2v_150.txt) are reffered to http://diego.asu.edu/Publications/ADRMine.html.

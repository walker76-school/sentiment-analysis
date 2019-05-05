# Joseph Collins and Vincent Yin
# WordNet extensions
# 2019/02/18

import nltk
from nltk.corpus import wordnet as wn
from nltk.corpus.reader.wordnet import Synset
import math

def Word_lch_similarity(w1, w2):
    l1 = wn.synsets(w1)
    l2 = wn.synsets(w2)
    max_sim = 0.0
    res = []
    for synset1 in l1:
        for synset2 in l2:
            if synset1._pos == synset2._pos:
                val = synset1.lch_similarity(synset2)
                if val > max_sim:
                    max_sim = val
                    res = [max_sim, synset1, synset2]
    return res

def Word_path_similarity(w1, w2):
    l1 = wn.synsets(w1)
    l2 = wn.synsets(w2)
    max_sim = 0.0
    res = []
    for synset1 in l1:
        for synset2 in l2:
            if synset1._pos == synset2._pos:
                val = synset1.path_similarity(synset2)
                if val > max_sim:
                    max_sim = val
                    res = [max_sim, synset1, synset2]
    return res

def Word_wup_similarity(w1, w2):
    l1 = wn.synsets(w1)
    l2 = wn.synsets(w2)
    max_sim = 0.0
    res = [0.0,"dummy","dummy"]
    for synset1 in l1:
        for synset2 in l2:
            if synset1._pos == synset2._pos:
                val = synset1.wup_similarity(synset2)
                if val is not None and val > max_sim:
                    max_sim = val
                    res = [max_sim, synset1, synset2]
    return res


if __name__ == "__main__":
    print(wn.synsets('dog'))
    print(Word_lch_similarity('dog', 'cat'))
    #print(wn.synset(synset1).lch_similarity(wn.synset(synset2)))
    #print(wn.synsets('dog')[0]._name)
    #cat = wn.synsets('cat')[0]
    #dog = wn.synsets('dog')[0]
    #print(cat.lch_similarity(dog))

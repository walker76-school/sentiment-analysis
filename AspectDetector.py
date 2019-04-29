# laird
import nltk
from nltk.collocations import *
from CorpusReader_TFIDF import CorpusReader_TFIDF

class AspectDetector:
    TF_IDF_ASPECT_PERCENT = .05
    NUMBER_OF_TWO_GRAMS_TO_CONSIDER = 10

    def __init__(self, corpus):
        self.corpus = corpus
        self.tf_idf_Model = CorpusReader_TFIDF(corpus)

    def run(self):
        # get all of the tf_idf values
        vectors = self.tf_idf_Model.tf_idf()
        sumVect = [int] * len(vectors[0])
        # now find the best words for all of them
        for vector in vectors:
            count = 0
            for value in vector:
                sumVect[count] += value
                count += 1
        # associate each word in the corpus with its average tf_idf value
        averageVect = dict(zip(self.tf_idf_Model.words(), sumVect))
        # we now have an orderering of all the words in the corpus
        biggestWords = list(sorted(averageVect.keys(), key=averageVect.get, reverse=True))
        # now get the most common words
        # these will be the preliminaty aspects thwich will be farther narrowed down
        # right now will just take the top 5% although this can be narrowed down
        potentialAspects = biggestWords[int(AspectDetector.TF_IDF_ASPECT_PERCENT * len(self.tf_idf_Model.words()))]

        # going to now consider collocations of the corpus
        bigram_measures = nltk.collocations.BigramAssocMeasures()
        finder = BigramCollocationFinder.from_words(self.corpus.words())
        # find the most common bigrams of the corpus
        good_Grams = finder.nbest(bigram_measures.pmi, AspectDetector.NUMBER_OF_TWO_GRAMS_TO_CONSIDER)

        # now we have both the "best" 2 grams, and the best "unigrams" (from the tfidf model)
        # what I am thinking is that now we see if any of the unigrams are mutually contained in a two-grams
        # ie if "battery" and "life" are both popular unigrams, and "battery life" is a common
        # bigram we cut out the term that appears later in the twogram
        # maybe should we also consider grams higher than 2 (probably big performance hit?!?)


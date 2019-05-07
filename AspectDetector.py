# laird
import nltk
from nltk.collocations import *
from CorpusReader_TFIDF import CorpusReader_TFIDF
from nltk.tokenize import sent_tokenize, word_tokenize
from collections import defaultdict
import pickle
from collections import Counter
import string


class AspectDetector:
    TF_IDF_ASPECT_PERCENT = .01
    NUMBER_OF_TWO_GRAMS_TO_CONSIDER = 10

    def __init__(self, trainingCorpus, reviewCorpus):
        print("Creating AspectDetector ... ")
        self.trainingCorpus = trainingCorpus
        self.reviewCorpus = reviewCorpus
        self.potentialAspects = None

        try:
            with open('data/potentialAspects.dat', 'rb') as handle:
                self.potentialAspects = pickle.load(handle)
        except FileNotFoundError:

            print(" ... Creating TF-IDF CorpusReader")
            # train the tfidf model on the training corpus
            self.tf_idf_Model = CorpusReader_TFIDF(trainingCorpus, stemmer=None)

        print("Done creating AspectDetector")

    def run(self):

        if self.potentialAspects is None:
            # get all of the tf_idf values for the documents in the review corpus
            vectors = list()
            for file in self.reviewCorpus.fileids():
                newVec = self.tf_idf_Model.td_idf_new(self.reviewCorpus.words(file))
                vectors.append(newVec)


            sumVect = [0.0] * len(vectors[0])
            # now find the best words for all of them
            for vector in vectors:
                count = 0
                for value in vector:
                    sumVect[count] += value
                    count += 1
            # associate each word in the corpus with its average tf_idf value
            averageVect = dict(zip(self.tf_idf_Model.tf_idf_dim(), sumVect))
            # we now have an orderering of all the words in the corpus
            biggestWords = list(sorted(averageVect.keys(), key=averageVect.get, reverse=True))
            # now get the most common words
            # these will be the preliminaty aspects thwich will be farther narrowed down
            # right now will just take the top 1% although this can be narrowed down
            self.potentialAspects = biggestWords[:int(AspectDetector.TF_IDF_ASPECT_PERCENT * len(self.tf_idf_Model.tf_idf_dim()))]

            # going to now consider collocations of the corpus
            bigram_measures = nltk.collocations.BigramAssocMeasures()
            finder = BigramCollocationFinder.from_words(self.reviewCorpus.words())
            # find the most common bigrams of the corpus
            bigrams = finder.nbest(bigram_measures.pmi, AspectDetector.NUMBER_OF_TWO_GRAMS_TO_CONSIDER)

            # now we have both the "best" 2 grams, and the best "unigrams" (from the tfidf model)
            # what I am thinking is that now we see if any of the unigrams are mutually contained in a two-grams
            # ie if "battery" and "life" are both popular unigrams, and "battery life" is a common
            # bigram we cut out the term that appears later in the twogram
            # maybe should we also consider grams higher than 2 (probably big performance hit?!?)
            for bigram_1, bigram_2 in bigrams:
                if(bigram_1 in self.potentialAspects and bigram_2 in self.potentialAspects):
                    self.potentialAspects.remove(bigram_2)

            with open('data/potentialAspects.dat', 'wb') as handle:
                pickle.dump(self.potentialAspects, handle, protocol=pickle.HIGHEST_PROTOCOL)

        tagged_sents = []
        sents = sent_tokenize(self.reviewCorpus.raw())
        for sent in sents:
            tokens = [e1.lower() for e1 in word_tokenize(sent)]
            tagged_sent = nltk.pos_tag(tokens, tagset='universal')
            tagged_sents.append(tagged_sent)

        gram_dict = {}
        for tagged_sent in tagged_sents:
            for tup in tagged_sent:
                gram = tup[0]
                pos = tup[1]
                if gram in gram_dict:
                    pos_dict = gram_dict[gram]
                    pos_dict[pos] += 1
                else:
                    pos_dict = defaultdict(int)
                    pos_dict[pos] += 1
                    gram_dict[gram] = pos_dict

        for gram in self.potentialAspects:
            if gram in gram_dict:
                try:
                    pos_dict = gram_dict[gram]
                    max_pos = max(pos_dict, key=lambda e1: e1[1])
                    if max_pos != "NOUN":
                        self.potentialAspects.remove(gram)
                except IndexError:
                    self.potentialAspects.remove(gram)
            else:
                self.potentialAspects.remove(gram)

        tokens = [e1.lower() for e1 in word_tokenize(self.reviewCorpus.raw())]
        freq_raw = Counter(tokens)
        for aspect in self.potentialAspects:
            count = freq_raw[aspect]
            if count > 3:
                self.potentialAspects.remove(aspect)

        nonAllowed = string.punctuation + "1234567890"
        for aspect in self.potentialAspects:
            for token in nonAllowed:
                if token in aspect:
                    try:
                        self.potentialAspects.remove(aspect)
                    except ValueError:
                        pass

        return self.potentialAspects





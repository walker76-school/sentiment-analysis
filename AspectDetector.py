# laird
import nltk
from nltk.collocations import *
from CorpusReader_TFIDF import CorpusReader_TFIDF
from HMMTagger import HmmTagger
from nltk.tokenize import sent_tokenize, word_tokenize
from collections import defaultdict

class AspectDetector:
    TF_IDF_ASPECT_PERCENT = .01
    NUMBER_OF_TWO_GRAMS_TO_CONSIDER = 10

    def __init__(self, trainingCorpus, reviewCorpus):
        print("Creating AspectDetector ... ")
        self.trainingCorpus = trainingCorpus
        self.reviewCorpus = reviewCorpus

        print(" ... Creating TF-IDF CorpusReader")
        # train the tfidf model on the training corpus
        self.tf_idf_Model = CorpusReader_TFIDF(trainingCorpus)
        print(" ... Creating HMMTagger")
        self.tagger = HmmTagger()
        print("Done creating AspectDetector")

    def run(self):
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
        potentialAspects = biggestWords[:int(AspectDetector.TF_IDF_ASPECT_PERCENT * len(self.tf_idf_Model.tf_idf_dim()))]

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
            if(bigram_1 in potentialAspects and bigram_2 in potentialAspects):
                potentialAspects.remove(bigram_2)

        # tag all of the sentences
        sents = []
        tagged_sents = []
        for sent in sent_tokenize(self.reviewCorpus.raw()):
            tokenized = word_tokenize(sent)
            sents.append(tokenized)
            tagged_sent = self.tagger.tag(tokenized)
            tagged_sents.append(tagged_sent)

        gram_dict = {}
        freq_dict = defaultdict(int)

        # Filter every possible gram
        for gram in potentialAspects:

            # Check every sentence to see if the word is in it
            for sent in tagged_sents:

                # Check every word in the sentende
                for index, tup in enumerate(sent):

                    # If it's the word
                    if tup[0].lower() == gram:

                        # Increase frequency
                        freq_dict[gram] = freq_dict[gram] + 1

                        # Retrieve the POS tag
                        pos = tup[1]

                        # Check if we've already seen it, otherwise create default value
                        if gram in gram_dict:

                            # Retrieve the existing POS tag dictionary and update the count
                            pos_dict = gram_dict[gram]
                            pos_dict[pos] = pos_dict[pos] + 1
                            gram_dict[gram] = pos_dict

                        else:
                            # Default dictionary with count of 1 for the current POS tag
                            pos_dict = defaultdict(int)
                            pos_dict[pos] = 1
                            gram_dict[gram] = pos_dict

            if gram in gram_dict:
                pos_dict = gram_dict[gram]
                v = list(pos_dict.values())
                k = list(pos_dict.keys())
                max_pos = k[v.index(max(v))]
                if not max_pos == "NNP":
                    potentialAspects.remove(gram)

        sorted(potentialAspects, key=lambda l_gram: freq_dict[l_gram], reverse=True)

        return potentialAspects


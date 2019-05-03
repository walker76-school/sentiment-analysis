# author: Ian Laird
# file name: CorpusReader_TFIDF.py
# class: NLP
# instructor: Dr Lin
# due date: February 11, 2019
# date last modified: February 12, 2019

import nltk
from nltk.corpus import stopwords
from collections import Counter
from nltk.stem import PorterStemmer, SnowballStemmer
# import numpy as np
from scipy import spatial
import math
import copy
import pickle


# CorpusReader_TFIDF
# This class adds tf-idf functionality to a given corpus reader
# it will return vectors that correspond to every document in the desired corpus
class CorpusReader_TFIDF:

    # custom constructor
    #
    # param:
    #   corpus: the corpus that documents will be grabbed from
    #   tf: indicating how the term frequency will be calculated
    #       raw -- indicates that the raw score will be used
    #       log -- indicates that the score will be log normalized
    #       binary -- indicates that the score will be binary in output
    #           1 indicates that the term is present in the document
    #           0 indicates that the term is not present in the document
    #   idf: indicates how the inverse document frequency will be calculated
    #       base: simply the default
    #       smooth: the frequency is smoothed
    #       prob: probabilistic inverse document frequency
    #   stopword: what the stopwords will be
    #       standard -- use the default English stopwords
    #       anything else -- a fileName to read the stopwords from
    #   stemmer: the stemmer for the words
    #       the default is the Porter Stemmer
    #   ignorecase: indicates if case should be ignored
    #       no -- do not ignore case
    #       yes -- do ignore case
    #
    #   This method generates a vector for each document in the corpus and stores it
    def __init__(self, corpus, tf='raw', idf='base', stopword='standard', stemmer=PorterStemmer(), ignorecase='yes'):
        if isinstance(stemmer, str):
            if stemmer.lower() == 'porter':
                stemmer = PorterStemmer()
            elif stemmer.lower() == 'snowball':
                stemmer = SnowballStemmer("english")
            else:
                print("Error Unknown stemmer option")
                exit()
        wordStorage = dict()
        termFrequency = dict()
        fileFrequencyStorage = dict()
        dictClone= dict()

        self.corpus = corpus
        self.tf = tf
        self.idf = idf
        self.stopword = stopword
        self.actualStopWords = set()
        self.stemmer = stemmer
        self.ignorecase = ignorecase
        self.vectors = dict()
        self.dimensions = list()
        self.wordDocumentFrequency = dict()

        # depending on preferences read in the stopwords
        # if 'none' do not remove any stopwords
        if(self.stopword != 'none'):
            if(self.stopword == 'standard'):
                self.actualStopWords = set(stopwords.words('english'))
            else:
                #read in the stopwords from a file
                with open(stopword) as file:
                    fileContents = file.read()
                    self.actualStopWords = set(nltk.word_tokenize(fileContents))

        self.actualStopWords = self.lowerCaseCheckConversion(self.actualStopWords)

        # where all words in any document will be stored
        self.allWords = list()

        #calculate the vectors for the documents
        for fileId in corpus.fileids():
            tempWords = corpus.words(fileId)

            filteredWords = self.lowerCaseCheckConversion(tempWords)
            #now filter out the stopwords
            filteredWords = self.filterWords(filteredWords)
            filteredWords = self.stemWords(filteredWords)

            self.allWords.extend(filteredWords)

            #now need to store the read words so that the vector can be calculated later
            wordStorage[fileId] = filteredWords

        self.allWords = set(self.allWords)

        # initially each term is present in no documents
        for word in self.allWords:
            self.wordDocumentFrequency[word] = 0
            termFrequency[word] = 0
            self.dimensions.append(word)

        dictClone = copy.deepcopy(termFrequency)

        # iterate through every file and its contents
        for file, wordList in wordStorage.items():
            # count up how many times each term occurs in each file
            for word in wordList:
                termFrequency[word] = termFrequency[word] + 1
            for word, wordFrequency in termFrequency.items():
                if wordFrequency > 0:
                    self.wordDocumentFrequency[word] = self.wordDocumentFrequency[word] + 1
            fileFrequencyStorage[file] = termFrequency
            termFrequency = copy.deepcopy(dictClone)

        # now actually calculate the tf-idf values
        for fileId in corpus.fileids():
            vector = list()
            frequencyMap = fileFrequencyStorage[fileId]
            for word in self.allWords:
                # calculate the TFIDF value  using both the term frequency of the term
                # and the number of docs it appears
                vector.append(self.calculateTfIdfValue(frequencyMap[word], self.wordDocumentFrequency[word]))
            del fileFrequencyStorage[fileId]
            self.vectors[fileId] = vector


    # calculateTfIdfValue
    #
    # generates the tf-idf value for a given tf and idf. Varies for how the tf
    #   and idf preferences were set in the constructor
    #
    # param:
    #   tf: the term frequency
    #   idf: the inverse document frequency
    #
    # return:
    #   the calculates value of tf times the calculates value of idf
    def calculateTfIdfValue(self, tf, idf):
        idf_val = 1
        tf_val = 1
        try:
            if(tf == 0):
                tf_val = 0
            else:
                if self.tf == 'raw':
                    tf_val = tf
                elif self.tf == 'log':
                    tf_val = 1 + math.log(tf, 2)
                elif self.tf == 'binary':
                    tf_val = 1

            if self.idf == 'base':
                idf_val = math.log(len(self.corpus.fileids()) / idf, 2)
            elif self.idf == 'smooth':
                idf_val = math.log(1 + (len(self.corpus.fileids()) / idf), 2)
            elif self.idf == 'prob':
                    idf_val = math.log((len(self.corpus.fileids()) - idf) / idf, 2)

            return tf_val * idf_val
        except ZeroDivisionError:
            return 0.0
        except ValueError:
            return 0.0

    # fileids
    # returns all fileids of the contained corpus
    # input: none
    def fileids(self):
        return self.corpus.fileids()

    # raw
    # returns the raw text of the indicated fileids in the corpus
    # input:
    #     fileids: a list of all fileids to return
    def raw(self, fileids = []):
        if len(fileids) == 0:
            return self.corpus.raw()
        return self.corpus.raw(fileids)


    # words
    # returns all words in the desired fileids
    # input:
    #    fileids: a list of the fileids to print words from
    def words(self, fileids=[]):
        if len(fileids) == 0:
            return self.corpus.words()
        return self.corpus.words(fileids)

    # open
    # opens the desired fileid
    # input:
    #    fileid: the file to open
    def open(self, fileid):
        return self.corpus.open(fileid)

    # abspath
    # returns the absolute path of the file
    def abspath(self, fileid):
        return self.corpus.abspath(fileid)

    # tf_idf
    # returns all vectors for the specified documents
    # param:
    #   fileList: all documents to list
    def tf_idf(self, filelist=[]):
        # see if all documents are desired
        if isinstance(filelist, list) and len(filelist) == 0:
            return [x for x in self.vectors.values()]
        # see if only one fileId is given
        if isinstance(filelist, str):
            return self.vectors[filelist]
        # if a list of fileids is given return a list of vector
        # for the corresponding files
        return [self.vectors[x] for x in self.vectors.keys() if x in filelist]

    # tf_idf_dim
    # returns a list showing what word each dimension of the vector corresponds to
    def tf_idf_dim(self):
        """show in what order the words are in the vector"""
        return self.dimensions

    # td_idf_new
    #
    # creates a vector corresponding to the given document
    #
    # param:
    #   words: the words that are to be treated as a document
    def td_idf_new(self, words = []):
        """finds a new vector based on the existing idf values"""
        return self.getVector(self.stemWords(self.filterWords(self.lowerCaseCheckConversion(words))))

    # cosine_sim
    # returns the dot product of the two files
    def cosine_sim(self, fileId=[]):
        """finds the cosine similarity of two files"""
        if len(fileId) == 2:
            # return np.dot(self.vectors[fileId[0]], self.vectors[fileId[1]]) / (len(self.vectors[fileId[0]]) * len(self.vectors[fileId[1]]))
            return 1 - spatial.distance.cosine(self.vectors[fileId[0]], self.vectors[fileId[1]])
        else:
            print("Error: the number of fileids given should be 2!")
            return -1

    # cosine_sim_new
    # returns the cosine similarity between a new document and one of the documents in the corpus
    # param:
    #   words: the words that are to be treated as a document
    #   fileid: the existing document that it is to be compared with
    def cosine_sim_new(self, words = [] , fileid = 0):
        new_list = self.td_idf_new(words)
        list_2   = self.vectors[fileid]
        return 1 - spatial.distance.cosine(new_list, list_2)

    # getVector
    # gets the vector corresponding to given list of words
    def getVector(self, words):
        #need to find the frequency of every word in words
        freq = Counter(words)
        keys = freq.keys()
        # treat words like a  document
        returnVector  = list()
        for word in self.allWords:
            # calculate the TFIDF value  using both the term frequency of the term and the number of docs it appears
            returnVector.append(self.calculateTfIdfValue((freq[word] if word in keys else 0), self.wordDocumentFrequency[word] if word in self.wordDocumentFrequency.keys() else 0))
        return returnVector

    # filterWords
    # filters all stop words out of the collection of words
    # return: the filtered words
    def filterWords(self, words):
        if isinstance(words, set):
            return {x for x in words if x not in self.actualStopWords}
        return [x for x in words if x not in self.actualStopWords]

    # lowerCaseCheckConversion
    # if necessary converts strings to all lower case
    def lowerCaseCheckConversion(self, words):
        if self.ignorecase == 'yes':
            if isinstance(words, set):
                return {x.lower() for x in words}
            else:
                return [x.lower() for x in words]
        return words

    # stemwords
    # stems all words in the collection
    def stemWords(self, words):
        if self.stemmer is None:
            return words
        if isinstance(words, set):
            return {self.stemmer.stem(x) for x in words}
        return [self.stemmer.stem(x) for x in words]

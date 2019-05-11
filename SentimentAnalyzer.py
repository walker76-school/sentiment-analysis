# author: Andrew Walker, Ian Laird
# file name: SentimentAnalyzer.py
# class: NLP
# instructor: Dr Lin
# due date: May 10, 2019
# date last modified: May 10, 2019

import re
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.svm import LinearSVC
import pickle


class SentimentAnalyzer:

    # constructor
    #
    # param: none
    #
    # This method creates the SentimentAnalyzer
    def __init__(self):
        print("Creating SentimentAnalyzer ...")

        try:
            with open('data/vectorizer.dat', 'rb') as handle:
                self.vectorizer = pickle.load(handle)
        except FileNotFoundError:

            reviews_train = []
            for line in open('training_data/full_train.txt', 'r', encoding="utf8"):
                reviews_train.append(line.strip())

            print("... Created raw training data")

            reviews_test = []
            for line in open('training_data/full_test.txt', 'r', encoding="utf8"):
                reviews_test.append(line.strip())

            print("... Created raw testing data")

            self.train_clean = self.preprocess(reviews_train)
            self.test_clean = self.preprocess(reviews_test)

            print("... Done cleaning data")

            self.vectorizer = CountVectorizer(binary=True, ngram_range=(1, 2))
            print("Creating Vectorizer ...")

            self.vectorizer.fit(self.train_clean)
            print("... Done fitting training data")

            pickle.dump(self.vectorizer, open("data/vectorizer.dat", "wb"))
            print("... Done saving vectorizer")

        try:
            with open('data/model.dat', 'rb') as handle:
                self.svc_model = pickle.load(handle)
        except FileNotFoundError:
            target = [1 if i < 12500 else 0 for i in range(25000)]

            final = self.vectorizer.transform(self.train_clean)
            print("... Done transforming training data")

            print("Creating LinearSVC Model ...")
            self.svc_model = LinearSVC(C=0.01)
            print("... Done creating model")
            self.svc_model.fit(final, target)
            print("... Done fitting model")

            pickle.dump(self.svc_model, open("data/model.dat", "wb"))
            print("... Done saving LinearSVC Model")

        print("Done creating SentimentAnalyzer")

    # preprocess
    #
    # param:
    #   reviews: a list of sentences from a review
    #
    # This method cleans the text of a review
    def preprocess(self, reviews):
        REPLACE_NO_SPACE = re.compile("(\.)|(\;)|(\:)|(\!)|(\?)|(\,)|(\")|(\()|(\))|(\[)|(\])|(\d+)")
        REPLACE_WITH_SPACE = re.compile("(<br\s*/><br\s*/>)|(\-)|(\/)")
        NO_SPACE = ""
        SPACE = " "

        reviews = [REPLACE_NO_SPACE.sub(NO_SPACE, line.lower()) for line in reviews]
        reviews = [REPLACE_WITH_SPACE.sub(SPACE, line) for line in reviews]

        return reviews

    # training_data
    #
    # param: none
    #
    # This method returns the training data
    def training_data(self):
        return self.train_clean

    # testing_data
    #
    # param: none
    #
    # This method returns the testing data
    def testing_data(self):
        return self.test_clean

    # model
    #
    # param: none
    #
    # This method returns the LinearSVC model
    def model(self):
        return self.model

    # analyze
    #
    # param:
    #   sentence: the sentence to be analyzed
    #
    # This method returns the sentiment of a sentence
    def analyze(self, sentence):
        transform = self.vectorizer.transform(self.preprocess([sentence]))
        return self.svc_model.predict(transform)

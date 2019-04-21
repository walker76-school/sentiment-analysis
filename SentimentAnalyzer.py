import re
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.svm import LinearSVC
import pickle

class SentimentAnalyzer:
    def __init__(self):
        print("Initializing SentimentAnalyzer ...")
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

        stop_words = ['in', 'of', 'at', 'a', 'the']
        target = [1 if i < 12500 else 0 for i in range(25000)]

        self.vectorizer = CountVectorizer(binary=True, ngram_range=(1, 2))
        print("Creating Vectorizer ...")
        self.vectorizer.fit(self.train_clean)
        print("... Done fitting training data")
        final = self.vectorizer.transform(self.train_clean)
        print("... Done transforming training data")

        pickle.dump(self.vectorizer, open("vectorizer.dat", "wb"))
        print("... Done saving vectorizer")

        print("Creating LinearSVC Model ...")
        self.svc_model = LinearSVC(C=0.01)
        print("... Done creating model")
        self.svc_model.fit(final, target)
        print("... Done fitting model")

        pickle.dump(self.svc_model, open("model.dat", "wb"))
        print("... Done saving LinearSVC Model")

        print("Done initializing SentimentAnalyzer")

    def preprocess(self, reviews):
        REPLACE_NO_SPACE = re.compile("(\.)|(\;)|(\:)|(\!)|(\?)|(\,)|(\")|(\()|(\))|(\[)|(\])|(\d+)")
        REPLACE_WITH_SPACE = re.compile("(<br\s*/><br\s*/>)|(\-)|(\/)")
        NO_SPACE = ""
        SPACE = " "

        reviews = [REPLACE_NO_SPACE.sub(NO_SPACE, line.lower()) for line in reviews]
        reviews = [REPLACE_WITH_SPACE.sub(SPACE, line) for line in reviews]

        return reviews

    def training_data(self):
        return self.train_clean

    def testing_data(self):
        return self.test_clean

    def model(self):
        return self.model

    def analyze(self, sentence):

        transform = self.vectorizer.transform(self.preprocess([sentence]))
        return self.svc_model.predict(transform)


analyzer = SentimentAnalyzer()
value = analyzer.analyze("This is the worst movie")
print(value)
if value[0] == 1:
    print("Positive")
else:
    print("Negative")

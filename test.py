from MyCorpusReader import MyCorpusReader
from AspectDetector import AspectDetector
from nltk.corpus import brown
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from SentimentAnalyzer import SentimentAnalyzer
from collections import defaultdict

nltk.download('universal_tagset')
corpus = MyCorpusReader("reviews")
print(corpus.fileids())
print(corpus.words())
print(corpus.words(corpus.fileids()[0]))

POSITIVE_KEY = "POSITIVE"
NEGATIVE_KEY = "NEGATIVE"
NEUTRAL_KEY = "NEUTRAL"

a = AspectDetector(brown, corpus)
sentimentAnalyzer = SentimentAnalyzer()

potentialAspects = a.run()

aspect_dict = {}

for aspect in potentialAspects:
    if aspect in aspect_dict:
        sentiment_dict = aspect_dict[aspect]
    else:
        sentiment_dict = defaultdict(int)

    raw = corpus.raw()
    sents = sent_tokenize(raw)

    for sent in sents:
        tokens = [e1.lower() for e1 in word_tokenize(sent)]
        for word in tokens:
            if word.lower() == aspect:
                res = sentimentAnalyzer.analyze(sent)
                if res[0] == 1:
                    sentiment_dict[POSITIVE_KEY] += 1
                else:
                    sentiment_dict[NEGATIVE_KEY] += 1
    aspect_dict[aspect] = sentiment_dict

print(aspect_dict)




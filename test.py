from MyCorpusReader import MyCorpusReader
from AspectDetector import AspectDetector
from nltk.corpus import brown
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from SentimentAnalyzer import SentimentAnalyzer
from MyViterbi import MyViterbi
from collections import defaultdict
from nltk.parse.corenlp import CoreNLPDependencyParser

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
viterbi = MyViterbi()

parser = CoreNLPDependencyParser(url='http://localhost:9000')

potentialAspects = a.run()

aspect_dict = {}

raw = corpus.raw()
sents = sent_tokenize(raw)

for aspect in potentialAspects:

    true_aspect = False

    for sent in sents:

        if aspect not in sent:
            continue

        if ", " in sent:
            sent_tokens = sent.split(", ")
        else:
            sent_tokens = [sent]

        for sent_token in sent_tokens:

            tokens = [e1.lower() for e1 in word_tokenize(sent_token)]
            parse_triples = []
            try:
                parse = next(parser.parse(tokens))
                parse_triples = [(governor, dep, dependent) for governor, dep, dependent in parse.triples()]
            except ValueError:
                print("No tree for - %s" % sent)

            for trip in parse_triples:
                if trip[1] == "nsubj":
                    subj = trip[2][0]
                    if subj == aspect:
                        true_aspect = True

    if not true_aspect:
        print("Removing %s" % aspect)
        potentialAspects.remove(aspect)

for aspect in potentialAspects:

    if aspect in aspect_dict:
        sentiment_dict = aspect_dict[aspect]
    else:
        sentiment_dict = defaultdict(int)

    for sent in sents:
            tokens = [e1.lower() for e1 in word_tokenize(sent)]

            for word in tokens:
                if word.lower() == aspect:
                    res = sentimentAnalyzer.analyze(sent)
                    print("%s - %s" % (aspect, sent))
                    if res[0] == 1:
                        sentiment_dict[POSITIVE_KEY] += 1
                    else:
                        sentiment_dict[NEGATIVE_KEY] += 1

    aspect_dict[aspect] = sentiment_dict

for aspect in potentialAspects[:10]:
    print("%s %d %d %d" % (aspect, aspect_dict[aspect][POSITIVE_KEY], aspect_dict[aspect][NEUTRAL_KEY], aspect_dict[aspect][NEGATIVE_KEY]))




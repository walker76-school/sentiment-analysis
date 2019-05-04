from MyCorpusReader import MyCorpusReader
from AspectDetector import AspectDetector
from nltk.corpus import brown
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from SentimentAnalyzer import SentimentAnalyzer
from MyViterbi import MyViterbi
from collections import defaultdict
from nltk.parse.corenlp import CoreNLPDependencyParser
from nltk.corpus import wordnet as wn

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

ndx = int(0.2 * len(potentialAspects))
potentialAspects = potentialAspects[:ndx]

'''
sim_dict = {}

for outer_aspect in potentialAspects:
    for inner_aspect in potentialAspects:
        if outer_aspect == inner_aspect:
            continue

        print("%s - %s" % (outer_aspect, inner_aspect))

        outer_synsets = wn.synsets(outer_aspect)
        inner_synsets = wn.synsets(inner_aspect)

        count = 0
        total = 0.0
        for outer_synset in outer_synsets:
            for inner_synset in inner_synsets:
                sim = wn.wup_similarity(outer_synset, inner_synset)
                if sim is not None:
                    count += 1
                    total += sim

        try:
            avg = total / count
            sim_dict[(outer_aspect, inner_aspect)] = avg
        except ZeroDivisionError:
            pass
'''

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

            if aspect not in sent_token:
                continue

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

# map sentences to their word tokenization in lowercase
sentToWords = dict(zip(sents, map(lambda s: [w.lower() for w in word_tokenize(s)], sents)))

# free up the memory
del sents

for aspect in potentialAspects:

    if aspect in aspect_dict:
        sentiment_dict = aspect_dict[aspect]
    else:
        sentiment_dict = defaultdict(int)

    for sent, tokens in sentToWords.items():

            # for each word check if that word is the desired aspect
            # if so that means that the sentence relates to this aspect
            # we then check the sentiment of the sentence and add it to
            # a positive or negative count
            for word in tokens:
                if word == aspect:
                    res = sentimentAnalyzer.analyze(sent)
                    print("%s - %s" % (aspect, sent))
                    # positive sentiment
                    if res[0] == 1:
                        sentiment_dict[POSITIVE_KEY] += 1
                    # negative sentiment
                    else:
                        sentiment_dict[NEGATIVE_KEY] += 1
                    # this sentence is done being considered for this aspect once a match is found
                    break

    aspect_dict[aspect] = sentiment_dict

for aspect in potentialAspects[:10]:
    print("%s %d %d %d" % (aspect, aspect_dict[aspect][POSITIVE_KEY], aspect_dict[aspect][NEUTRAL_KEY], aspect_dict[aspect][NEGATIVE_KEY]))




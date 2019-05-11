from collections import defaultdict
from nltk.corpus import brown
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.parse.corenlp import CoreNLPDependencyParser
from MyCorpusReader import MyCorpusReader
from AspectDetector import AspectDetector
from MyWordNetSimilarity import wup_similarity
from SentimentAnalyzer import SentimentAnalyzer

POSITIVE_KEY = "POSITIVE"
NEGATIVE_KEY = "NEGATIVE"
NEUTRAL_KEY = "NEUTRAL"

corpus = MyCorpusReader("reviews")
# corpus = MyCorpusReader("_samplereview3")
a = AspectDetector(brown, corpus)
sentimentAnalyzer = SentimentAnalyzer()
parser = CoreNLPDependencyParser(url='http://localhost:9000')

raw = corpus.raw()
sents = sent_tokenize(raw)

# Retrieve the initial list of aspects
potentialAspects = a.run()

# Only consider the top 20% of aspects
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

print("Creating Word2Vec model")
model = gensim.models.Word2Vec([potentialAspects], min_count=1, size=5, window=2)
print("Done with Word2Vec model")

for aspect in potentialAspects:
    try:
        print("%s most similar to %s" % (aspect, model.most_similar(aspect)))
    except KeyError:
        pass

model_wordSimilarity = dict()
for w in potentialAspects:
    model_wordSimilarity[w] = dict()
# now try and eliminate bad ones using Wu Palmer similarity

# find the similarity of every potential aspect
for word1 in potentialAspects:
    for word2 in potentialAspects:
        if word1 == word2:
            continue
        try:
            l = model.wv.similarity(word1, word2)
            model_wordSimilarity[word1][word2] = l
        except KeyError:
            pass
# find the average similarity of each word
model_averageSimilarity = dict()
for word in potentialAspects:
    model_averageSimilarity[word] = sum(model_wordSimilarity[word].values()) / len(potentialAspects)
for word in potentialAspects:
    if model_averageSimilarity[word] < .35:
        potentialAspects.remove(word)
        print("Model - Removing %s with average similarity of %f" % (word, model_averageSimilarity[word]))
'''

# Setup variables for calculating average similarity
wordSimilarity = dict()
for w in potentialAspects:
    wordSimilarity[w] = dict()

# Now try and eliminate bad ones using Wu Palmer similarity

# Find the similarity of every potential aspect
for word1 in potentialAspects:
    for word2 in potentialAspects:

        # Don't compare the same word
        if word1 == word2:
            continue

        # Calculate similarity and store it in the dictionary
        l = wup_similarity(word1, word2)
        wordSimilarity[word1][word2] = l[0]

# Find the average similarity of each word
averageSimilarity = dict()
for word in potentialAspects:
    averageSimilarity[word] = sum(wordSimilarity[word].values()) / len(potentialAspects)

# Remove any potentialAspects with a similarity less than .35
for word in potentialAspects:
    if averageSimilarity[word] < .35:
        potentialAspects.remove(word)
        print("Removing %s with average similarity of %f" % (word, averageSimilarity[word]))

# Check to remove any potential aspects that aren't subjects of the sentence
for aspect in potentialAspects:

    true_aspect = False

    # Test every sentence
    for sent in sents:

        # Consider each chunk of the sentence individually
        if ", " in sent:
            sent_tokens = sent.split(", ")
        else:
            sent_tokens = [sent]

        for sent_token in sent_tokens:

            tokens = [e1.lower() for e1 in word_tokenize(sent_token)]

            inSentToken = False
            for token in tokens:
                if aspect == token:
                    inSentToken = True

            # If the word isn't in the sentence chunk then don't consider it
            if not inSentToken:
                continue

            # Tokenize and then retrieve the first parse tree
            parses = parser.parse(tokens)
            for parse in parses:
                parse_triples = [(governor, dep, dependent) for governor, dep, dependent in parse.triples()]

                # Check to see if it's the subject at least once
                for trip in parse_triples:
                    if trip[1] == "nsubj":
                        subj = trip[2][0]
                        pos = trip[2][1]
                        if subj == aspect and "NN" in pos:
                            true_aspect = True
                            break
                    elif trip[1] == "compound":
                        subj = trip[2][0]
                        pos = trip[2][1]
                        if subj == aspect and "NN" in pos:
                            true_aspect = True
                            break
                    elif trip[1] == "conj":
                        subjA = trip[0][0]
                        subjB = trip[2][0]
                        posA = trip[0][1]
                        posB = trip[2][1]
                        if (subjA == aspect and "NN" == posA) or (subjB == aspect and "NN" == posB):
                            true_aspect = True
                            break
                    elif trip[1] == "dobj":
                        subj = trip[2][0]
                        pos = trip[2][1]
                        if subj == aspect and "NN" == pos:
                            true_aspect = True
                            break

                if true_aspect:
                    break

    # If it's not ever the subject then remove it
    if not true_aspect:
        print("Removing %s" % aspect)
        potentialAspects.remove(aspect)

# Map sentences to their word tokenization in lowercase
sentToWords = dict(zip(sents, map(lambda s: [w.lower() for w in word_tokenize(s)], sents)))

# free up the memory
del sents

# Setup variables for calculating sentiment per aspect
aspect_dict = {}
for w in potentialAspects:
    aspect_dict[w] = defaultdict(int)

# Map each aspect to a dictionary holding the count of positive, neutral and negative reviews
for aspect in potentialAspects:

    sentiment_dict = aspect_dict[aspect]

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
                    if len(res) > 1:
                        sentiment_dict[NEUTRAL_KEY] += 1
                    elif res[0] == 1:
                        sentiment_dict[POSITIVE_KEY] += 1
                    # negative sentiment
                    elif res[0] == 0:
                        sentiment_dict[NEGATIVE_KEY] += 1

                    # this sentence is done being considered for this aspect once a match is found
                    break

    aspect_dict[aspect] = sentiment_dict

handle = open("out.csv", "w")
for aspect in potentialAspects:
    print("%-10s %d %d %d" % (aspect, aspect_dict[aspect][POSITIVE_KEY], aspect_dict[aspect][NEUTRAL_KEY], aspect_dict[aspect][NEGATIVE_KEY]))
    handle.write("%s,%d,%d,%d\n" % (aspect, aspect_dict[aspect][POSITIVE_KEY], aspect_dict[aspect][NEUTRAL_KEY], aspect_dict[aspect][NEGATIVE_KEY]))
handle.close()




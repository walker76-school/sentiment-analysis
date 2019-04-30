from MyCorpusReader import MyCorpusReader
from AspectDetector import AspectDetector
from nltk.corpus import *

corpus = MyCorpusReader("reviews")
print(corpus.fileids())
print(corpus.words())
print(corpus.words(corpus.fileids()[0]))

a = AspectDetector(state_union, corpus)

print(a.run())
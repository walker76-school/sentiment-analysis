from MyCorpusReader import MyCorpusReader

corpus = MyCorpusReader("testData", "")
print(corpus.fileIds())
print(corpus.words())
print(corpus.words(corpus.fileIds()[0]))
# Yunzhe Liu
# Austin Lau
# CSI 4V96
import nltk
import os
from nltk.corpus import PlaintextCorpusReader


class MyCorpusReader:
    # Root of path
    root = ""
    # stemmer
    stemmer = None
    # list of stop words
    stop_words = None

    # Constructor
    # dname - Name of directory of corpus
    # sfile - File with stopwords, default English stopwords in NLTK
    # stemmer - porter, snowball, etc., default none
    def __init__(self, dname, sfile=None, stemmer=None):
        if not os.path.isdir(os.path.abspath(dname)):
            raise FileExistsError('invalid directory!')

        if sfile == "":
            self.stop_words = ()
        elif sfile is None:
            self.stop_words = set(nltk.corpus.stopwords.words('english'))
        else:
            if not os.path.exists(sfile):
                raise FileExistsError('invalid file!')
            else:
                reader = PlaintextCorpusReader(str(os.getcwd()), sfile)
                self.stop_words = set(reader.words([sfile, ]))

        self.root = os.path.abspath(dname)
        if isinstance(stemmer, nltk.stem.porter.PorterStemmer) or \
            isinstance(stemmer, nltk.stem.snowball.SnowballStemmer) or \
                stemmer is None:
            self.stemmer = stemmer
        else:
            raise Exception('invalid stemmer')

    # the raw content of the specified files
    def raw(self, fileidList=None):
        content_list = ""

        if fileidList is None:
            fileidList = os.listdir(self.root)
        else:
            if isinstance(fileidList, str):
                fileidList = [fileidList, ]

        for file in fileidList:
            with open(os.path.join(self.root, file)) as opened_file:
                content_list = content_list + opened_file.read()

        return content_list

    # the words of the specified fileids
    def words(self, fileidList=None):
        if fileidList is None:
            fileidList = os.listdir(self.root)
        else:
            if isinstance(fileidList, str):
                fileidList = [fileidList, ]
        return_words = []
        for file in fileidList:
            reader = PlaintextCorpusReader(self.root, file)
            return_words.extend(reader.words())
        return return_words

    # the sentences of the specified fileids
    def sents(self, fileidList=None):
        if fileidList is None:
            fileidList = os.listdir(self.root)
        else:
            if isinstance(fileidList, str):
                fileidList = [fileidList, ]
        return_sents = []
        for file in fileidList:
            reader = PlaintextCorpusReader(self.root, file)
            return_sents.extend(reader.sents())
        return return_sents

    # the location of the given file on disk
    def abspath(self, fileid):
        path = os.path.join(self.root, fileid)
        if not os.path.exists(path):
            raise FileExistsError('invalid file!')
        return path

    # count the word after stemming of each file
    def count(self):
        stemmed_words = []
        if self.stemmer is None:
            stemmed_words = self.words(os.listdir(self.root))
        else:
            for word in self.words(os.listdir(self.root)):
                stemmed_words.append(self.stemmer.stem(word))

        word_freq_dict = {}
        for word in stemmed_words:
            word_freq_dict[word] = word_freq_dict.get(word, 0) + 1

        return word_freq_dict.items()

    # count the frequency of a given file before stemming in each file
    def countWord(self, string):
        string_freq = {}
        for doc in os.listdir(self.root):
            for word in self.words(doc):
                if word == string:
                    string_freq[doc] = string_freq.get(doc, 0) + 1
        return string_freq

    # count the total words in each file, remove stop words
    def totalWords(self):
        word_count = {}
        for doc in os.listdir(self.root):
            for word in self.words(doc):
                if word not in self.stop_words:
                    word_count[doc] = word_count.get(doc, 0) + 1
        return word_count

    def fileIds(self):
        return os.listdir(self.root)

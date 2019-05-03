# author: Ian, Justin, Peter
# file name: MyViterbi.py
# class: NLP
# instructor: Dr Lin
# due date: April 17, 2019
# date last modified: April 16, 2019

import nltk
from nltk.corpus import treebank
from nltk.parse import ViterbiParser


class MyViterbi:

    # init
    # create the object
    # param: void
    # return: void
    def __init__(self):
        productions = list()
        S = nltk.Nonterminal('S')
        for tree in treebank.parsed_sents():
            productions += tree.productions()
        # create the grammar
        pcfg = nltk.induce_pcfg(S, productions)
        # print(pcfg)
        self.viterb = ViterbiParser(pcfg)
        self.mostRecentTree = None
        self.validTags = set()

        # pos tags
        self.validTags.add("CC")
        self.validTags.add("CD")
        self.validTags.add("DT")
        self.validTags.add("EX")
        self.validTags.add("FW")
        self.validTags.add("IN")
        self.validTags.add("JJ")
        self.validTags.add("JJR")
        self.validTags.add("JJS")
        self.validTags.add("LS")
        self.validTags.add("MD")
        self.validTags.add("NN")
        self.validTags.add("NNS")
        self.validTags.add("NNP")
        self.validTags.add("NNPS")
        self.validTags.add("PDT")
        self.validTags.add("POS")
        self.validTags.add("PRP")
        self.validTags.add("PRP$")
        self.validTags.add("PR")
        self.validTags.add("PBR")
        self.validTags.add("PBS")
        self.validTags.add("RP")
        self.validTags.add("SYM")
        self.validTags.add("TO")
        self.validTags.add("UH")
        self.validTags.add("VB")
        self.validTags.add("VBZ")
        self.validTags.add("VBP")
        self.validTags.add("VBD")
        self.validTags.add("VBG")
        self.validTags.add("WDT")
        self.validTags.add("WP")
        self.validTags.add("WP$")
        self.validTags.add("WRB")
        self.validTags.add(".")
        self.validTags.add(",")
        self.validTags.add(":")
        self.validTags.add("(")
        self.validTags.add(")")

        # chunk tags
        self.validTags.add("NP")
        self.validTags.add("PP")
        self.validTags.add("VP")
        self.validTags.add("ADVP")
        self.validTags.add("ADJP")
        self.validTags.add("SBAR")
        self.validTags.add("PRT")
        self.validTags.add("INTJ")

        # IOB tags
        self.validTags.add("I-")
        self.validTags.add("O-")
        self.validTags.add("B-")

        # prepositional noun phrase
        self.validTags.add("PNP")

        # relation tags
        self.validTags.add("-SBJ")
        self.validTags.add("-OBJ")
        self.validTags.add("-PRD")
        self.validTags.add("-TMP")
        self.validTags.add("-CLR")
        self.validTags.add("-LOC")
        self.validTags.add("-DIR")
        self.validTags.add("-EXT")
        self.validTags.add("-PRP")

        # anchor tags
        self.validTags.add("A1")
        self.validTags.add("P1")


    # parse
    # returns a parse tree corresponding to the given string
    #
    # param:
    #   x : the string to be parsed
    # return:
    #   the parse tree corresponding to x
    def parse(self, x):
        tokenizedSent = nltk.word_tokenize(x)
        trees = self.viterb.parse_all(tokenizedSent)
        # save the first one and then return it
        self.mostRecentTree = trees[0]
        return self.mostRecentTree


    # lastparse_label
    # returns all subtrees that has the given label for the root for the last
    # generated tree
    # param:
    #   x : the label
    # return:
    #   a list of all subtrees that have x as the label of the root
    def lastparse_label(self, x):
        # see if a previous tree exists
        if self.mostRecentTree is None:
            raise RuntimeError("No previous tree exists")
        if x not in self.validTags:
            raise RuntimeError("Invalid label")
        return [subtree for subtree in self.mostRecentTree.subtrees(lambda t: t.label() == x)]

    # lastparse_height
    # returns the height of the tree that was just generated
    #
    # return: the height of the tree
    def lastparse_height(self):
        # see if a previous tree exists
        if self.mostRecentTree is None:
            raise RuntimeError("No previous tree exists")
        return self.mostRecentTree.height()

    # wordsFromChunks
    # helper function for taking the trees given and turning them into a lists of words
    def wordsFromChunks(self, label):
        chunks = self.lastparse_label(label)
        returnList = list()
        for chunk in chunks:
            temp = chunk.pos()
            returnList.append([word for word, pos in temp])
        return returnList

    # lastparse_nounphrase
    # returns all noun phrases of the most recently generated tree
    # return:
    #   all noun phrases
    def lastparse_nounphrase(self):
        return self.wordsFromChunks("NP")

    # lastparse_verbphrase
    # returns all verb phrases of the most recently generated tree
    # return:
    #   all verb phrases
    def lastparse_verbphrase(self):
        return self.wordsFromChunks("VP")

    # lastparse_verbs
    # returns all verbs of the most recently generated tree
    # return:
    #   all verbs
    def lastparse_verbs(self):
        result = []
        verbList = ['VB','VBZ','VBP','VBD','VBG']
        for i in range(0,len(verbList)):
            tmp = self.wordsFromChunks(verbList[i])
            for j in range(0,len(tmp)):
                result.append(tmp[j])

        return result

    # lastparse_nouns
    # returns all nouns of the most recently generated tree
    # return:
    #   all nouns
    def lastparse_nouns(self):
        result = []
        nounList = ['NN','NNS','NNP','NNPS']
        for i in range(0,len(nounList)):
            tmp = self.wordsFromChunks(nounList[i])
            for j in range(0,len(tmp)):
                result.append(tmp[j])

        return result

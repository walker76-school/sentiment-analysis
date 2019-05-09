# author: Ian, Justin, Peter
# file name: MyViterbi.py
# class: NLP
# instructor: Dr Lin
# due date: April 17, 2019
# date last modified: April 16, 2019

import nltk
from nltk.corpus import treebank
from nltk.parse import ViterbiParser
from collections import defaultdict


class MyViterbi:

    # init
    # create the object
    # param: void
    # return: void
    def __init__(self):
        self.wordToTags = defaultdict(set)
        convertedTaggedWords = [(w,nltk.tag.mapping.map_tag('en-ptb', 'universal', t)) for w,t in treebank.tagged_words()]
        for word, tag in convertedTaggedWords:
            self.wordToTags[word].add(tag)

        productions = list()
        S = nltk.Nonterminal('S')
        for tree in treebank.parsed_sents():
            productions += tree.productions()
        # create the grammar
        pcfg = nltk.induce_pcfg(S, productions)
        # print(pcfg)
        self.viterb = ViterbiParser(pcfg)
        self.mostRecentTree = None
        self.validPosTags = set()
        self.validChunkTags = set()
        self.validIOBTags = set()
        self.relationTags = set()
        self.anchorTags = set()

        # pos tags
        self.validPosTags.add("CC")
        self.validPosTags.add("CD")
        self.validPosTags.add("DT")
        self.validPosTags.add("EX")
        self.validPosTags.add("FW")
        self.validPosTags.add("IN")
        self.validPosTags.add("JJ")
        self.validPosTags.add("JJR")
        self.validPosTags.add("JJS")
        self.validPosTags.add("LS")
        self.validPosTags.add("MD")
        self.validPosTags.add("NN")
        self.validPosTags.add("NNS")
        self.validPosTags.add("NNP")
        self.validPosTags.add("NNPS")
        self.validPosTags.add("PDT")
        self.validPosTags.add("POS")
        self.validPosTags.add("PRP")
        self.validPosTags.add("PRP$")
        self.validPosTags.add("PR")
        self.validPosTags.add("PBR")
        self.validPosTags.add("PBS")
        self.validPosTags.add("RP")
        self.validPosTags.add("SYM")
        self.validPosTags.add("TO")
        self.validPosTags.add("UH")
        self.validPosTags.add("VB")
        self.validPosTags.add("VBZ")
        self.validPosTags.add("VBP")
        self.validPosTags.add("VBD")
        self.validPosTags.add("VBG")
        self.validPosTags.add("WDT")
        self.validPosTags.add("WP")
        self.validPosTags.add("WP$")
        self.validPosTags.add("WRB")
        self.validPosTags.add(".")
        self.validPosTags.add(",")
        self.validPosTags.add(":")
        self.validPosTags.add("(")
        self.validPosTags.add(")")

        # chunk tags
        self.validChunkTags.add("NP")
        self.validChunkTags.add("PP")
        self.validChunkTags.add("VP")
        self.validChunkTags.add("ADVP")
        self.validChunkTags.add("ADJP")
        self.validChunkTags.add("SBAR")
        self.validChunkTags.add("PRT")
        self.validChunkTags.add("INTJ")
        self.validChunkTags.add("PNP")

        # IOB tags
        self.validIOBTags.add("I-")
        self.validIOBTags.add("O-")
        self.validIOBTags.add("B-")

        # relation tags
        self.relationTags.add("SBJ")
        self.relationTags.add("OBJ")
        self.relationTags.add("PRD")
        self.relationTags.add("TMP")
        self.relationTags.add("CLR")
        self.relationTags.add("LOC")
        self.relationTags.add("DIR")
        self.relationTags.add("EXT")
        self.relationTags.add("PRP")

        # anchor tags
        self.anchorTags.add("A1")
        self.anchorTags.add("P1")


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
        # see if it is a POS tag
        if x not in self.validPosTags:
            # if not see if it is a chunk tag
            stringParts = x.split("-")
            if len(stringParts) == 2 and stringParts[1] not in self.relationTags:
                raise RuntimeError("Invalid relation label")
            if stringParts[0] not in self.validChunkTags:
                raise RuntimeError("Invalid tag")
        return [subtree for subtree in self.mostRecentTree.subtrees(lambda t: t.label() == x)]

    def lastparse_phrase(self, x):
        # find all subtrees of a certain type
        # see if a previous tree exists
        if self.mostRecentTree is None:
            raise RuntimeError("No previous tree exists")
        if x not in self.validChunkTags:
            raise RuntimeError("not a valid type of chunk")
        return [subtree for subtree in self.mostRecentTree.subtrees(lambda t: x in t.label())]

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
    def wordsFromChunks(self, label, alternateMode = False):
        chunks = self.lastparse_phrase(label) if alternateMode else self.lastparse_label(label)
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
        return self.wordsFromChunks("NP", True)

    # lastparse_verbphrase
    # returns all verb phrases of the most recently generated tree
    # return:
    #   all verb phrases
    def lastparse_verbphrase(self):
        return self.wordsFromChunks("VP", True)

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

    def parse_with_substitution(self,x):
        tokenizedSent = nltk.word_tokenize(x)
        posTags = nltk.pos_tag(tokenizedSent, tagset='universal')
        fixedSentence = list()
        for word, tag in posTags:
            if tag in self.wordToTags[word]:
                fixedSentence.append(word)
            else:
                for word, tags in self.wordToTags.items():
                    if tag in tags:
                        fixedSentence.append(word)
                        break

        print(fixedSentence)

        trees = self.viterb.parse_all(fixedSentence)
        # save the first one and then return it
        self.mostRecentTree = trees[0]
        return self.mostRecentTree

"""
v = MyViterbi()
try:
    v.parse("The board of directors is eating a cupcake")
except Exception:
    print("bad")
v.parse_with_substitution("The board of directors is eating a cupcake")
"""


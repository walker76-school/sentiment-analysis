# Authors: Everett Wilson and Andrew Walker
# Assignment: NLP Undergrad Project - Part 5, HMM POS tagging

from nltk.corpus import treebank
from nltk.tag import hmm
import numpy as np


# A modified Hidden Markov Model POS tagger, that allows for modified POS tags
class HmmTagger:

    # Constructor
    def __init__(self):
        self.substitute = {
            "VB": "VBE",
            "VBD": "VBE",
            "VBG": "VBE",
            "VBN": "VBE",
            "VBP": "VBE",
            "VBZ": "VBE",
            "VH": "VH",
            "VHD": "VH",
            "VHG": "VH",
            "VHN": "VH",
            "VHP": "VH",
            "VHZ": "VH",
            "VV": "V",
            "VVD": "V",
            "VVG": "V",
            "VVN": "V",
            "VVP": "V",
            "VVZ": "V",
            "JJ": "J",
            "JJR": "J",
            "JJS": "J",
            "NN": "N",
            "NNS": "N",
            "NP": "N",
            "NPS": "N",
            "RB": "ADV",
            "RBR": "ADV",
            "RBS": "ADV"
        }

        # Instantiate, train, and get tagger
        tagged_sents = treebank.tagged_sents()
        trainer = hmm.HiddenMarkovModelTrainer()
        self.my_tagger = trainer.train_supervised(tagged_sents)

    # Preprocess an array of tagged sentence tuples to use our substitution dictionary
    # Unfortunately, have to build new arrays, since in-place tuple editing is not allowed
    # Parameters:
    # lines - array of tuples from a tagged corpus of the format (word, POS)
    def process_lines(self, lines):
        new_lines = []
        for line in lines:
            new_line = []
            for tup in line:
                elem1 = tup[1]
                if tup[1] in self.substitute:
                    elem1 = self.substitute[tup[1]]
                new_line.append((tup[0], elem1))
            new_lines.append(new_line)
        return new_lines

    # Return trained tagger
    def tagger(self):
        return self.my_tagger

    def tag(self, sent):
        return self.my_tagger.tag(sent)

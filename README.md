# Aspect Based Sentiment Analysis

This library is built using Python and the NLTK library to detect aspects and sentiment of reviews for a certain product

## Getting Started

These instructions will get a copy of the project up and running on your local machine

### Prerequisites

Before installing this library you'll want to make sure you have NLTK downloaded

```
> pip install nltk
```

Next you'll need to install the required NLTK corpra by first opening the Python terminal
```
> python
```

Once the Python terminal is open use the following commands to open the NLTK downloader
```
>> import nltk
>> nltk.download()
```

Once the download window pops up, install the following corpra -

* brown - Brown Corpus
* punkt - Punkt Tokenizer Models
* stopwords - Stopwords Corpus
* treebank - Penn Treebank Sample
* universal_tagset - Mappings to the Universal Part-of-Speech Tagset

Next you will have to install the CoreNLP models for using the CoreNLP Dependency Parser

Navigate to [CoreNLP's download page](https://stanfordnlp.github.io/CoreNLP/download.html) or use the direct [link](https://nlp.stanford.edu/software/stanford-parser-full-2018-10-17.zip) to download the required models and unzip them to a convinent location.

To host the models locally use the following command:

```
> java -mx4g -cp "*" edu.stanford.nlp.pipeline.StanfordCoreNLPServer -preload tokenize,ssplit,pos,lemma,ner,parse,depparse -status_port 9000 -port 9000 -timeout 15000 & 
``` 

The models will need to be hosted or the project will not be able to run

## Built With

* [NLTK](https://www.nltk.org/) - Python library for Natural Language Processing
* [Standford CoreNLP](https://stanfordnlp.github.io/CoreNLP/) - NLP tools developed by Standford

## Authors

* [**Andrew Walker**](https://github.com/walker76)
* [**Ian Laird**](https://github.com/i-laird)

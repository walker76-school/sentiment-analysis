# Aspect Based Sentiment Analysis

This library is built using Python and the NLTK library to detect aspects and sentiment of reviews for a certain product

## Getting Started

These instructions will get a copy of the project up and running on your local machine

### NLTK Prerequisites

Before installing this project you'll want to make sure you have NLTK downloaded

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

Once the downloader window pops up, install the following corpra -

* averaged_perceptron_tagger - Averaged Perceptron Tagger
* brown - Brown Corpus
* punkt - Punkt Tokenizer Models
* stopwords - Stopwords Corpus
* treebank - Penn Treebank Sample
* universal_tagset - Mappings to the Universal Part-of-Speech Tagset
* wordnet - WordNet
* words - Word List

### Additional Prerequisites

You will need the SciPy and scikit-learn libraries
```
> pip install scipy
> pip install sklearn
```

### CoreNLP

Next you will have to install CoreNLP for using the CoreNLP Dependency Parser

Navigate to [CoreNLP's download page](https://stanfordnlp.github.io/CoreNLP/download.html) or use the direct [link](https://nlp.stanford.edu/software/stanford-corenlp-full-2018-10-05.zip) to download the required models and unzip them to a convenient location.

In terminal before hosting the models, change directory to the CoreNLP folder
```
> cd /<path-to-corenlp>/
```

To host the models use the following command

```
> java -mx4g -cp "*" edu.stanford.nlp.pipeline.StanfordCoreNLPServer -preload tokenize,ssplit,pos,lemma,ner,parse,depparse -status_port 9000 -port 9000 -timeout 15000 & 
``` 

The models will need to be hosted or the project will not be able to run

### Caching Data

If you want to cache data to make subsequent runs on the same data faster then create a data folder in the root directory of this project

```
> cd /<path-to-project>/
> mkdir data
```

The files that will be generated into the data folder are the following

* model.dat - Stores the LinearSVC model used in sentiment analysis
* potentialAspects.dat - Stores the initial potential aspects from the AspectDetector
* vectorizer.dat - Stores the CountVectorizer used in sentiment analysis

These files will be quite large ( > 10mb )

## Built With

* [NLTK](https://www.nltk.org/) - Library for Natural Language Processing
* [Standford CoreNLP](https://stanfordnlp.github.io/CoreNLP/) - NLP tools developed by Standford
* [Scikit-Learn](https://scikit-learn.org/stable/) - Machine Learning framework for Python
* [SciPy](https://www.scipy.org/) - Library used for scientific and technical computing

## Authors

* [**Andrew Walker**](https://github.com/walker76)
* [**Ian Laird**](https://github.com/i-laird)

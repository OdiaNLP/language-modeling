# Language Modeling in Odia

Build language model in Odia for predicting next set of words.

## Dependencies
See the dependencies in `requirements.txt`. ⏰ TODO: create `requirements.txt`.
The code has been tested with Python 3.6.

## Overview

- First download Odia text data.

```shell
mkdir data
cd data

!wget https://storage.googleapis.com/ai4bharat-public-indic-nlp-corpora/data/monolingual/indicnlp_v1/sentence/or.txt.gz
tar -zxvf or.txt.gz
head or
```
- Train a language model:
  - To train a ngram language model, see the notebook `ngram_language_model.ipynb`. The ngram language model only captures context till a fixed history (determined by n) while generating next words.
  - ⏰ TODO: neural language models
- Finally run `controller.py` by setting the `--lm` argument to the desired language model and start the web app. Go to http://127.0.0.1:31137/generate to access the web app. There are three choices of the language model: `ngram`, `lstm` and `transformer`. Make sure that you trained model file is available before using it in the web app.

```shell
# web app
python controller.py --lm ngram # open http://127.0.0.1:31137/generate in browser
```

## Snapshot of web app

<img src="/snapshot.png" width="75%" height="75%"/>

(In the above snapshot, the ngram language model is used for generation.)

[LICENSE](https://github.com/OdiaNLP/language-modeling/blob/main/LICENSE)
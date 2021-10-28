# Language Modeling in Odia

Build language model in Odia for predicting next set of words.

## Dependencies
See the dependencies in `requirements.txt`.  :fire_alarm: TODO
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

- TODO
- Finally run `controller.py` to start the web app. Go to http://127.0.0.1:31137/generate to access the web app.

```shell
# web app
python controller.py  # open http://127.0.0.1:31137/generate in browser
```

## Snapshot of web app
TODO

<img src="/snapshot.png" width="75%" height="75%"/>

[LICENSE](https://github.com/OdiaNLP/language-modeling/blob/main/LICENSE)
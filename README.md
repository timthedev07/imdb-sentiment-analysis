# IMDb Sentiment Analysis

## Downloading the dataset

```bash
cd data
wget 'http://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz'
tar -xf 'aclImdb_v1.tar.gz'
```

## Install Dependencies(if not already)

```bash
pip install -r requirements.txt
```

## Predicting sentiments with the model

Populate the file `sample.txt` with the text you want to analyze.

Then, in the root directory of the project:

```bash
python src/model.py
```

The number above the conclusion indicates how positive/negative your text is, but it also depends on how much text you have.

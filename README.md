# Tweet Sentiment Analysis

Sentiment analysis on about 11,000 tweets about the Pfizer/BioNTech vaccine.

The dataset has no sentiment labels, so the tweets are labelled automatically with TextBlob (positive / neutral / negative). A Logistic Regression model is then trained on TF-IDF features to predict those labels. At the end the same tweets are labelled with VADER to see how far the two tools agree.

## Result

Logistic Regression reaches **84.49%** accuracy on the test set (20% of the data).

Keep in mind this is accuracy against TextBlob labels, not human labels. It shows how well a simple model can reproduce TextBlob, it does not say the sentiment itself is right 84% of the time.

## Dataset

[Pfizer Vaccine Tweets](https://www.kaggle.com/datasets/gpreda/pfizer-vaccine-tweets) by Gabriel Preda on Kaggle (CC0).

The csv is not stored in this repo. The notebook downloads it on the first run using `kagglehub` and saves it to `data/vaccination_tweets.csv`. If that fails, download the file from the Kaggle page and put it in the `data` folder yourself.

## What the notebook does

1. Loads the tweets, keeps the text and date
2. Cleans the text (links, mentions, punctuation, duplicates)
3. Labels every tweet with TextBlob polarity
4. Charts
   - tweets per sentiment
   - polarity score histogram
   - tweets per week by sentiment
   - tweet length by sentiment
   - most frequent words per sentiment
   - confusion matrix of the model
   - TextBlob vs VADER heatmap
5. TF-IDF (unigrams + bigrams, 5000 features) + Logistic Regression, 80/20 split
6. VADER labels and comparison with TextBlob

All charts are saved to `images/`.

![sentiment counts](images/01_sentiment_counts.png)

![confusion matrix](images/06_confusion_matrix.png)

![textblob vs vader](images/07_textblob_vs_vader.png)

## How to run

Python 3.9 or newer.

```
git clone <repo url>
cd tweet-sentiment-analysis
pip install -r requirements.txt
jupyter notebook tweet_sentiment_analysis.ipynb
```

Then Run All. No GPU needed, the whole thing takes less than a minute on an old laptop.

## Files

```
tweet_sentiment_analysis.ipynb   the notebook
requirements.txt
data/                            csv goes here (ignored by git)
images/                          charts saved by the notebook
```

## Limitations

- Labels are machine made, so there is no real ground truth
- TextBlob and VADER are both word-list based and miss sarcasm
- Only English tweets are handled properly

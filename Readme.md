 # Sentiment Analysis on Tweets
 
 
## Dataset Information

We use and compare various different methods for sentiment analysis on tweets (a binary classification problem). The training dataset is expected to be a csv file of type `tweet_id,sentiment,tweet` where the `tweet_id` is a unique integer identifying the tweet, `sentiment` is either `1` (positive) or `0` (negative), and `tweet` is the tweet enclosed in `""`. Similarly, the test dataset is a csv file of type `tweet_id,tweet`. Please note that csv headers are not expected and should be removed from the training and test datasets.  

## Requirements

There are some general library requirements for the project and some which are specific to individual methods. The general requirements are as follows.  
* `numpy`
* `scikit-learn`
* `scipy`
* `nltk`

The library requirements specific to some methods are:
* `keras` with `TensorFlow` backend for Logistic Regression, MLP, RNN (LSTM), and CNN.

## Information about other files

* `dataset/train_tweets.csv`: Training tweets datasets
* `dataset/test_tweets.csv`: Test tweets datasets
* `dataset/positive-words.txt`: List of positive words
* `dataset/negative-words.txt`: List of negative words
* `dataset/glove-seeds.txt`: GloVe words vectors from StanfordNLP which match our dataset for seeding word embeddings.
* `code/sentiment-analysis.ipynb`: IPython notebook used to perfom sentiment classification using different model. (Not updated)
* `code/word_cloud_generation_and_hashtag_processing.ipynb`: IPython notebook used to perfom word cloud generation and hashtag processing.

## Visualizations
1. Tweets distribution
<img src="https://github.com/vaishanth-rmrj/Twitter-Sentiment-Analysis-Using-Different-Models/blob/master/extras/tweet_distribution.png" width=500/>

2. Tweet words freq distribution
<img src="https://github.com/vaishanth-rmrj/Twitter-Sentiment-Analysis-Using-Different-Models/blob/master/extras/word_freq.png" width=500/>

3. Word Cloud generation
<img src="https://github.com/vaishanth-rmrj/Twitter-Sentiment-Analysis-Using-Different-Models/blob/master/extras/word_cloud.png" width=500/>

4. Positive Hashtag Freq
<img src="https://github.com/vaishanth-rmrj/Twitter-Sentiment-Analysis-Using-Different-Models/blob/master/extras/pos_hastag.png" width=500/>
          
5. Negative Hashtag Freq
<img src="https://github.com/vaishanth-rmrj/Twitter-Sentiment-Analysis-Using-Different-Models/blob/master/extras/neg_hastag.png" width=500/>




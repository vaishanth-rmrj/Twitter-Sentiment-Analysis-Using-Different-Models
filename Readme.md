 # Sentiment Analysis on Tweets
 
 
## Dataset Information

Dataset Information
We use and compare various different methods for sentiment analysis on tweets (a binary classification problem). The training dataset is expected to be a csv file in the following format: 

`tweet_id,sentiment,tweet`

where the tweet_id is a unique integer identifying the tweet, sentiment is either 1 (positive) or 0 (negative), and tweet is the tweet enclosed in quotes.

Similarly, the test dataset is a csv file in the following format: 

`tweet_id,tweet`

Please note that csv headers are not expected and should be removed from the training and test datasets. The datasets are provided in the dataset/ directory.

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
1. Tweets distribution: This visualization shows the distribution of positive and negative tweets in the dataset.
<img src="https://github.com/vaishanth-rmrj/Twitter-Sentiment-Analysis-Using-Different-Models/blob/master/extras/tweet_distribution.png" width=500/>

2. Tweet words frequency distribution: This visualization shows the frequency of words used in the tweets in the dataset.
<img src="https://github.com/vaishanth-rmrj/Twitter-Sentiment-Analysis-Using-Different-Models/blob/master/extras/word_freq.png" width=500/>

3. Word Cloud generation: This visualization shows the most frequent words used in the tweets in the form of a word cloud.
<img src="https://github.com/vaishanth-rmrj/Twitter-Sentiment-Analysis-Using-Different-Models/blob/master/extras/word_cloud.png" width=500/>

4. Positive Hashtag Frequency: This visualization shows the frequency of hashtags used in positive tweets.
<img src="https://github.com/vaishanth-rmrj/Twitter-Sentiment-Analysis-Using-Different-Models/blob/master/extras/pos_hastag.png" width=500/>
          
5. Negative Hashtag Frequency: This visualization shows the frequency of hashtags used in negative tweets.
<img src="https://github.com/vaishanth-rmrj/Twitter-Sentiment-Analysis-Using-Different-Models/blob/master/extras/neg_hastag.png" width=500/>

## Conclusion

The goal of this project is to perform sentiment analysis on tweets and compare the performance of various different methods. The datasets and visualizations generated in this project can be used as a starting point for further research and development in the field of sentiment analysis.


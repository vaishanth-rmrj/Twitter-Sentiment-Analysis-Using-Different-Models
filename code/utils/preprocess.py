import re
import sys
from nltk.stem.porter import PorterStemmer
import os
import pandas as pd

from nltk import FreqDist
import pickle
import sys
from collections import Counter

class TextPreprocessor:
  def __init__(self, output_dir_path, use_stemmer=False):
    self.output_dir_path = output_dir_path
    self.use_stemmer = use_stemmer
    self.porter_stemmer = PorterStemmer()

  def preprocess_txt(self, input_file_path, is_test=False):

    input_file_name = input_file_path.split("/")[-1].split('.')[0]
    output_file_name = input_file_name + "-processed.csv"

    # reading as panda df
    print("Reading datasets ...")
    df = pd.read_csv(input_file_path) 
    df = df.dropna()
    print("Preprocessing datasets ...")
    df['tweet'] = df['tweet'].apply(self.preprocess_tweet)
    df = df[pd.notnull(df['tweet'])]

    if is_test:
      # df = df.drop(['label'], axis=1)
      df.to_csv(os.path.join(self.output_dir_path, output_file_name), index=False)
    else:
      df.to_csv(os.path.join(self.output_dir_path, output_file_name), index=False)
    
    print("Preprocess datasets saved.")

  def preprocess_tweet(self, tweet):

    processed_tweet = []
    # Convert to lower case
    tweet = tweet.lower()
    # Replaces URLs with the word URL
    tweet = re.sub(r'((www\.[\S]+)|(https?://[\S]+))', ' URL ', tweet)
    # Replace @handle with the word USER_MENTION
    tweet = re.sub(r'@[\S]+', 'USER_MENTION', tweet)
    # Replaces #hashtag with hashtag
    tweet = re.sub(r'#(\S+)', r' \1 ', tweet)
    # Remove RT (retweet)
    tweet = re.sub(r'\brt\b', '', tweet)
    # Replace 2+ dots with space
    tweet = re.sub(r'\.{2,}', ' ', tweet)
    # Strip space, " and ' from tweet
    tweet = tweet.strip(' "\'')
    # Replace emojis with either EMO_POS or EMO_NEG
    tweet = self.handle_emojis(tweet)
    # Replace multiple spaces with a single space
    tweet = re.sub(r'\s+', ' ', tweet)
    words = tweet.split()

    for word in words:
        word = self.preprocess_word(word)
        if self.is_valid_word(word):
            if self.use_stemmer:
                word = str(self.porter_stemmer.stem(word))
            processed_tweet.append(word)

    return ' '.join(processed_tweet)     

  def preprocess_word(self, word):
    # Remove punctuation
    word = word.strip('\'"?!,.():;')
    # Convert more than 2 letter repetitions to 2 letter
    # funnnnny --> funny
    word = re.sub(r'(.)\1+', r'\1\1', word)
    # Remove - & '
    word = re.sub(r'(-|\')', '', word)
    return word


  def is_valid_word(self, word):
      # Check if word begins with an alphabet
      return (re.search(r'^[a-zA-Z][a-z0-9A-Z\._]*$', word) is not None)


  def handle_emojis(self, tweet):
      # Smile -- :), : ), :-), (:, ( :, (-:, :')
      tweet = re.sub(r'(:\s?\)|:-\)|\(\s?:|\(-:|:\'\))', ' EMO_POS ', tweet)
      # Laugh -- :D, : D, :-D, xD, x-D, XD, X-D
      tweet = re.sub(r'(:\s?D|:-D|x-?D|X-?D)', ' EMO_POS ', tweet)
      # Love -- <3, :*
      tweet = re.sub(r'(<3|:\*)', ' EMO_POS ', tweet)
      # Wink -- ;-), ;), ;-D, ;D, (;,  (-;
      tweet = re.sub(r'(;-?\)|;-?D|\(-?;)', ' EMO_POS ', tweet)
      # Sad -- :-(, : (, :(, ):, )-:
      tweet = re.sub(r'(:\s?\(|:-\(|\)\s?:|\)-:)', ' EMO_NEG ', tweet)
      # Cry -- :,(, :'(, :"(
      tweet = re.sub(r'(:,\(|:\'\(|:"\()', ' EMO_NEG ', tweet)
      return tweet  


class StatsPreprocessor:
  def __init__(self, output_dir_path):
    self.output_dir_path = output_dir_path
    self.num_tweets, self.num_pos_tweets, self.num_neg_tweets = 0, 0, 0
    self.num_mentions, self.max_mentions = 0, 0
    self.num_emojis, self.num_pos_emojis, self.num_neg_emojis, self.max_emojis = 0, 0, 0, 0
    self.num_urls, self.max_urls = 0, 0
    self.num_words, self.num_unique_words, self.min_words, self.max_words = 0, 0, 1e6, 0
    self.num_bigrams, num_unique_bigrams = 0, 0
    self.all_words = []
    self.all_bigrams = []

  def generate_stats(self, input_file_path):
    print("Reading datasets ...")
    df = pd.read_csv(input_file_path) 
    print("Preprocessing datasets stats ...")
    self.num_tweets = len(df['tweet'])

    for i, tweet in enumerate(df['tweet']):
      if type(tweet) == str:
        if df['label'][i] == 1:
          self.num_pos_tweets += 1
        else:
          self.num_neg_tweets += 1
      
        result, words, bigrams = self.analyze_tweet(tweet)
        # mentions stats
        self.num_mentions += result['MENTIONS']
        self.max_mentions = max(self.max_mentions, result['MENTIONS'])
        
        # emoji stats
        self.num_pos_emojis += result['POS_EMOS']
        self.num_neg_emojis += result['NEG_EMOS']
        self.max_emojis = max(self.max_emojis, result['POS_EMOS'] + result['NEG_EMOS'])
        
        #  url stats
        self.num_urls += result['URLS']
        self.max_urls = max(self.max_urls, result['URLS'])
        
        # word stats
        self.num_words += result['WORDS']
        self.min_words = min(self.min_words, result['WORDS'])
        self.max_words = max(self.max_words, result['WORDS'])
        self.all_words.extend(words)
        self.num_bigrams += result['BIGRAMS']
        self.all_bigrams.extend(bigrams)

    self.num_emojis = self.num_pos_emojis + self.num_neg_emojis
    unique_words = list(set(self.all_words))
    with open(self.output_dir_path + 'unique.txt', 'w') as file:
        file.write('\n'.join(unique_words))
    self.num_unique_words = len(unique_words)

    self.num_unique_bigrams = len(set(self.all_bigrams))

    # Unigrams
    print('Calculating frequency distribution')  
    freq_dist = FreqDist(self.all_words)
    pkl_file_name = self.output_dir_path + 'freqdist.pkl'
    with open(pkl_file_name, 'wb') as pkl_file:
        pickle.dump(freq_dist, pkl_file)
    print('Saved uni-frequency distribution to: ', pkl_file_name)

    # Bigrams
    bigram_freq_dist = self.get_bigram_freqdist(self.all_bigrams)
    bi_pkl_file_name = self.output_dir_path + 'freqdist-bi.pkl'
    with open(bi_pkl_file_name, 'wb') as pkl_file:
        pickle.dump(bigram_freq_dist, pkl_file)
    print('Saved bi-frequency distribution to: ', bi_pkl_file_name)

  def print_stats(self):
    print('[Analysis Statistics]')
    print('Tweets => Total: {}, Positive: {}, Negative: {}'.format(self.num_tweets, self.num_pos_tweets, self.num_neg_tweets))
    print('User Mentions => Total: {}, Avg: {}, Max: {}' .format(self.num_mentions, self.num_mentions / float(self.num_tweets), self.max_mentions))
    print('URLs => Total: {}, Avg: {}, Max: {}' .format(self.num_urls, self.num_urls / float(self.num_tweets), self.max_urls))
    print('Emojis => Total: {}, Positive: {}, Negative: {}, Avg: {}, Max: {}' .format(self.num_emojis, self.num_pos_emojis, self.num_neg_emojis, self.num_emojis / float(self.num_tweets), self.max_emojis))
    print('Words => Total: {}, Unique: {}, Avg: {}, Max: {}, Min: {}' .format(self.num_words, self.num_unique_words, self.num_words / float(self.num_tweets), self.max_words, self.min_words))
    print('Bigrams => Total: {}, Unique: {}, Avg: {}' .format(self.num_bigrams, self.num_unique_bigrams, self.num_bigrams / float(self.num_tweets)))
   
  def analyze_tweet(self, tweet):
    result = {}
    result['MENTIONS'] = tweet.count('USER_MENTION')
    result['URLS'] = tweet.count('URL')
    result['POS_EMOS'] = tweet.count('EMO_POS')
    result['NEG_EMOS'] = tweet.count('EMO_NEG')
    tweet = tweet.replace('USER_MENTION', '').replace(
        'URL', '')
    words = tweet.split()
    result['WORDS'] = len(words)
    bigrams = self.get_bigrams(words)
    result['BIGRAMS'] = len(bigrams)
    return result, words, bigrams


  def get_bigrams(self, tweet_words):
      bigrams = []
      num_words = len(tweet_words)
      for i in range(num_words - 1):
          bigrams.append((tweet_words[i], tweet_words[i + 1]))
      return bigrams


  def get_bigram_freqdist(self, bigrams):
      freq_dict = {}
      for bigram in bigrams:
          if freq_dict.get(bigram):
              freq_dict[bigram] += 1
          else:
              freq_dict[bigram] = 1
      counter = Counter(freq_dict)
      return counter

  






  





from keras.models import Sequential, load_model
from keras.layers import Dense
import sys
from pandas.core.common import random_state
from utils.utils import top_n_words, top_n_bigrams
import random
import numpy as np
import pandas as pd

class NeuralNetClassifier:
  def __init__(self, freq_dist_file, bi_freq_dist_file, use_bigram=False):
    # params
    self.unigram_size = 15000
    self.vocab_size = self.unigram_size
    self.use_unigram = False
    self.use_bigram = use_bigram
    if self.use_bigram:
        self.bigram_size = 10000
        self.vocab_size = self.unigram_size + self.bigram_size
    self.feat_type = 'frequency'

    # variables
    self.unigrams = top_n_words(freq_dist_file, self.unigram_size)
    if self.use_bigram:
      self.bigrams = top_n_bigrams(bi_freq_dist_file, self.bigram_size)

    # Neural net model
    self.model = Sequential()
    self.model.add(Dense(500, input_dim=self.vocab_size, activation='sigmoid'))
    self.model.add(Dense(1, activation='sigmoid'))
    self.model.compile(loss='binary_crossentropy',
                  optimizer='adam', metrics=['accuracy'])


  def fit(self, data_file, batch_size=64, epochs=2):
    print("Processing tweets ...")
    df = self.process_tweets(data_file)

    print("Splitting data for train and val ...")
    train_df = df.sample(frac=0.9, random_state=42)
    val_df = df.drop(train_df.index)

    del df

    num_train_batches = int(np.ceil(len(train_df) / float(batch_size)))
    best_val_acc = 0.0

    for epoch in range(epochs):
      print("Training | Epoch: ", epoch)
      i = 0

      for training_set_X, training_set_y in self.generate_embeddings(train_df, feat_type=self.feat_type, batch_size=batch_size, is_test=False):
        hist = self.model.train_on_batch(training_set_X, training_set_y)
        if i % 100 == 0:
          print('Iteration {}/{}, loss:{}, acc:{}'.format(i, num_train_batches, hist[0], hist[1]))
        i += 1

      val_acc = self.evaluate_model(self.model, val_df)
      print('Epoch: {}, val_acc:{}'.format(epoch + 1, val_acc))

      if val_acc > best_val_acc:
          print('Accuracy improved from %.4f to %.4f, saving model' % (best_val_acc, val_acc))
          best_val_acc = val_acc
          self.model.save('best_model.h5')
      
  
  def evaluate_model(self, model, val_df):
    correct, total = 0, len(val_df)
    for val_set_X, val_set_y in self.generate_embeddings(val_df, feat_type=self.feat_type, is_test=False):
        prediction = model.predict_on_batch(val_set_X)
        prediction = np.round(prediction)
        correct += np.sum(prediction == val_set_y[:, None])
    return float(correct) / total


  def predict(self, data_file):
    pass

  def process_tweets(self, data_file, is_test=False):
    print("Generating features")
    df = pd.read_csv(data_file)
    df = df.dropna()
    df['tweet'] = df['tweet'].apply(self.get_feature_vect)
    return df
  
  def get_feature_vect(self, tweet):
    if type(tweet) != str:
      return tweet
      
    uni_gram_feat_vect = []
    bi_gram_feat_vect = []
    words = tweet.split()
    for i, word in enumerate(words):
      
      if word in self.unigrams:
        uni_gram_feat_vect.append(word)
      
      if self.use_bigram and i < len(words)-1:
        next_word = words[i+1]

        if (word, next_word) in self.bigrams:
          bi_gram_feat_vect.append((word, next_word))
    
    return uni_gram_feat_vect, bi_gram_feat_vect

  def generate_embeddings(self, df, batch_size=500, is_test=False, feat_type='presence'):
    num_batches = int(np.ceil(len(df) / float(batch_size)))
    feature_batches = []
    label_batches = []

    # generating batches
    for i in range(num_batches):
        # batch = df.iloc[i * batch_size: (i + 1) * batch_size]
        batch = df.sample(n=batch_size)

        # initializinf feature and label vects
        features = np.zeros((batch_size, self.vocab_size ))
        labels = np.zeros(batch_size)

        # looping thru tweets in this batch
        for j in range(batch_size):

          if is_test:
            tweet_words = batch.iloc[j][1][0]
            tweet_bigrams = batch.iloc[j][1][1]
          else:
            tweet_words = batch.iloc[j][2][0]
            tweet_bigrams = batch.iloc[j][2][1]
            labels[j] = batch.iloc[j][1]

          if feat_type == 'presence':
            tweet_words = set(tweet_words)
            tweet_bigrams = set(tweet_bigrams)

          # converting words to embeddings
          for word in tweet_words:
            idx = self.unigrams.get(word)
            if idx:
                features[j, idx] += 1

          if self.use_bigram:
            for bigram in tweet_bigrams:
              idx = self.bigrams.get(bigram)
              if idx:
                features[j, self.unigram_size + idx] += 1


        yield features, labels

















import pandas as pd

class WordBasedClassifier:
  def __init__(self, pos_word_file, neg_word_file):
    self.pos_words = self.convert_to_wordset(pos_word_file)
    self.neg_words = self.convert_to_wordset(neg_word_file)
    self.predictions = []

  def convert_to_wordset(self, filename):
    ''' 
    convert file with words into a set of words
    '''
    words = []
    with open(filename, 'r', encoding="ISO-8859-1") as f:        
        for line in f:
            words.append(line.strip())
    return set(words)

  def predict(self, file_path, is_test=False):
    df = pd.read_csv(file_path)
    for tweet in df['tweet']:

      if type(tweet) == str:
        pos_count, neg_count = 0, 0
        for word in tweet.split(' '):
          if word in self.pos_words:
            pos_count += 1
          elif word in self.neg_words:
            neg_count += 1

        pred = 1 if pos_count >= neg_count else 0
        self.predictions.append(pred)
      else:
        self.predictions.append(0)

    
    if not is_test:
      correct_pred = sum(df['label']==self.predictions)
      print("Correct predictions: {}/{}".format(correct_pred, len(df['label'])))
      print("Training accuracy: ", correct_pred/len(df['label']))
    
    return self.predictions     



  
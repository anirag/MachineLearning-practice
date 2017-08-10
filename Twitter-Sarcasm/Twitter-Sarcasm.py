
import os
from scipy.sparse import csc_matrix
import sys
import operator
import numpy as np
import pandas as pd
from scipy import sparse
import xgboost as xgb
import string
import sklearn
from sklearn import model_selection, preprocessing, ensemble
from sklearn.metrics import log_loss
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from textblob import TextBlob


# Function to calculate the f-score for the prediction
def evalerror(preds, dtrain):
    labels = dtrain.get_label()
    y_bin = [1. if y_cont > 0.5 else 0. for y_cont in preds] # binaryzing your output
    # return a pair metric_name, result
    # since preds are margin(before logistic transformation, cutoff at 0)
    return 'error', sklearn.metrics.f1_score(labels, y_bin, average='weighted')

# Function to get the number of vowels from a text
def check_vowels(sentence):
    counter=0
    for word in sentence.split():
        if set('aeiou').isdisjoint(word.lower()):
            counter+=1
    return counter


# Function to get the number of capitalised words from a text
def cap_feature(sentence):
    counter = 0
    threshold = 3
    for j in range(len(sentence)):
        counter+=int(sentence[j].isupper())
    return int(counter>=threshold)

# Function to get the number of common words between two texts
def str_common_word(str1, str2):
    return sum(int(str2.find(word)>=0) for word in str1)


# Extract the position of positive words in a text
def pos_feature(sentence):
    
    sentence_pos = exp_replace.replace_emo(sentence)
    tokens = nltk.word_tokenize(sentence_pos)
    tokens = [(t.lower()) for t in tokens] 
    pos_vector = sentiments.posvector(tokens)
    for j in range(len(pos_vector)):
        features['POS' + str(j+1)] = pos_vector[j]



# Functions to extract ngrams from a text
def getUnigram(words):

    assert type(words) == list
    return words



def getBigram(words, join_string, skip=0):

    assert type(words) == list
    L = len(words)
    if L > 1:
        lst = []
        for i in range(L-1):
            for k in range(1,skip+2):
                if i+k < L:
                    lst.append( join_string.join([words[i], words[i+k]]) )
    else:
        # set it as unigram
        lst = getUnigram(words)
    return lst



def getTrigram(words, join_string, skip=0):
    """
       Input: a list of words, e.g., ['I', 'am', 'Denny']
        Output: a list of trigram, e.g., ['I_am_Denny']
        I use _ as join_string for this example.
    """
    assert type(words) == list
    L = len(words)
    if L > 2:
        lst = []
        for i in range(L-2):
            for k1 in range(1,skip+2):
                for k2 in range(1,skip+2):
                    if i+k1 < L and i+k1+k2 < L:
                        lst.append( join_string.join([words[i], words[i+k1], words[i+k1+k2]]) )
    else:
        # set it as bigram
        lst = getBigram(words, join_string, skip)
    return lst
    


# xgboost alogorithm for cross-validation and prediction
def runXGB(train_X, train_y, test_X, test_y=None, feature_names=None, seed_val=0, num_rounds=2500):
    param = {}
    param['objective'] = 'binary:logistic'
    param['eta'] = 0.05
    param['max_depth'] = 6
    param['silent'] = 1
    #param['eval_metric'] = "auc"
    param['min_child_weight'] = 1
    param['subsample'] = 0.7
    param['colsample_bytree'] = 0.7
    param['seed'] = seed_val
    num_rounds = num_rounds

    plst = list(param.items())
    xgtrain = xgb.DMatrix(train_X, label=train_y)

    if test_y is not None:
        xgtest = xgb.DMatrix(test_X, label=test_y)
        watchlist = [ (xgtrain,'train'), (xgtest, 'test') ]
        model = xgb.train(plst, xgtrain, num_rounds, watchlist,feval=evalerror,maximize=True,early_stopping_rounds=20)
    else:
        xgtest = xgb.DMatrix(test_X)
        model = xgb.train(plst, xgtrain, num_rounds,feval=evalerror,maximize=True)

    pred_test_y = model.predict(xgtest)
    return pred_test_y, model


# read in the dataset
data_path = "/Users/Twitter-Sarcasm/"
train_file = data_path + "train.csv"
test_file = data_path + "test.csv"
train_df = pd.read_csv(train_file)
test_df = pd.read_csv(test_file)
print(train_df.shape)
print(test_df.shape)


train_df['label'].value_counts()
ntrain = train_df.shape[0]
df_all = train_df.append(test_df)
df_all.head(5)

# get the list of positive and negative words
with open('/Users/opinion-lexicon-English/positive-words.txt',encoding = "ISO-8859-1") as f:
    positive_words_list = f.read().splitlines()
with open('/Users/opinion-lexicon-English/negative-words.txt',encoding = "ISO-8859-1") as f:
    negative_words_list = f.read().splitlines()

#Feature Creation
df_all['Caps'] = df_all['tweet'].apply(lambda x: cap_feature(x))
df_all['tweet'] = df_all['tweet'].apply(lambda x:x.lower())
df_all.head(5)
df_all['tweet'] = df_all['tweet'].apply(lambda x: x.replace('b\'', ''))
df_all['tweet'] = df_all['tweet'].apply(lambda x: x.replace('b"', ''))
df_all.head(5)

# Extract cleaned text from a tweet
import preprocessor as p
p.set_options(p.OPT.URL, p.OPT.EMOJI,p.OPT.MENTION,p.OPT.RESERVED,p.OPT.SMILEY)
df_all['tweet'] = df_all['tweet'].apply(lambda x: p.clean(x))
df_all.head(5)

# Manual replacement of few shorthands
app = {"'s":" is","'re":" are","'m":" am","n't":"nt"}
def replace_all(text, dic):
    for i, j in dic.items():
        text = text.replace(i, j)
    return text
     
df_all['tweet'] = df_all['tweet'].apply(lambda x: replace_all(x,app))
df_all.head(5)


df_all['ind1'] = df_all['tweet'].apply(lambda x: int("#sarca" in x))
df_all['ind2'] = df_all['tweet'].apply(lambda x: int("#not" in x))
df_all['ind3'] = df_all['tweet'].apply(lambda x: int("#serious" in x))


punctuations = '''!()-[]{};:'"\,<>./?#$@%^&*_~'''
replace_punctuation = str.maketrans(punctuations,' '*len(punctuations))
df_all['tweet'] = df_all['tweet'].apply(lambda x: x.translate(replace_punctuation))
df_all.head(5)

df_all['tweet'] = df_all['tweet'].apply(lambda x: " ".join(x.split()));
df_all['tweet'] = df_all['tweet'].apply(lambda x: x.replace("lots of laughs or laughing out loud","lol"));
df_all['tweet_length'] = df_all['tweet'].apply(lambda x: len(x.split(" ")))
df_all['tweet_chars'] = df_all['tweet'].apply(lambda x: len("".join(x)))
df_all['polarity'] = df_all.apply(lambda x: TextBlob(x['tweet']).sentiment.polarity, axis=1)
df_all['subjectivity'] = df_all.apply(lambda x: TextBlob(x['tweet']).sentiment.subjectivity, axis=1)
df_all['np_phrases'] = df_all.apply(lambda x: len(TextBlob(x['tweet']).noun_phrases), axis=1)
df_all['num_positive'] = df_all['tweet'].map(lambda x:str_common_word(positive_words_list,x))
df_all['num_negative'] = df_all['tweet'].map(lambda x:str_common_word(negative_words_list,x))
df_all['positive_ratio'] = 1. * df_all['num_positive']/df_all['tweet_length']
df_all['negative_ratio'] = 1. * df_all['num_negative']/df_all['tweet_length']




df_all['no_vowel'] = df_all['tweet'].apply(lambda x: check_vowels(x))

df_all["tweet_bigram"] = df_all.apply(lambda x: getBigram(x["tweet"].split(), "_"), axis=1)
df_all["tweet_trigram"] = df_all.apply(lambda x: getTrigram(x["tweet"].split(), "_"), axis=1)
df_all["tweet_bigram"] = df_all["tweet_bigram"].apply(lambda x: " ".join(x))
df_all["tweet_trigram"] = df_all["tweet_trigram"].apply(lambda x: " ".join(x))

train_df = df_all.iloc[:ntrain].reset_index(drop = True)
test_df = df_all.iloc[ntrain:].reset_index(drop = True)

tweet_id = test_df.ID.values
train_df.drop('ID', axis=1, inplace=True)
test_df.drop('ID', axis=1, inplace=True)
features_to_use  = ["ind1","ind2","ind3","tweet_length","tweet_chars","polarity","subjectivity","num_positive","num_negative","Caps","np_phrases","positive_ratio","negative_ratio","no_vowel"]

print(train_df["tweet"].head())

tfidf = CountVectorizer(stop_words='english', max_features=200,ngram_range=(1, 1))
tr_sparse = tfidf.fit_transform(train_df["tweet"])
te_sparse = tfidf.transform(test_df["tweet"])


print(train_df["tweet_bigram"].head())

tfidf_ = CountVectorizer(max_features=200)
tr_sparse_ = tfidf_.fit_transform(train_df["tweet_bigram"])
te_sparse_ = tfidf_.transform(test_df["tweet_bigram"])


print(train_df["tweet_trigram"].head())

tfidf__ = CountVectorizer(max_features=200)
tr_sparse__ = tfidf__.fit_transform(train_df["tweet_trigram"])
te_sparse__ = tfidf__.transform(test_df["tweet_trigram"])


del df_all

# Stack tfidf features with other features
train_X = sparse.hstack([train_df[features_to_use], tr_sparse,tr_sparse_,tr_sparse__]).tocsr()
test_X = sparse.hstack([test_df[features_to_use], te_sparse,te_sparse_,te_sparse__]).tocsr()
target_num_map = {'non-sarcastic':0, 'sarcastic':1}
train_y = np.array(train_df['label'].apply(lambda x: target_num_map[x]))
print(train_X.shape, test_X.shape)

# Cross-Validation
cv_scores = []
kf = model_selection.KFold(n_splits=5, shuffle=True, random_state=2016)
for dev_index, val_index in kf.split(range(train_X.shape[0])):
        dev_X, val_X = train_X[dev_index,:], train_X[val_index,:]
        dev_y, val_y = train_y[dev_index], train_y[val_index]
        preds, model = runXGB(dev_X, dev_y, val_X, val_y)
        y_bin = [1. if y_cont > 0.5 else 0. for y_cont in preds] # binaryzing your output
        cv_scores.append(sklearn.metrics.f1_score(val_y, y_bin, average='weighted') )
        break

# Prdeiction on test file
preds, model = runXGB(train_X, train_y, test_X, num_rounds=24)
y_bin = [1. if y_cont > 0.55 else 0. for y_cont in preds] # binaryzing your output
out_df = pd.DataFrame(y_bin)
out_df.columns = ["label"]
target_num_map = {0:'non-sarcastic', 1:'sarcastic'}
out_df["ID"] = tweet_id
out_df['label'] = out_df['label'].apply(lambda x: target_num_map[x])
out_df = out_df[['ID','label']]
out_df.head(10) 
out_df.to_csv(data_path +"xgb.csv", index=False)







import re
import os
import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
import seaborn as sns
import string
import nltk
import math
from joblib import dump, load
from nltk.stem.porter import *
from nltk.corpus import stopwords 
from nltk.tokenize import word_tokenize 
import warnings 
from wordcloud import WordCloud
from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from helper_functions import *


warnings.filterwarnings("ignore", category=DeprecationWarning)
s = prompt_classifier()

train  = clean_tweets(pd.read_csv('train_tweets.csv'))
test = clean_tweets(pd.read_csv('test_tweets.csv'))
new = clean_tweets(load("tweets_labeled.csv"))
new2 = clean_tweets(load("tweets_labeled2.csv"))
train = train.append(new)
train = train.append(new2)

#analyse_tags(train)

tfidf_vectorizer = TfidfVectorizer(max_df=0.90, min_df=2, max_features=1000, stop_words='english')
# TF-IDF feature matrix

train_tfidf = tfidf_vectorizer.fit_transform(train['tidy_tweet'])

train_label = train['label']

# splitting data into training and validation set
xtrain_tfidf, xvalid_tfidf, ytrain, yvalid = train_test_split(train_tfidf, train_label, random_state=42, test_size=0.3)

if s == 1:
    text_classifier = LogisticRegression()
    filename = 'logisticregression_model.joblib'
elif s == 2:
    text_classifier = RandomForestClassifier(n_estimators=200, random_state=0)
    filename = 'randomforest_model.joblib'
elif s == 3:
    text_classifier = SVC(kernel='poly', degree=8)
    filename = "svcpolynomial_model.joblib"
elif s == 4:
    text_classifier = SVC(kernel='rbf')
    filename = "svcgaussian_model.joblib"
elif s == 5:
    text_classifier = SVC(kernel='sigmoid')
    filename = "svcsigmoid_model.joblib"
elif s == 6:
    text_classifier = KNeighborsClassifier(n_neighbors=4)
    filename = "kneighbors_model.joblib"

text_classifier.fit(xtrain_tfidf, ytrain)

if os.path.isfile(os.getcwd()+'/' + filename):
    os.remove(os.getcwd()+'/' + filename) 
dump(text_classifier, filename)
if os.path.isfile(os.getcwd()+'/' + "tfidf.pkl"):
    os.remove(os.getcwd()+'/' + "tfidf.pkl") 
dump(tfidf_vectorizer, open("tfidf.pkl", "wb"))

print("Model saved in file " + filename + '\n')

if s == 1:
    prediction = text_classifier.predict_proba(xvalid_tfidf)
    prediction_int = prediction[:,1] >= 0.3
    predictions = prediction_int.astype(np.int)
else:
    predictions = text_classifier.predict(xvalid_tfidf)

print(confusion_matrix(yvalid,predictions))
print(classification_report(yvalid,predictions))
#print(f1_score(yvalid, predictions))
#print(accuracy_score(yvalid, predictions))

#test_tfidf = tfidf_vectorizer.fit_transform(test['tidy_tweet'])
#predictions = text_classifier.predict(test_tfidf)
#dump(text_classifier,'randomforest_sentiment_model.joblib')

'''

error = []

# Calculating error for K values between 1 and 40
for i in range(1, 40):
    knn = KNeighborsClassifier(n_neighbors=i)
    knn.fit(xtrain_tfidf, ytrain)
    pred_i = knn.predict(xvalid_tfidf)
    error.append(np.mean(pred_i != yvalid))


plt.figure(figsize=(12, 6))
plt.plot(range(1, 40), error, color='red', linestyle='dashed', marker='o',
         markerfacecolor='blue', markersize=10)
plt.title('Error Rate K Value')
plt.xlabel('K Value')
plt.ylabel('Mean Error')
plt.show()
'''


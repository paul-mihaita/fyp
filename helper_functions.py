import re
import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
import seaborn as sns
import string
import nltk
import math
from joblib import dump, load
from nltk.stem.porter import *
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

def make_wordmap(data,label):
    normal_words =' '.join([text for text in data['tidy_tweet'][data['label'] == label]])
    wordcloud = WordCloud(width=800, height=500, random_state=21,max_font_size=110).generate(normal_words)
    plt.figure(figsize=(10, 7))
    plt.imshow(wordcloud, interpolation="bilinear")
    plt.axis('off')
    plt.show()
def remove_pattern(input_txt, pattern):
    r = re.findall(pattern, input_txt)
    for i in r:
        input_txt = re.sub(i, '', input_txt)
        
    return input_txt    
def hashtag_extract(x):
    hashtags = []
    # Loop over the words in the tweet
    for i in x:
        ht = re.findall(r"#(\w+)", i)
        hashtags.append(ht)

    return hashtags
def plot_hashtags(l):
    a = nltk.FreqDist(l)
    d = pd.DataFrame({'Hashtag': list(a.keys()),'Count': list(a.values())})
    # selecting top 10 most frequent hashtags     
    d = d.nlargest(columns="Count", n = 10) 
    plt.figure(figsize=(16,5))
    ax = sns.barplot(data=d, x= "Hashtag", y = "Count")
    ax.set(ylabel = 'Count')
    plt.show()
def clean_tweets(toClean):
    # remove twitter handles (@user)
    toClean['tidy_tweet'] = np.vectorize(remove_pattern)(toClean['tweet'], "@[\w]*")
    # remove links
    toClean['tidy_tweet'] = toClean['tidy_tweet'].str.replace("(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)"," ")
    # remove special characters, numbers, punctuations
    toClean['tidy_tweet'] = toClean['tidy_tweet'].str.replace("[^a-zA-Z#]", " ")
    # removing short words
    toClean['tidy_tweet'] = toClean['tidy_tweet'].apply(lambda x: ' '.join([w for w in x.split() if len(w)>3]))

    tokenized_tweet = toClean['tidy_tweet'].apply(lambda x: x.split())

    stemmer = PorterStemmer()

    tokenized_tweet = tokenized_tweet.apply(lambda x: [stemmer.stem(i) for i in x]) # stemming

    for i in range(len(tokenized_tweet)):
        tokenized_tweet[i] = ' '.join(tokenized_tweet[i])

    toClean['tidy_tweet'] = tokenized_tweet
    return toClean
def loadModel():
    s = prompt_classifier()
    if s == 1:
        filename = 'logisticregression_model.joblib'
    elif s == 2:
        filename = 'randomforest_model.joblib'
    elif s == 3:
        filename = "svcpolynomial_model.joblib"
    elif s == 4:
        filename = "svcgaussian_model.joblib"
    elif s == 5:
        filename = "svcsigmoid_model.joblib"
    elif s == 6:
        filename = "kneighbors_model.joblib"
    return load(filename)
def prompt_classifier():
    s = 0
    while s<1 or s > 6:
        s = "Which technique to use for training?" + '\n' + "1. Logistic Regression" + '\n' + "2. Random Forest" + '\n' + "3. SVC , polynomial" + '\n' + "4. SVC, gaussian" + '\n' + "5. SVC, sigmoid" + '\n' + "6. K-nearest " + '\n' + "Please enter a number from 1 to 6" + '\n'
        try:
            s = int(input(s))
        except:
            s = 0
    return s
def analyse_tags(data):
    make_wordmap(data,0)
    make_wordmap(data,1)

    # extracting hashtags from non racist/sexist tweets

    HT_regular = hashtag_extract(data['tidy_tweet'][data['label'] == 0])

    # extracting hashtags from racist/sexist tweets
    HT_negative = hashtag_extract(data['tidy_tweet'][data['label'] == 1])

    # unnesting list
    HT_regular = sum(HT_regular,[])
    HT_negative = sum(HT_negative,[])

    plot_hashtags(HT_negative)
    plot_hashtags(HT_regular)

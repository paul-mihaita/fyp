# access token: 1088736062548164608-WNXuMGPWX6eeCTmQH4ZPimYuAMFIV6
# access token secret: kPuHJbpxL8LNpmt5azMxhX8zJ5xdSfTjz1f09J4EBXWUo
# API key: D47QJqNbG66KSJUw6pOEnieWw
# API secret key: lSq17uyRnnNVUVIwXmHgf43P1LQB7dX6horFt9Vqtz25bqCKIe
import re 
import tweepy 
from tweepy import OAuthHandler 
from textblob import TextBlob 
import os
import pandas as pd 
import numpy as np 
import string
import nltk
import math
from joblib import dump, load
from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from helper_functions import *
  
class TwitterClient(object): 
    ''' 
    Generic Twitter Class for sentiment analysis. 
    '''
    def __init__(self): 
        ''' 
        Class constructor or initialization method. 
        '''
        # keys and tokens from the Twitter Dev Console 
        consumer_key = 'D47QJqNbG66KSJUw6pOEnieWw'
        consumer_secret = 'lSq17uyRnnNVUVIwXmHgf43P1LQB7dX6horFt9Vqtz25bqCKIe'
        access_token = '1088736062548164608-WNXuMGPWX6eeCTmQH4ZPimYuAMFIV6'
        access_token_secret = 'kPuHJbpxL8LNpmt5azMxhX8zJ5xdSfTjz1f09J4EBXWUo'
  
        # attempt authentication 
        try: 
            # create OAuthHandler object 
            self.auth = OAuthHandler(consumer_key, consumer_secret) 
            # set access token and secret 
            self.auth.set_access_token(access_token, access_token_secret) 
            # create tweepy API object to fetch tweets 
            self.api = tweepy.API(self.auth) 
        except: 
            print("Error: Authentication Failed") 
  
    def clean_tweet(self, tweet): 
        ''' 
        Utility function to clean tweet text by removing links, special characters 
        using simple regex statements. 
        '''
        return ' '.join(re.sub("(@[A-Za-z0-9]+)|([^0-9A-Za-z \t]) |(\w+:\/\/\S+)", " ", tweet).split()) 
  
    def get_tweet_sentiment(self, tweet): 
        ''' 
        Utility function to classify sentiment of passed tweet 
        using textblob's sentiment method 
        '''
        # create TextBlob object of passed tweet text 
        analysis = TextBlob(self.clean_tweet(tweet)) 
        # set sentiment 
        if analysis.sentiment.polarity > 0: 
            return 'positive'
        elif analysis.sentiment.polarity == 0: 
            return 'neutral'
        else: 
            return 'negative'
  
    def get_tweets(self, query, count = 10): 
        ''' 
        Main function to fetch tweets and parse them. 
        '''
        # empty list to store parsed tweets 
        tweets = [] 
        fetched_tweets = {}
        fetched_tweets['tweet'] = []
        try: 
            # call twitter api to fetch tweets 
            fetched= self.api.search(q = query, count = count)
            for tweet in fetched:
                fetched_tweets['tweet'].append(tweet.text)

            df = pd.DataFrame(fetched_tweets, columns = ['tweet'])

            #fetched_tweets['tweet'] = fetched_tweets['tweet'].tolist()
            fetched_tweets = clean_tweets(df)
            text_classifier = load('randomforest_model.joblib')
            #tfidf_vectorizer = TfidfVectorizer(max_df=0.90, min_df=2, max_features=1000, stop_words='english')
            #loaded_vec = CountVectorizer(decode_error="replace",vocabulary=pickle.load(open("feature.pkl", "rb")))
            #tfidf = transformer.fit_transform(loaded_vec.fit_transform(np.array(["aaa ccc eee"])))
            tfidf_vectorizer = load("tfidf.pkl")

            test_tfidf = tfidf_vectorizer.transform(fetched_tweets['tidy_tweet'])
            predictions = text_classifier.predict(test_tfidf)
            i = 0
            # parsing tweets one by one
            print(len(predictions))
            print(len(fetched))
            print(len(fetched_tweets['tidy_tweet']))
            for original,tweet in zip(fetched,fetched_tweets['tidy_tweet']): 
                # empty dictionary to store required params of a tweet 
                parsed_tweet = {} 
  
                # saving text of tweet 
                parsed_tweet['text'] = original.text +'\n' +"-----------" +'\n'+ tweet + '\n'
                # saving sentiment of tweet 
                #parsed_tweet['sentiment'] = self.get_tweet_sentiment(tweet.text) 
                parsed_tweet['sentiment'] = predictions[i]
                i = i + 1
                # appending parsed tweet to tweets list 
                if original.retweet_count > 0: 
                    # if tweet has retweets, ensure that it is appended only once 
                    if parsed_tweet not in tweets: 
                        tweets.append(parsed_tweet) 
                else: 
                    tweets.append(parsed_tweet) 
            # return parsed tweets 
            return tweets 
  
        except tweepy.TweepError as e: 
            # print error (if any) 
            print("Error : " + str(e))


def main(): 
    # creating object of TwitterClient Class 
    api = TwitterClient() 
    # calling function to get tweets 
    word = input("Enter keyword for search:" + '\n')
    tweets = api.get_tweets(query = word , count = 300)

    # picking positive tweets from tweets 
    ptweets = [tweet for tweet in tweets if tweet['sentiment'] == 1] 
    # percentage of positive tweets 
    print("Positive tweets percentage: {} %".format(100*len(ptweets)/len(tweets))) 
    # picking negative tweets from tweets 
    ntweets = [tweet for tweet in tweets if tweet['sentiment'] == 0] 
    # percentage of negative tweets 
    print("Negative tweets percentage: {} %".format(100*len(ntweets)/len(tweets))) 
    # percentage of neutral tweets 
    print("Neutral tweets percentage: {} %".format(100*(len(tweets) - len(ntweets) - len(ptweets))/len(tweets)))
  
    # printing first 5 positive tweets 
    print("\n\nPositive tweets:") 
    for tweet in ptweets[:10]: 
        print(tweet['text']) 
  
    # printing first 5 negative tweets 
    print("\n\nNegative tweets:") 
    for tweet in ntweets[:10]: 
        print(tweet['text']) 

if __name__ == "__main__": 
    # calling main function 
    main() 

"""
################  Part 1  ###################
#  Scapping Data from Twitter using Tweepy  #
#############################################
"""
"""
Created on Wed Oct  7 12:28:55 2020

@author: shaoqianchen
"""
import csv
import tweepy
import ssl
import time
from datetime import datetime
#Twitter API account @chenshaoqian
consumer_key = "***************************"
consumer_secret = "***************************"
access_token = "***************************"
access_token_secret = "***************************"

def search_tag(consumer_key,consumer_secret,access_token,access_token_secret,tag,num_scrap,since_date,until_date):
    begin = time.time()
    auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
    auth.set_access_token(access_token, access_token_secret)
    api = tweepy.API(auth)
    api = tweepy.API(auth, wait_on_rate_limit=True)
    #get the name of the spreadsheet we will write to
    fname = tag+since_date
    #open the spreadsheet we will write to 
    with open('%s.csv' % (fname), 'w') as file:
        w = csv.writer(file)
        #write header row to spreadsheet
        w.writerow(['timestamp', 'tweet_text', 'username'])
        #for each tweet matching our hashtags, write relevant info to the spreadsheet
        for tweet in tweepy.Cursor(api.search, q=tag+' since:'+since_date+' until:'+until_date+' -filter:retweets', \
                                   lang="en",tweet_mode='extended').items(num_scrap):
            w.writerow([tweet.created_at, tweet.full_text.replace('\n',' ').encode('utf-8'),\
                        tweet.user.screen_name.encode('utf-8')])
    end = time.time()
    print("Finished scraping ",num_scrap, " tweets includes ",tag, " within ",end-begin," second")

candidate = ['@realDonaldTrump','@JoeBiden']

for i in candidate:    
    search_tag(consumer_key,consumer_secret,access_token,access_token_secret,i,5000,"2020-10-29","2020-10-30")




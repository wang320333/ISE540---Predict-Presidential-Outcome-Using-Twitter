
"""
Created on Wed Oct  7 12:28:55 2020

@author: ShaoqianChen, Jiahe Zhang
"""

#importing libraries
import numpy as np
from collections import Counter
import pandas as pd
import lightgbm as lgb
import pandas as pd
import numpy as np
import re
import nltk
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')
from nltk.corpus import stopwords
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.metrics import r2_score
from sklearn.metrics import f1_score
from nltk.stem import PorterStemmer
from nltk.stem import LancasterStemmer
from nltk.stem import WordNetLemmatizer 

from sklearn.datasets import load_breast_cancer,load_boston,load_wine
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import mean_squared_error,roc_auc_score,precision_score
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score


#Import annotated dataset for training
dfj = pd.read_csv('50K_Annotated.csv')

#Select X and y
dfjprocessed = dfj[['tweet_text','Description_num']]
X = dfjprocessed.iloc[:,:1].values
y = dfjprocessed.iloc[:,1:2].values


"""
#############################################
#           Preprocessing Tweets            #
#############################################
1. Word Tokenization
2. Stemming
3. Lemmatization
"""
processed_tweets = []
lancaster=LancasterStemmer()

for tweet in range(0, len(X)):  
    # Remove all the special characters
    processed_tweet = re.sub(r'\W', ' ', str(X[tweet]))
    # Remove URLs
    processed_tweet = re.sub(r"http\S+|www\S+|https\S+", '',processed_tweet, flags=re.MULTILINE)
    # remove all single characters
    processed_tweet = re.sub(r'\s+[a-zA-Z]\s+', ' ', processed_tweet)
    # Remove single characters from the start
    processed_tweet = re.sub(r'\^[a-zA-Z]\s+', ' ', processed_tweet) 
    # Removing prefixed 'b'
    processed_tweet = re.sub(r'^b\s+', '', processed_tweet)
    # Substituting multiple spaces with single space
    processed_tweet= re.sub(r'\s+', ' ', processed_tweet, flags=re.I)
    # Converting to Lowercase
    processed_tweet = processed_tweet.lower()
    processed_tweet = re.sub(r'\bx[a-z0-9]{1,2}\b',' ',processed_tweet)
    word_token = nltk.word_tokenize(processed_tweet)
    #Perform Stemming
    #processed_tweet = lancaster.stem(processed_tweet)
    ## stemmed_words = [ps.stem(w) for w in word_token]
    lemmatizer = WordNetLemmatizer()
    lemma_words = [lemmatizer.lemmatize(w, pos='n') for w in word_token]
    lemma_words = [lemmatizer.lemmatize(w, pos='v') for w in lemma_words]
    lemma_words = [lemmatizer.lemmatize(w, pos='a') for w in lemma_words]
    lemma_words = [lemmatizer.lemmatize(w, pos='r') for w in lemma_words]

    processed_tweet = " ".join(lemma_words)
    processed_tweets.append(processed_tweet)



"""
#############################################
#             Count  Vectorizer             #
#############################################
"""

from sklearn.feature_extraction.text import CountVectorizer
vec = CountVectorizer(stop_words=stopwords.words('english'),min_df=0.1,max_df=0.85)
vec.fit(processed_tweets)
X = vec.transform(processed_tweets)




"""
#############################################
#                Train Models               #
#############################################
"""



######################
#Light GBM Classifier#
######################
#Grid Search
from sklearn.model_selection import train_test_split  
XX_train,XX_test,yy_train,yy_test=train_test_split(X_cv,y,test_size=0.1,random_state=1,stratify = y)
clf = lgb.LGBMClassifier()

#Grid Search
from sklearn.model_selection import GridSearchCV
param_dist = {
        'max_bin':[10,50,100,150,255,300],
        'bagging_fraction':[0.005,0.01,0.1,0.2,0.5,0.8,1],
        'feature_fraction':[0.01,0.1,0.2,0.5,0.8,1],
        'save_binary':[True, False],
        'num_leaves':[20,50,100,150,200]
        }
grid = GridSearchCV(clf,param_dist,n_jobs = -1, cv = 10,scoring = 'accuracy')
grid.fit(XX_train,yy_train)
best_estimator = grid.best_estimator_
print(best_estimator)

#Train Light GBM
XX_train,XX_test,yy_train,yy_test=train_test_split(X,y,test_size=0.1,random_state=1,stratify = y)
clf = lgb.LGBMClassifier(max_bin=300, learning_rate=0.1,num_leaves=50)
clf.fit(XX_train, yy_train)
yy_pred=clf.predict(XX_test)
print(classification_report(yy_test,yy_pred))  
print(accuracy_score(yy_test,yy_pred))




#Using fitted model to predict unannotated data
dfj = pd.read_csv('5W_NO_annotation.csv')
dfjprocessed = dfj[['tweet_text']]
X = dfjprocessed.iloc[:,:1].values

processed_tweets = []
lancaster=LancasterStemmer()

for tweet in range(0, len(X)):  
    # Remove all the special characters
    processed_tweet = re.sub(r'\W', ' ', str(X[tweet]))
    # Remove URLs
    processed_tweet = re.sub(r"http\S+|www\S+|https\S+", '',processed_tweet, flags=re.MULTILINE)
    # remove all single characters
    processed_tweet = re.sub(r'\s+[a-zA-Z]\s+', ' ', processed_tweet)
    # Remove single characters from the start
    processed_tweet = re.sub(r'\^[a-zA-Z]\s+', ' ', processed_tweet) 
    # Removing prefixed 'b'
    processed_tweet = re.sub(r'^b\s+', '', processed_tweet)
    # Substituting multiple spaces with single space
    processed_tweet= re.sub(r'\s+', ' ', processed_tweet, flags=re.I)
    # Converting to Lowercase
    processed_tweet = processed_tweet.lower()
    processed_tweet = re.sub(r'\bx[a-z0-9]{1,2}\b',' ',processed_tweet)
    word_token = nltk.word_tokenize(processed_tweet)
    #Perform Stemming
    processed_tweet = lancaster.stem(processed_tweet)
    ## stemmed_words = [ps.stem(w) for w in word_token]
    lemmatizer = WordNetLemmatizer()
    lemma_words = [lemmatizer.lemmatize(w, pos='n') for w in word_token]
    lemma_words = [lemmatizer.lemmatize(w, pos='v') for w in lemma_words]
    lemma_words = [lemmatizer.lemmatize(w, pos='a') for w in lemma_words]
    lemma_words = [lemmatizer.lemmatize(w, pos='r') for w in lemma_words]

    processed_tweet = " ".join(lemma_words)
    processed_tweets.append(processed_tweet)

#count vectorizer
#################
X_cv2 = vec.transform(processed_tweets)

GBM5W_pred = clf.predict(X_cv2)
pred_5w = dfj
pred_5w["Description_num"] = GBM5W_pred
pred_5w.to_csv(r'Prediction_new5w(LGBM+CountVec).csv', index = False)
































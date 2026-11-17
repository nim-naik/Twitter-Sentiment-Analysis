# -*- coding: utf-8 -*-

! pip install kaggle

!mkdir -p ~/.kaggle
!cp kaggle.json ~/.kaggle/
!chmod 600 ~/.kaggle/kaggle.json

! kaggle datasets download -d kazanova/sentiment140

from zipfile import ZipFile
dataset = '/content/sentiment140.zip'
with ZipFile(dataset, 'r') as zip:
  zip.extractall()
  print('The dataset is extracted')

!pip install scikit-learn

!pip install --upgrade scikit-learn
import numpy as np
import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
nltk.download('wordnet')
nltk.download('stopwords')
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

import nltk
nltk.download('stopwords')

print(stopwords.words('english'))

"""Data processing"""

column_names=['target', 'id','date', 'flag', 'user', 'text']
twitter_df = pd.read_csv('/content/training.1600000.processed.noemoticon.csv',names=column_names, encoding='ISO-8859-1')

twitter_df.shape

twitter_df.isnull().sum()

"""Data distribution of target column"""

twitter_df['target'].value_counts()

twitter_df.replace({'target':{4:1}}, inplace=True)

def lemmatize(content):
  lemmatizer = WordNetLemmatizer()
  lemmatized_content= re.sub('[^a-zA-Z]',' ',content)
  stop_words = set(stopwords.words('english'))
  lemmatized_content = lemmatized_content.lower()
  lemmatized_content = lemmatized_content.split()
  lemmatized_content = [lemmatizer.lemmatize(word) for word in lemmatized_content if word not in stopwords.words('english')]
  lemmatized_content = ' '.join(lemmatized_content)
  return lemmatized_content

twitter_df['stemmed_content'] = twitter_df['text'].apply(lemmatize) #will run for 50 mins

twitter_df.head()

#separating the data and label
X = twitter_df['stemmed_content'].values
Y = twitter_df['target'].values

print(X)

print(Y)

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, stratify=Y, random_state=2)

vectorizer = TfidfVectorizer()
X_train = vectorizer.fit_transform(X_train)
X_test = vectorizer.transform(X_test)

#Training the logistic regression
model = LogisticRegression(max_iter=1000)

model.fit(X_train, Y_train)

X_train_prediction = model.predict(X_train)
training_data_accuracy=accuracy_score(Y_train, X_train_prediction)
print('Accuracy score on training data', training_data_accuracy)

X_test_prediction = model.predict(X_test)
testing_data_accuracy=accuracy_score(Y_test, X_test_prediction)
print('Accuracy score on testing data', testing_data_accuracy)

import pickle
filename='trained_model.sav'
pickle.dump(model,open(filename, 'wb'))

loaded_model = pickle.load(open('/content/trained_model.sav','rb'))

x_new = X_test[200]
print(Y_test[200])
prediction = loaded_model.predict(x_new)
print(prediction)
if(prediction[0] == 0):
  print("Negative tweet")
else:
  print("Positive tweet")

x_new = X_test[3]
print(Y_test[3])
prediction = loaded_model.predict(x_new)
print(prediction)
if(prediction[0] == 0):
  print("Negative tweet")
else:
  print("Positive tweet")

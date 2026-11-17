🐦 Twitter Sentiment Analysis

📌 Overview

This project applies Natural Language Processing (NLP) and machine learning techniques to classify the sentiments of tweets as either positive or negative. By analyzing large-scale social media text data, this model helps automatically gauge public opinion and emotional tone from raw tweet content.

🔧 Features

Automated dataset download and extraction via the Kaggle API.

Text preprocessing and cleaning (regex filtering, lowercasing, stopword removal, and WordNet lemmatization).

Target label encoding.

Feature extraction using TF-IDF Vectorization.

Logistic Regression model training and evaluation.

Model serialization for saving, loading, and running inference on new tweets.

🚀 How to Run

Clone or download the repository.

Ensure required libraries are installed:

pip install numpy pandas nltk scikit-learn kaggle.
Set up your Kaggle API credentials (kaggle.json in ~/.kaggle/) or download the dataset manually.

Run the script.

python twitter_sentiment_analysis_using_ml.py


Note: Lemmatizing around 1.6 million tweets takes roughly 50 minutes depending on your hardware.

📂 Data Used

Dataset: Sentiment140 (kazanova/sentiment140) sourced via Kaggle.

File Used: training.1600000.processed.noemoticon.csv

Dataset Size: 1.6 million tweets (balanced evenly between positive and negative sentiments).

Columns Included in the CSV:

target: The polarity of the tweet (0 = Negative, 4 = Positive, which is mapped to 1 during preprocessing).

id: The unique ID of the tweet.

date: The exact date and time the tweet was posted.

flag: Query flag (if no query was used, this is NO_QUERY).

user: The Twitter handle/username of the person who posted the tweet.

text: The actual text content/body of the tweet.

📈 Model Summary

Model Used: Logistic Regression

Text Vectorizer: TfidfVectorizer

Training-Test Split: 80-20 

Target Variable: target (0 = Negative tweet, 1 = Positive tweet)

Note: As per existing dataset value of target is 4 for positive tweet, but it is replaced wit 1 to simplify binary classification.

Evaluation Metrics:

Training & Testing Accuracy Score like accuracy_score.

Conclusion: TF-IDF combined with Logistic Regression provides a fast, scalable baseline for binary sentiment classification on large tweet corpora, with room for further improvement using n-grams, hyperparameter tuning, or deep learning models (e.g., LSTMs/Transformers).

🛠️ Libraries & Requirements

Python Version: Python 3.7+

Required Libraries & Tools:

pandas & numpy: Used for data loading, manipulation, array handling, and dataframe operations.

nltk (Natural Language Toolkit): Used for Natural Language Processing, specifically for removing stop words and the WordNetLemmatizer for text normalization.

scikit-learn: Used for splitting the dataset, turning text into numerical features, training the classification model, and evaluating performance.

kaggle: Used to download the dataset directly from Kaggle via the command line.

pickle: Built into Python standard library used for saving and loading the trained machine learning model.

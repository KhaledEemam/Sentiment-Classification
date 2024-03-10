# Introduction
This sentiment analysis project aims to analyze the sentiment of tweets using various machine learning and deep learning approaches. The dataset used in this project is sourced from Kaggle, containing tweets labeled from -1 to 1, where -1 represents negative sentiment, 0 represents neutral sentiment, and 1 represents positive sentiment.

# Approaches Used
## 1. Classical Machine Learning Models
* Utilized traditional machine learning algorithms such as logistic regression, random forest, SGD, and GradientBoosting.
* Features engineered from text data using TF-IDF
## 2. Deep Learning Approach with LSTM and Embedding
* Implemented a deep learning model by stacking Keras layers, such as Bidirectional LSTM and Dropout.
* Used word embedding techniques (word2vec) to represent words in a continuous vector space.
## 3. Fine-Tuned BERT Model
* Utilized the BERT (Bidirectional Encoder Representations from Transformers) model, pre-trained on a corpus, and fine-tuned it on the dataset.
## Technologies Used
PyTorch, Scikit-learn (sklearn), NLTK and SpaCy.

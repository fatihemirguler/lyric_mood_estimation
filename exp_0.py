import sys

import pandas as pd
import vectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVC
from sklearn.metrics import classification_report

data=pd.read_csv('data_with_cleaned_lyrics.csv')

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(data['cleaned_lyrics'], data['mood_cats'], test_size=0.2, random_state=42)
svc_model = SVC(kernel='rbf',C=10)

print()# Experiment 1: Unigrams
# unigram_vectorizer = TfidfVectorizer(ngram_range=(1, 1), max_features=5000)
# X_train_unigram = unigram_vectorizer.fit_transform(X_train)
# X_test_unigram = unigram_vectorizer.transform(X_test)
# svc_model.fit(X_train_unigram, y_train)
# y_pred_unigram = svc_model.predict(X_test_unigram)
# print("Unigram Classification Report:")
# print(classification_report(y_test, y_pred_unigram))

print()# Experiment 2: Bigrams
# bigram_vectorizer = TfidfVectorizer(ngram_range=(1, 2), max_features=5000) #Max features-- 5000 to 20000 performance slightly same
# X_train_bigram = bigram_vectorizer.fit_transform(X_train)
# X_test_bigram = bigram_vectorizer.transform(X_test)
#
# svc_model.fit(X_train_bigram, y_train)
# y_pred_bigram = svc_model.predict(X_test_bigram)
# print("Bigram Classification Report:")
# print(classification_report(y_test, y_pred_bigram))

print()# # Experiment 3: Trigrams
# trigram_vectorizer = TfidfVectorizer(ngram_range=(1, 3), max_features=5000)
# X_train_trigram = trigram_vectorizer.fit_transform(X_train)
# X_test_trigram = trigram_vectorizer.transform(X_test)
#
# svc_model.fit(X_train_trigram, y_train)
# y_pred_trigram = svc_model.predict(X_test_trigram)
# print("Trigram Classification Report:")
# print(classification_report(y_test, y_pred_trigram))
print()# GridSearch
# Experiment 2: Bigrams
bigram_vectorizer = TfidfVectorizer(ngram_range=(1, 2), max_features=5000)  # Max features-- 5000
X_train_bigram = bigram_vectorizer.fit_transform(X_train)
X_test_bigram = bigram_vectorizer.transform(X_test)

# Define a wider parameter grid for C and gamma
# param_grid = {
#     'C': [0.001, 0.01, 0.1, 1, 10, 100],  # Wider range of C values
#     'gamma': [0.001, 0.01, 0.1, 1],             # Gamma values for RBF kernel
#     'kernel': ['rbf'],                           # Using the RBF kernel
#     'class_weight': ['balanced', None]           # Option to balance class weights
###### } OUTPUT: Best parameters: {'C': 100, 'class_weight': None, 'gamma': 1, 'kernel': 'rbf'}
param_grid = {
    'C': [100,1000,2000],  # Wider range of C values
    'gamma': [1,10,100],             # Gamma values for RBF kernel
    'kernel': ['rbf'],                           # Using the RBF kernel
    'class_weight': ['balanced', None]           # Option to balance class weights
}
grid_search = GridSearchCV(SVC(), param_grid, cv=3, verbose=1)
grid_search.fit(X_train_bigram, y_train)

# Get the best parameters from the grid search
best_params = grid_search.best_params_
print(f"Best parameters: {best_params}")

# Evaluate the best model
best_svc = grid_search.best_estimator_
y_pred_bigram = best_svc.predict(X_test_bigram)

# Print classification report
print("Bigram Classification Report:")
print(classification_report(y_test, y_pred_bigram))
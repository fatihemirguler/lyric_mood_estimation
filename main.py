import torch
import sys
print("GPU Available: ", torch.cuda.is_available())
import contractions
from datasets import load_dataset
import pandas as pd
import numpy as np
import string
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
from datasets import load_dataset, Dataset
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification, Trainer, TrainingArguments
from collections import Counter


ds= load_dataset("Annanay/aml_song_lyrics_balanced", split="train")
data=ds.to_pandas()
train_dataset = load_dataset("Annanay/aml_song_lyrics_balanced", split="train")
train_df = train_dataset.to_pandas()

test_dataset  = load_dataset("Annanay/aml_song_lyrics_balanced", split="test")
test_df = test_dataset.to_pandas()

# Download necessary NLTK data
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('punkt_tab')
# stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

# Define a basic list of common English stopwords
stop_words = { 'a', 'an', 'the', 'and', 'or', 'nor', 'so', 'yet', 'for', 'in', 'on', 'at', 'by',
              'with', 'about', 'against', 'between', 'into', 'through', 'during', 'before', 'after',
              'above', 'below', 'to', 'from', 'up', 'down', 'out', 'over', 'under', 'again', 'further',
              'then', 'once', 'he', 'she', 'they', 'you', 'me', 'we', 'him', 'her', 'us', 'them',
              'my', 'mine', 'yours', 'his', 'hers', 'ours', 'theirs', 'is', 'am', 'are',
              'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'do', 'does', 'did', 'doing',
              'will', 'would', 'shall', 'should', 'can', 'could', 'may', 'might', 'must', 'who', 'whom',
              'whose', 'which', 'that', 'this', 'these', 'those', 'any', 'some', 'all', 'both',
              'each', 'every', 'either', 'neither', 'another', 'such', 'up', 'down', 'out', 'off',
              'over', 'under', "it's", "he's", "she's", "that's", "there's", "who's", "what's", "where's", "when's",
              "why's", "how's", "let's", "you're", "we're", "they're", "i've", "you've", "we've",
              "they've", "i'll", "you'll", "he'll", "she'll", "we'll", "they'll", "I'd", "you'd",
              "he'd", "she'd", "we'd", "they'd"}


# Define a function for text preprocessing
def preprocess_lyrics(lyrics):
    lyrics = contractions.fix(lyrics)
    # Remove anything in brackets (like [Verse 1])
    lyrics = re.sub(r'\[.*?\]', '', lyrics)
    # Replace literal '\n' with spaces
    lyrics = lyrics.replace('\\n', ' ')
    # Remove any isolated 'n' caused by incorrect newline handling
    lyrics = re.sub(r'\bn\b', '', lyrics)
    # Remove non-alphabetic characters
    # lyrics = re.sub(r'[^a-zA-Z\s]', '', lyrics)
    # Convert to lowercase
    lyrics = lyrics.lower()
    # Tokenize the lyrics
    tokens = word_tokenize(lyrics)
    print()
    # Remove stopwords and lemmatize the tokens
    cleaned_lyrics = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]
    return ' '.join(cleaned_lyrics)
def minimal_cleaning(lyrics):
    # Expand contractions (e.g., "can't" -> "cannot")
    lyrics = contractions.fix(lyrics)
    # Remove anything in brackets (like [Verse 1])
    lyrics = re.sub(r'\[.*?\]', '', lyrics)
    # Replace literal '\n' with spaces
    lyrics = lyrics.replace('\\n', ' ')
    # Remove any isolated 'n' caused by incorrect newline handling
    lyrics = re.sub(r'\bn\b', '', lyrics)
    # Convert to lowercase (optional, depending on your model)
    lyrics = lyrics.lower()
    return lyrics

def advanced_processing(lyrics):
    # Tokenize the lyrics
    tokens = word_tokenize(lyrics)
    # Remove stopwords and lemmatize the tokens
    cleaned_lyrics = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]
    return ' '.join(cleaned_lyrics)


# Analyze the lyrics column
def analyze_lyrics(lyrics):
    # Tokenize the lyrics
    tokens = word_tokenize(lyrics.lower())

    # Get the word count
    word_count = len(tokens)

    # Get the unique word count
    unique_word_count = len(set(tokens))

    # Get the most common words
    most_common_words = Counter(tokens).most_common(10)

    return word_count, unique_word_count, most_common_words


# Apply analysis to the lyrics column
data['word_count'], data['unique_word_count'], data['most_common_words'] = zip(*data['lyrics'].apply(analyze_lyrics))

# Summary statistics
print("\nSummary statistics:")
print(data[['word_count', 'unique_word_count']].describe())

# Display a few examples of the most common words in the lyrics
print("\nMost common words in sample lyrics:")
print(data[['lyrics', 'most_common_words']].head())


# Apply the corrected preprocessing to the lyrics column
# cleaned_lyrics = [preprocess_lyrics(lyric) for lyric in train_df['lyrics']]
# train_df['cleaned_lyrics'] = cleaned_lyrics
# cleaned_lyrics_test = [preprocess_lyrics(lyric) for lyric in test_df['lyrics']]
# test_df['cleaned_lyrics'] = cleaned_lyrics_test

# Minimal cleaning only (suitable for pre-trained models like BERT)
cleaned_lyrics = [minimal_cleaning(lyric) for lyric in train_df['lyrics']]
train_df['cleaned_lyrics'] = cleaned_lyrics

cleaned_lyrics_test = [minimal_cleaning(lyric) for lyric in test_df['lyrics']]
test_df['cleaned_lyrics'] = cleaned_lyrics_test

# Encode the target variable (mood)
label_encoder = LabelEncoder()
train_df['mood_encoded'] = label_encoder.fit_transform(train_df['mood'])
test_df['mood_encoded'] = label_encoder.fit_transform(test_df['mood'])
# # Get the target variable
# y_train = train_df['mood_encoded']
# y_test=test_df['mood_encoded']

# # Use TF-IDF to convert text data into numerical format
# tfidf = TfidfVectorizer(max_features=5000)  # Limiting to the top 5000 features
# X_train = tfidf.fit_transform(train_df['cleaned_lyrics']).toarray()
# X_test=tfidf.fit_transform(test_df['cleaned_lyrics']).toarray()

# print(train_df[['lyrics', 'cleaned_lyrics']].head())
# Display the full original and cleaned lyrics of the first row
print("Original Lyrics:\n", train_df['lyrics'].iloc[0])
print("\nCleaned Lyrics:\n", train_df['cleaned_lyrics'].iloc[0])

# # Train a Support Vector Machine (SVM) model
# model = SVC(kernel='linear', random_state=42)
# model.fit(X_train, y_train)
#
# # Predict on the validation set
# y_pred = model.predict(X_test)
#
# # Evaluate the model
# print("Accuracy:", accuracy_score(y_test, y_pred))
# print("\nClassification Report:\n", classification_report(y_test, y_pred, target_names=label_encoder.classes_))
print() #BERT
# BERT FINE TUNE START
# tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
# # Keep only the necessary columns
# train_df = train_df[['cleaned_lyrics', 'mood_encoded']].reset_index(drop=True)
# test_df = test_df[['cleaned_lyrics', 'mood_encoded']].reset_index(drop=True)
#
# # Convert the cleaned DataFrame to a Hugging Face Dataset object
# train_dataset = Dataset.from_pandas(train_df)
# test_dataset = Dataset.from_pandas(test_df)
#
# # Tokenize the dataset
# def tokenize_function(examples):
#     return tokenizer(examples['cleaned_lyrics'], padding='max_length', truncation=True, max_length=128)
#
# # Apply the tokenization function
# train_dataset = train_dataset.map(tokenize_function, batched=True)
# test_dataset = test_dataset.map(tokenize_function, batched=True)
#
# # Set the format for PyTorch (or TensorFlow)
# train_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'mood_encoded'])
# test_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'mood_encoded'])
#
# # Rename 'mood_encoded' to 'labels' as required by Hugging Face models
# train_dataset = train_dataset.rename_column("mood_encoded", "labels")
# test_dataset = test_dataset.rename_column("mood_encoded", "labels")
#
# # Initialize BERT model for sequence classification
# model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=4)
#
# # Define training arguments
# training_args = TrainingArguments(
#     output_dir='./results',
#     evaluation_strategy='epoch',
#     save_strategy='epoch',  # Ensure save strategy matches evaluation strategy
#     learning_rate=2e-5,
#     per_device_train_batch_size=8,
#     per_device_eval_batch_size=8,
#     num_train_epochs=5,
#     weight_decay=0.01,
#     load_best_model_at_end=True,
#     metric_for_best_model="eval_loss",
#     save_total_limit=1,
# )
#
#
#
# # Initialize the Trainer
# trainer = Trainer(
#     model=model,
#     args=training_args,
#     train_dataset=train_dataset,
#     eval_dataset=test_dataset,
# )
#
# # Fine-tune the model
# trainer.train()
#
# # Evaluate the model
# results = trainer.evaluate()
# print(results)

# BERT FINE TUNE END
print() #DISTILBERT
# # Load DistilBERT tokenizer
#
# # Convert to Hugging Face dataset
# train_dataset = Dataset.from_pandas(train_df[['cleaned_lyrics', 'mood_encoded']])
# test_dataset = Dataset.from_pandas(test_df[['cleaned_lyrics', 'mood_encoded']])
#
# # Load DistilBERT tokenizer
# tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
#
# # Tokenize dataset
# def tokenize_function(examples):
#     return tokenizer(examples['cleaned_lyrics'], padding='max_length', truncation=True, max_length=128)
#
# train_dataset = train_dataset.map(tokenize_function, batched=True)
# test_dataset = test_dataset.map(tokenize_function, batched=True)
#
# train_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'mood_encoded'])
# test_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'mood_encoded'])
#
# train_dataset = train_dataset.rename_column("mood_encoded", "labels")
# test_dataset = test_dataset.rename_column("mood_encoded", "labels")
#
# # Load DistilBERT model and move to GPU
# model = DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased', num_labels=len(label_encoder.classes_)).to('cuda')
#
# # Training arguments with GPU support
# training_args = TrainingArguments(
#     output_dir='./results',
#     evaluation_strategy='epoch',
#     save_strategy='epoch',
#     learning_rate=3e-5,
#     per_device_train_batch_size=16,
#     per_device_eval_batch_size=16,
#     num_train_epochs=3,
#     weight_decay=0.01,
#     load_best_model_at_end=True,
#     metric_for_best_model="eval_loss",
#     save_total_limit=1,
#     logging_dir='./logs',
#     logging_steps=10,
# )
#
# # Trainer with evaluation dataset
# trainer = Trainer(
#     model=model,
#     args=training_args,
#     train_dataset=train_dataset,
#     eval_dataset=test_dataset,  # Add evaluation dataset here
# )
#
# # Fine-tune the model
# trainer.train()
#
# # Evaluate the model
# results = trainer.evaluate()
# print(results)
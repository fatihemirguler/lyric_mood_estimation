import sys

import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from collections import Counter
import pandas as pd
import contractions
import re
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler

# print(contractions.fix("These Days Lyrics[Verse 1]\nLet's take the long way home tonight\nAnd just cruise along the boulevards\nLet's shine a little, we're the streetlights\nAnd dream ourselves away to Mars\nLet's follow all the stars above\nOr maybe take the subway home\n\n[Chorus]\nThese days are mine\nAnd I'm gonna take it slow\nLike the peaceful waters flow\nI'll make it up as I go\nThat's all I really know\nSo let's take the long way home\n[Verse 2]\nLet's cry a little if we wanna\nOr laugh out loud the best we can\nLet's meet the comfort of convenience\nAnd search a silence deep within\nLet's stay a while and take it easy\nIf that's something you can overcome\nLet's take a minute just to slip away\nAnd linger on before it's gone\n\n[Chorus]\nThese days are mine\nAnd I'm gonna take it slow\nLike the peaceful waters flow\nI'll make it up as I go\nThat's all I really know\nSo let's take the long way home\n\n[Chorus]\nThese days are mine\nAnd I'm gonna take it slow\nLike the peaceful waters flow\nI'll make it up as I go\nThat's all I really know\nSo let's take the long way home\nYou might also like[Outro]\nAnd I'm gonna take it slow\nLike the peaceful waters flow\nAnd that's all I really know\nSo let's take the long way homeEmbed" ))
# sys.exit()

# Set Pandas to display the full content of each column
pd.set_option('display.max_colwidth', None)

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
              'then', 'once', 'they', 'them',
              'my', 'mine', 'yours', 'his', 'hers', 'ours', 'theirs', 'is', 'am', 'are',
              'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'do', 'does', 'did', 'doing',
              'will', 'would', 'shall', 'should', 'can', 'could', 'may', 'might', 'must', 'who', 'whom',
              'whose', 'which', 'that', 'this', 'these', 'those', 'any', 'some', 'all', 'both',
              'each', 'every', 'either', 'neither', 'another', 'such', 'up', 'down', 'out', 'off',
              'over', 'under', "it's", "he's", "she's", "that's", "there's", "who's", "what's", "where's", "when's",
              "why's", "how's", "let's", "you're", "we're", "they're", "i've","I've", "you've", "we've",
              "they've", "i'll","I'll", "you'll", "he'll", "she'll", "we'll", "they'll", "I'd", "you'd",
              "he'd", "she'd", "we'd", "they'd"}


# data=pd.read_csv('data_ids.csv')
# data=pd.read_csv('data_all_unique.csv')
def preprocess_lyrics(lyrics):
    lyrics = lyrics.replace('\\n', ' ')

    # print(lyrics)
    lyrics = lyrics.replace('Embed', '')
    lyrics=contractions.fix(lyrics)
    # print(lyrics)
    lyrics = lyrics.lower()
    # Remove anything in brackets (like [Verse 1])
    lyrics = re.sub(r'\[.*?\]', '', lyrics)
    # Replace literal '\n' with spaces

    # Remove non-alphabetic characters
    # lyrics = re.sub(r'[^a-zA-Z\s]', '', lyrics)
    # Convert to lowercase

    # Tokenize the lyrics
    #tokens = word_tokenize(lyrics)
    # Remove stopwords and lemmatize the tokens

    # Process each word in the lyrics
    words = lyrics.split()  # Split the lyrics by whitespace
    cleaned_lyrics = [lemmatizer.lemmatize(word) for word in words if word not in stop_words]
    return ' '.join(cleaned_lyrics)

# data['cleaned_lyrics'] = [preprocess_lyrics(lyric) for lyric in data['lyrics']]
# print(data['cleaned_lyrics'])
# data.to_csv('data_with_cleaned_lyrics_af.csv', index=False)

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

# # Apply analysis to the lyrics column
# data['word_count'], data['unique_word_count'], data['most_common_words'] = zip(*data['lyrics'].apply(analyze_lyrics))
#
# # Summary statistics
# print("\nSummary statistics:")
# print(data[['word_count', 'unique_word_count']].describe())
#
# # Display a few examples of the most common words in the lyrics
# print("\nMost common words in sample lyrics:")
# print(data[['lyrics', 'most_common_words']].head())

data=pd.read_csv('data.csv')
# print(data.describe())
# print(data.info())  # This will give you a summary of the dataset
# print(data.head())  # Display the first few rows to get a sense of the data


# List of audio features to explore
audio_features = ['acousticness', 'danceability', 'energy', 'instrumentalness', 'liveness', 'loudness', 'speechiness', 'tempo', 'valence']

# # Plot histograms of each feature
# for feature in audio_features:
#     plt.figure(figsize=(10, 4))
#     plt.hist(data[feature], bins=30, color='blue', alpha=0.7)
#     plt.title(f'Distribution of {feature}')
#     plt.xlabel(feature)
#     plt.ylabel('Frequency')
#     plt.show()
#
#
# # Calculate the correlation matrix
# corr_matrix = data[audio_features].corr()
#
# # Plot the correlation heatmap
# plt.figure(figsize=(12, 8))
# sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f')
# plt.title('Correlation Matrix of Audio Features')
# plt.show()
#
#
# for feature in audio_features:
#     plt.figure(figsize=(10, 4))
#     sns.boxplot(x='mood_cats', y=feature, data=data)
#     plt.title(f'{feature} by Mood')
#     plt.xlabel('Mood')
#     plt.ylabel(feature)
#     plt.show()


scaler = StandardScaler()


data[audio_features] = scaler.fit_transform(data[audio_features])

# Group the data by label (assuming 'mood' is the label column) and calculate the mean of each feature
mean_values_per_label = data.groupby('mood_cats')[audio_features].mean()

# Display the mean values for each label
print(mean_values_per_label)

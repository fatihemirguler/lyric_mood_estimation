import sys

import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
import pandas as pd
import time
import urllib.parse

# Initialize Spotify API client
client_credentials_manager = SpotifyClientCredentials(client_id='75886f430e5c480b86703ce577eaef1e', client_secret='c884a964b96048849e3743b5cfa1b994')
sp = spotipy.Spotify(client_credentials_manager=client_credentials_manager)

# results = sp.search(q='weezer', limit=20)
# for idx, track in enumerate(results['tracks']['items']):
#     print(idx, track['name'])
#
# sys.exit()
print() #csv
# df_train = pd.read_csv('training.csv')
# df_test = pd.read_csv('test.csv')
# df_ids=pd.read_csv('data_ids.csv')
# df_all=pd.read_csv('data_all.csv')
# df_ids_500=pd.read_csv('data_ids.csv').head(500)
df=pd.read_csv('data_all_unique.csv')
print()#Base Data Prep
#df = pd.concat([df_train, df_test], ignore_index=True)
# df=df_ids.dropna()

# # Drop duplicate rows based on all columns
# df_unique = df_all.drop_duplicates()
#
# # Save the unique rows to a new CSV file
# df_unique.to_csv('data_all_unique.csv', index=False)
#
# # Display the number of unique rows
# print(f"Number of unique rows: {df_unique.shape[0]}")

# Function to split 'lyrics_filename' into song_name and artist_name

def extract_song_artist(lyrics_filename):
    try:
        parts = lyrics_filename.split('___')
        if len(parts) < 2:
            return None, None  # In case the format isn't as expected

        song_name = parts[-1].replace('_', ' ')
        artist_name = parts[0].replace('_', ' ')
        return song_name, artist_name
    except ValueError:
        return None, None  # In case the format isn't as expected

# Apply the function to create new columns
df['song_name'], df['artist_name'] = zip(*df['lyrics_filename'].apply(extract_song_artist))

# Drop the old 'lyrics_filename' column if you no longer need it
df = df.drop(columns=['lyrics_filename'])

# Save the updated DataFrame
df.to_csv('data_all_updated.csv', index=False)

# Display the first few rows to verify the changes
print(df.head())
# Apply the function to create new columns
#df['song_name'], df['artist_name'] = zip(*df['lyrics_filename'].apply(extract_song_artist))

# Function to search Spotify for song and return the first matching Spotify ID
added_index=0
def get_spotify_id(song_name, artist_name):
    try:
        global added_index
        global not_added_index
        query = f"track:{song_name} artist:{artist_name}"
        result = sp.search(q=urllib.parse.quote(query), type='track', limit=1)
        if result['tracks']['items']:
            added_index+=1
            print(added_index)
            return result['tracks']['items'][0]['id']
        else:
            return None
    except Exception as e:
        print(f"Error retrieving Spotify ID for {song_name} by {artist_name}: {e}")
        return None

# # Add a new column for Spotify IDs
# df['spotify_id'] = df.apply(lambda row: get_spotify_id(row['song_name'], row['artist_name']), axis=1)

# Save the updated dataset with Spotify IDs
#df.to_csv('data_all.csv', index=False)

# Optionally, pause between requests to avoid hitting API rate limits

# Function to get audio features for a batch of Spotify IDs
def get_audio_features(spotify_ids):
    try:
        features = sp.audio_features(tracks=spotify_ids)
        return features
    except Exception as e:
        print(f"Error retrieving audio features: {e}")
        return None

print() #BATCHING AND RETRIEVING

# # Split the Spotify IDs into batches of 100
# spotify_ids = df['spotify_id'].dropna().tolist()  # Drop any NaN values
# batch_size = 100
# spotify_id_batches = [spotify_ids[i:i + batch_size] for i in range(0, len(spotify_ids), batch_size)]
#
# # Initialize a list to store all audio features
# all_audio_features = []
# batch_index=0
# # Loop through each batch and retrieve audio features
# for batch in spotify_id_batches:
#     batch_index+=1
#     print(batch_index)
#     batch_features = get_audio_features(batch)
#     if batch_features:
#         all_audio_features.extend(batch_features)
#     time.sleep(1)  # Pause to avoid hitting rate limits
#
# # Convert the list of features to a DataFrame, filtering out any None entries
# audio_features_df = pd.DataFrame([feature for feature in all_audio_features if feature is not None])
#
# # audio_features_df = pd.DataFrame(all_audio_features)
#
# # Merge the audio features back with the original DataFrame
# df_with_audio_features = pd.merge(df, audio_features_df, left_on='spotify_id', right_on='id', how='left')
#
# # Save the updated DataFrame with audio features
# df_with_audio_features.to_csv('data_all.csv', index=False)

import numpy as np
import pandas as pd
import os
import random
from sklearn.utils import resample
import re

file_names = ['goemotions_1.csv', 'goemotions_2.csv', 'goemotions_3.csv']
base_save_path = 'data/datasets/goEmotions'

os.makedirs(base_save_path, exist_ok=True)

df1 = pd.read_csv(f'{base_save_path}/goemotions_1.csv')
df2 = pd.read_csv(f'{base_save_path}/goemotions_2.csv')
df3 = pd.read_csv(f'{base_save_path}/goemotions_3.csv')


df_combined = pd.concat([df1, df2, df3], ignore_index=True)
print("DF1 shape:", df1.shape)
print("DF2 shape:", df2.shape)
print("DF3 shape:", df3.shape)
print("combined DF:", df_combined.shape)

features = df_combined.columns.tolist()
print(features)

emotion_columns = ['admiration', 'amusement', 'anger', 'annoyance', 'approval',
                   'caring', 'confusion', 'curiosity', 'desire', 'disappointment',
                   'disapproval', 'disgust', 'embarrassment', 'excitement',
                   'fear', 'gratitude', 'grief', 'joy', 'love', 'nervousness',
                   'optimism', 'pride', 'realization', 'relief', 'remorse',
                   'sadness', 'surprise', 'neutral']

# Group by 'text_id' to aggregate ratings for the same text
df_grouped = df_combined.groupby('id', as_index=False)[emotion_columns].sum(numeric_only=True)
df_filtered = df_grouped[(df_grouped[emotion_columns] > 1).any(axis=1)].copy()


def random_emotion(row):
    emotions_with_agreement = [emotion for emotion in emotion_columns if row[emotion] > 1]
    random.shuffle(emotions_with_agreement)
    return emotions_with_agreement[0] if emotions_with_agreement else None


df_filtered['emotion_label'] = df_filtered.apply(random_emotion, axis=1)

final_df = pd.merge(df_filtered, df_combined[['id', 'text']], on='id').drop_duplicates()

rebalanced_df = final_df
rebalanced_df.loc[rebalanced_df['emotion_label'].isin(['admiration', 'relief']), 'emotion_label'] = 'gratitude'
rebalanced_df.loc[rebalanced_df['emotion_label'].isin(['annoyance', 'disgust', 'disapproval']),'emotion_label'] = 'anger'
rebalanced_df.loc[rebalanced_df['emotion_label'].isin(['optimism', 'caring', 'excitement', 'amusement', 'love']), 'emotion_label'] = 'joy'
rebalanced_df.loc[rebalanced_df['emotion_label'].isin(['grief', 'remorse', 'disappointment']), 'emotion_label'] = 'sadness'
rebalanced_df.loc[rebalanced_df['emotion_label'].isin(['nervousness', 'embarrassment', 'confusion']), 'emotion_label'] = 'fear'

excluded = ['approval', 'curiosity', 'realization', 'surprise','desire','pride']
rebalanced_df = rebalanced_df[~rebalanced_df['emotion_label'].isin(excluded)]

emotion_counts = rebalanced_df['emotion_label'].value_counts()
print('Emotion counts after re-balancing: ', emotion_counts)

target_size = 10000
neutral_df = rebalanced_df[rebalanced_df['emotion_label'] == 'neutral']

# Downsample the 'Neutral' entries to the target size
downsampled_neutral_df = neutral_df.sample(n=target_size, random_state=42)

# Combine back with the non-'Neutral' entries
downsampled_df = pd.concat([downsampled_neutral_df, rebalanced_df[rebalanced_df['emotion_label'] != 'neutral']])

# Verify the changes
print('Emotion count after down-sampling: ', downsampled_df['emotion_label'].value_counts())

df_clean = pd.DataFrame()
df_clean['text'] = downsampled_df['text']
df_clean['emotion_label'] = downsampled_df['emotion_label']


def clean_text(text):
    text = str(text).lower()  # Convert text to lowercase
    return text


df_clean['text'] = df_clean['text'].apply(clean_text)

# Save the cleaned data
df_clean.to_csv(f'{base_save_path}/cleaned_data.csv', index=False)

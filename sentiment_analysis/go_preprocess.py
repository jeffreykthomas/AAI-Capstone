import numpy as np
import pandas as pd
import requests
import os
import re

"""Downloading the Dataset"""

base_url = 'https://storage.googleapis.com/gresearch/goemotions/data/full_dataset/'
file_names = ['goemotions_1.csv', 'goemotions_2.csv', 'goemotions_3.csv']
base_save_path = 'data/datasets/goEmotions'

os.makedirs(base_save_path, exist_ok=True)

for file_name in file_names:
    url = base_url + file_name
    save_path = os.path.join(base_save_path, file_name)

    response = requests.get(url)
    response.raise_for_status()

    # Save the file
    with open(save_path, 'wb') as f:
        f.write(response.content)

    print(f'File downloaded and saved at {save_path}')

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

# Group the rows by 'id' and 'text' and sum the values of the emotion columns
df_combined = df_combined.groupby(['id', 'text'], as_index=False).sum()
print("Shape after grouping by 'id' and 'text':", df_combined.shape)

df_combined.drop(['id', 'author', 'subreddit', 'link_id', 'parent_id',
                  'created_utc', 'rater_id'], axis=1, inplace=True)
print("Shape after dropping columns:", df_combined.shape)

df_combined = df_combined.loc[df_combined['example_very_unclear'] != True]
print("Shape after filtering examples very unclear:", df_combined.shape)

df_combined.drop_duplicates(inplace=True)
print("Shape after dropping duplicates:", df_combined.shape)

df_combined.drop('example_very_unclear', axis=1, inplace=True)
print("Shape after dropping unclear:", df_combined.shape)

emotions_to_clear = ['admiration', 'amusement', 'caring', 'realization']
df_combined = df_combined[~(df_combined[emotions_to_clear] == 1).any(axis=1)]
print("Shape:", df_combined.shape)

df_combined.drop(emotions_to_clear, axis=1, inplace=True)

print("Shape of combined:", df_combined.shape)

df_filtered = df_combined.drop(columns=['text'])

"""Lets downsample neutral"""

# neutral_1_df = df_combined[df_combined['neutral'] == 1]
# neutral_0_df = df_combined[df_combined['neutral'] == 0]
# n_samples_to_keep = int(10000)
#
# reduced_neutral_1_df = neutral_1_df.sample(n=n_samples_to_keep)
# df_reduced = pd.concat([reduced_neutral_1_df, neutral_0_df])
#
# approval_1_df = df_reduced[df_reduced['approval'] == 1]
# approval_0_df = df_reduced[df_reduced['approval'] == 0]
# n_samples_to_keep = int(10000)
#
# reduced_approval_1_df = approval_1_df.sample(n=n_samples_to_keep)
# df_reduced = pd.concat([reduced_approval_1_df, approval_0_df])
df_reduced = df_combined.copy()

"""Relabeling for a more balanced yet diverse distribution"""

df_reduced.loc[(df_reduced['nervousness'] == 1) | (df_reduced['embarrassment'] == 1), 'fear'] = 1
df_reduced.loc[(df_reduced['love'] == 1), 'joy'] = 1
df_reduced.loc[(df_reduced['relief'] == 1) | (df_reduced['pride'] == 1), 'optimism'] = 1
df_reduced.loc[(df_reduced['remorse'] == 1) | (df_reduced['grief'] == 1), 'sadness'] = 1
df_reduced.loc[(df_reduced['disgust'] == 1), 'disapproval'] = 1
df_reduced.loc[(df_reduced['excitement'] == 1), 'optimism'] = 1

df_reduced.drop(['nervousness', 'embarrassment', 'love', 'relief', 'pride',
                 'remorse', 'grief', 'desire', 'surprise', 'disgust', 'excitement'], axis=1, inplace=True)

print("Shape after relabeling:", df_reduced.shape)

# df_reduced.head(10)
#
# disapproval_1_df = df_reduced[df_reduced['disapproval'] == 1]
# disapproval_0_df = df_reduced[df_reduced['disapproval'] == 0]
# n_samples_to_keep = int(12000)
#
# reduced_disapproval_1_df = disapproval_1_df.sample(n=n_samples_to_keep)
# df_reduced = pd.concat([reduced_disapproval_1_df, disapproval_0_df])
#
# df_reduced_values = df_reduced.drop(columns=['text'])
df_balanced = df_reduced

"""Remove duplicates and find null values."""

df_unique = df_balanced
print("Shape of df_balanced:", df_balanced.shape)
print("Shape of df_unique:", df_unique.shape)

df_unique_values = df_unique.drop(columns=['text'])


nulls_per_column = df_unique.isnull().sum()
total_nulls = nulls_per_column.sum()
print("Nulls per column:\n", nulls_per_column)
print("\nTotal nulls in the DataFrame:", total_nulls)


def clean_text(text):
    text = str(text).lower()  # Convert text to lowercase
    text = re.sub("[\[].*?[\]]", "", text)  # Remove text within square brackets
    text = re.sub(r"http\S+", "", text)  # Remove URLs
    text = re.sub(r"[^\w\s]", "", text)  # Remove non-alphanumeric characters (excluding spaces)
    text = re.sub(r"\d+", "", text)  # Remove digits
    return text


df_unique['cleaned_text'] = df_unique['text'].apply(clean_text)
df_clean = df_unique.drop(columns=['text'])
print(df_clean.head(20))

column_names = df_clean.columns
# column_names.remove('text')
print(column_names)
print('Shape of df_clean:', df_clean.shape)


# Function to find the emotion label for a row
def find_emotion_label(row):
    agreed_emotions = []
    for col in column_names:
        if col == 'cleaned_text':
            continue
        if row[col] > 1:
            agreed_emotions.append(col)
    if len(agreed_emotions) > 0:
        # return random emotion from agreed emotions
        return np.random.choice(agreed_emotions)
    else:
        return np.nan  # Returns NaN if no emotion column has a value of 1


# Apply the function to each row to create a new 'emotion_label' column
df_clean['emotion_label'] = df_clean.apply(find_emotion_label, axis=1)
print('Shape of df_clean after selecting labels:', df_clean.shape)
df_labeled = df_clean[['cleaned_text', 'emotion_label']].copy()

emotion_counts = df_labeled['emotion_label'].value_counts()
print("Emotion counts:\n", emotion_counts)

# Save the cleaned data
save_path = os.path.join(base_save_path, 'goemotions_cleaned.csv')
df_labeled.to_csv(save_path, index=False)



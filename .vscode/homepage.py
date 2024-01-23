import streamlit as st
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
import re

st.title("""Sentiments in social media""")
st.text('Social networks have become an integral part of our lives. They influence many areas: social life, business, marketing, etc. There people express their emotions and opinions, which are further reflected in these areas. Using real data from popular social networks as an example, we will try to analyze them and draw conclusions.')

st.text("""First, let's get acquainted with the data with which we will work. Here is collected information from social networks Facebook, Twitter(X) and Instagram for a period of time""")

#reading a csv file and checking if a csv is available at a given address
csv_path="C:\\Users\\Xiaomi\\Desktop\\project\\.vscode\\sentimentdataset.csv"

if os.path.exists(csv_path):
    sentiment_df=pd.read_csv(csv_path, header=0)
    #st.dataframe(sentiment_df)
else:(f"File '{csv_path}' not found. Please check the path.")


st.text(f'Numbers of rows and columns: {sentiment_df.shape}')
#first few rows of the data set
st.text('First few rows of the data set')
st.dataframe(sentiment_df.head())

#display statistical information
st.dataframe(sentiment_df.describe().T)

#display the types of data
data_types=sentiment_df.dtypes
st.dataframe(data_types)


#display all the dataset information 
#st.write(sentiment_df.info)

info_text = sentiment_df.info()
st.text(info_text)
st.dataframe(sentiment_df.info())

#Data cleaning
st.title('Data cleaning')
st.text('In order to make it more convenient for us to work with the data, we must first clean it.')
 #In our case, remove what does not convey the mood in the texts, this includes pronouns, prepositions, and articles. We will also remove emoticons and signs.
#remove the first two columns, since they do not contain useful information, but only index rows

#st.dataframe(sentiment_df.columns)
st.text("First, let’s remove the columns that don’t make sense. That is, columns that duplicate the index.")
#Exploring the columns of a dataset
st.dataframe(sentiment_df.columns)
#removing unnamed columns
cleaned_df=(sentiment_df.drop(columns=['Unnamed: 0.1','Unnamed: 0']))

#Check for missing values in a dataset
st.text(cleaned_df.isnull().sum())

#data cleaning function
def data_cleaning(Text):
    Text = Text.lower()
    Text = re.sub(r'[^\w\s]','',Text)
    return Text

#remove spaces in columns
cleaned_df['Sentiment'] = cleaned_df['Sentiment'].str.strip()
cleaned_df['Platform'] = cleaned_df['Platform'].str.strip()
cleaned_df['Hashtag'] = cleaned_df['Hashtags'].str.strip()
cleaned_df['Country'] = cleaned_df['Country'].str.strip()

#display the cleaned dataset
st.dataframe(cleaned_df)

#comparing the original text with the cleaned text
cleaned_df['Cleaned_Text'] = cleaned_df['Text'].apply(data_cleaning)
st.dataframe(cleaned_df[['Text', 'Cleaned_Text']].iloc[0:4])

#display the plots
st.title("Interesting plots")

#data_types=cleaned_df.dtypes
#st.dataframe(data_types)

# Count the occurrences of each sentiment
sentiment_counts = cleaned_df['Sentiment'].value_counts()
#Convert Series to DataFrame
sentiment_counts_df = sentiment_counts.to_frame().reset_index()
sentiment_counts_df.columns = ['Sentiment', 'Count']
st.dataframe(sentiment_counts_df)
# Print data types of sentiment_counts_df
#data_types = sentiment_counts_df.dtypes
#st.text("Типы данных в DataFrame:")
#st.dataframe(data_types)

#categorize sentiments
positive_sentiments =[' Positive ', 'Happy', 'Joy', 'Excitement', 'Contentment', 'Gratitude', 'Curiosity', 'Serenity', 'Nostalgia', 'Awe', 'Hopeful', 'Euphoria', 'Elation', 'Enthusiasm', 'Pride', 'Determination', 'Playful', 'Inspiration', 'Hope', 'Happiness', 'Inspired', 'Empowerment', 'Proud', 'Grateful', 'Thrill', 'Overwhelmed', 'Compassionate', 'Reflection', 'Enchantment', 'Admiration', 'Reverence', 'Fulfillment', 'Compassion', 'Tenderness', 'Amusement', 'Arousal', 'Adventure', 'Satisfaction', 'Wonder', 'Accomplishment', 'Creativity', 'Harmony', 'Kind', 'Love', 'Confident', 'Free-spirited', 'Empathetic', 'Resilient', 'Rejuvenation', 'Relief', 'Creative Inspiration', 'Celestial Wonder', "Nature's Beauty", 'Thrilling Journey', 'Winter Magic', 'Culinary Adventure', 'Mesmerizing', 'Vibrancy', 'Imagination', 'Joy in Baking', 'Breakthrough', 'Solace', 'Celebration', 'Renewed Effort', 'Challenge', 'Mindfulness', 'Energy', 'Melodic', 'Motivation', 'Culinary Odyssey', 'Artistic Burst', 'Adrenaline', 'Dazzle', 'Freedom', 'Inner Journey', 'Festive Joy', 'Joyful Reunion', 'Grandeur', 'Blessed', 'Appreciation', 'Confidence', 'Wonderment', 'Optimism', 'Pensive', 'Playful Joy', 'Elegance', 'Immersion', 'Spark', 'Marvel', 'Overjoyed', 'DreamChaser', 'Romance', 'Amazement', 'Success', 'Friendship', 'Kindness', 'Positivity', 'Solitude', 'Coziness', 'Whimsy', 'Contemplation']
negative_sentiments =['Despair', 'Grief', 'Sad', 'Loneliness', 'Embarrassed', 'Regret', 'Frustration', 'Ambivalence', 'Melancholy', 'Numbness', 'Bad', 'Hate', 'Surprise', 'Bitterness', 'Frustrated', 'Betrayal', 'Disgust', 'Shame', 'Jealousy', 'Sorrow', 'Loss', 'Mischievous', 'Disappointment', 'Isolation', 'Coziness', 'Intimidation', 'Anxiety', 'Helplessness', 'Envy', 'Anger', 'Zest', 'Yearning', 'Apprehensive', 'Fear', 'Sadness', 'Enjoyment', 'Adoration', 'Affection', 'Disappointed', 'Engagement', 'Obstacle', 'Heartwarming', 'Triumph', 'Suspense', 'Touched', 'Devastated', 'Heartbreak', 'Ruins', 'Desperation', 'Darkness', 'Exhaustion', 'Lost Love', 'Emotional Storm', 'Suffering', 'Bittersweet', 'Intrigue']
neutral_sentiments = ['Acceptance', 'Confusion', 'Indifference', 'Engaged', 'Dismissive', 'Confident', 'Runway Creativity', 'Sympathy', 'Iconic', 'Connection', 'Hypnotic', 'Colorful', 'Ecstasy', 'Charm', 'Journey', 'Pressure', 'Ocean\'s Freedom', 'Relief', 'Winter Magic', 'Culinary Adventure', 'Mesmerizing', 'Vibrancy', 'Imagination', 'Envisioning History', 'Joy in Baking', 'Breakthrough', 'Solace', 'Celebration', 'Miscalculation', 'Renewed Effort', 'Whispers of the Past', 'Challenge', 'Mindfulness', 'Energy', 'Melodic', 'Motivation', 'Culinary Odyssey', 'Artistic Burst', 'Adrenaline', 'Dazzle', 'Freedom', 'Inner Journey', 'Festive Joy', 'Joyful Reunion', 'Grandeur', 'Blessed', 'Appreciation', 'Confidence', 'Wonderment', 'Optimism', 'Pensive', 'Playful Joy', 'Elegance', 'Immersion', 'Spark', 'Marvel', 'DreamChaser', 'Romance', 'Amazement', 'Success', 'Friendship', 'Kindness', 'Positivity', 'Solitude', 'Coziness', 'Whimsy', 'Contemplation']



#Function for filtering emotions by category and counting them
def filter_and_sum(dataframe, sentiments):
    selected_rows = dataframe[dataframe['Sentiment'].isin(sentiments)]
    #st.dataframe(selected_rows)
    
    # Sum of values in the 'Counts' column
    sum_counts = selected_rows['Count'].sum()
    #st.text("Number of chosen sentimnets:")
    st.text(sum_counts)

st.text('Number of posts with positive sentiment:') 
positive_sum = filter_and_sum(sentiment_counts_df, positive_sentiments)
st.text('Number of posts with negative sentiment:') 
negative_sum = filter_and_sum(sentiment_counts_df, negative_sentiments)
st.text('Number of posts with neutral sentiment:') 
neutral_sum = filter_and_sum(sentiment_counts_df, neutral_sentiments)


import streamlit as st
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
import io
import os
import re

#setting page options
st.set_page_config(
    page_title="Sentiments in social media",
    page_icon="😃",
)

st.title("""Sentiments in social media""")

#adding the image
image_url="https://colibriwp.com/blog/wp-content/uploads/2021/06/image9-1.png"
st.image(image_url, caption='Sentiments Image', use_column_width=True)

#reading a csv file and checking if a csv is available at a given address
csv_path="C:\\Users\\Xiaomi\\Desktop\\project\\.vscode\\sentimentdataset.csv"

if os.path.exists(csv_path):
    sentiment_df=pd.read_csv(csv_path, header=0)
    st.text('Dataset:')
    st.dataframe(sentiment_df)
    st.markdown("[Kaggle](https://www.kaggle.com/datasets/kashishparmar02/social-media-sentiments-analysis-dataset/)", unsafe_allow_html=True)
else:(f"File '{csv_path}' not found. Please check the path.")

#display last rows of dataset
st.text(f'Numbers of rows and columns: {sentiment_df.shape}')
st.text('Last rows of the data set:')
st.dataframe(sentiment_df.tail())

#statistics on likes
max_likes = sentiment_df['Likes'].max()
min_likes = sentiment_df['Likes'].min()
average_likes = sentiment_df['Likes'].mean()
show_likes_button = st.button("Show Likes Statistics")
if show_likes_button:
    st.write(f"Maximum number of likes: {max_likes}")
    st.write(f"Minimum number of likes: {min_likes}")
    st.write(f"Average number of likes: {average_likes}")

#statistics on retweets
max_retweets = sentiment_df['Retweets'].max()
min_retweets = sentiment_df['Retweets'].min()
average_retweets = sentiment_df['Retweets'].mean()
show_retweets_button = st.button("Show Retweets Statistics")
if show_retweets_button:
    st.write(f"Maximum number of retweets: {max_retweets}")
    st.write(f"Minimum number of retweets: {min_retweets}")
    st.write(f"Average number of retweets: {average_retweets}")

#statistics by date
first_date = sentiment_df['Timestamp'].min()
last_date =sentiment_df['Timestamp'].max()
first_post = sentiment_df[sentiment_df['Timestamp'] == first_date]
last_post = sentiment_df[sentiment_df['Timestamp'] == last_date]

show_dates_button = st.button("Show Earliest and Latest Dates")
if show_dates_button:  
    st.write(f"The earliest date: {first_date}")
    st.write("Post on the earliest date:")
    st.write(first_post[['Text', 'Likes', 'Retweets']])

    st.write(f"\nThe latest date: {last_date}")
    st.write("Post on the latest date:")
    st.write(last_post[['Text', 'Likes', 'Retweets']])

#display of countries
unique_countries=sentiment_df['Country'].unique()
if st.button('Show Countries'):
    st.write('Countries:', unique_countries)


#display statistical information
st.text('Statistical information:')
st.dataframe(sentiment_df.describe().T)

#display the types of data
st.text('The types of data:')
data_types=sentiment_df.dtypes
st.dataframe(data_types)

#display the dataset information 
#getting information about the sentiment_df and storing it in the buffer
st.text("Information about the dataset:")
db_info_buffer = io.StringIO()
sentiment_df.info(buf=db_info_buffer)
st.text(db_info_buffer.getvalue())

#data cleaning
st.title('Data cleaning')
st.text('In order to make it more convenient to work with the data, we must clean it.')

#exploring the columns of a dataset
if st.button('Show Dataset Columns'):
    st.text('The columns of a dataset:')
    st.dataframe(sentiment_df.columns)
#removing unnamed columns and columns of month, day, hour
cleaned_df=sentiment_df.drop(columns=['Unnamed: 0.1','Unnamed: 0','Month','Day','Hour'])

#сheck for missing values in a dataset
st.text('Checking for missing values in a dataset.')
st.text(cleaned_df.isnull().sum())

#text clearing function
def data_cleaning(Text):
    Text = Text.lower()
    Text = re.sub(r'[^\w\s]','',Text)
    return Text

#remove spaces in columns
cleaned_df['Sentiment'] = cleaned_df['Sentiment'].str.strip()
cleaned_df['Platform'] = cleaned_df['Platform'].str.strip()
cleaned_df['Hashtag'] = cleaned_df['Hashtags'].str.strip()
cleaned_df['Country'] = cleaned_df['Country'].str.strip()

#deviding the hashtags column'
cleaned_df[['Hashtag1', 'Hashtag2']] = cleaned_df['Hashtags'].str.split(expand=True, n=1)
cleaned_df = cleaned_df.drop(columns=['Hashtag'])
cleaned_df = cleaned_df.drop(columns=['Hashtags'])

#comparing the original text with the cleaned text
st.text('Comparing the original text with the cleaned text.')
cleaned_df['Cleaned_Text'] = cleaned_df['Text'].apply(data_cleaning)
st.dataframe(cleaned_df[['Text', 'Cleaned_Text']].iloc[0:4])

#replacing column Text to column Cleaned_Text
cleaned_df['Text'] = cleaned_df['Cleaned_Text']
cleaned_df=cleaned_df.drop(columns=['Cleaned_Text'])

st.text('Cleaned dataset')
st.text(f'Numbers of rows and columns: {cleaned_df.shape}')
st.text('The text was replaced. Spaces were removed in columns Sentiment,Platform,Hashtag and Country.')
st.text('Duplicate columns removed. Hashtags were divided into 2 columns.')
st.dataframe(cleaned_df)

#display the plots
st.title("Interesting plots")

#calculate the sentiments
st.text('Сalculating the sentiments of posts to view them on a chart.')
sentiment_counts = cleaned_df['Sentiment'].value_counts()
# Convert Series to DataFrame
sentiment_counts_df = sentiment_counts.to_frame().reset_index()
sentiment_counts_df.columns = ['Sentiment', 'Count']
st.dataframe(sentiment_counts_df)

#categorize sentiments
positive_sentiments =[' Positive ', 'Happy', 'Joy', 'Excitement', 'Contentment', 'Gratitude', 'Curiosity', 'Serenity', 'Nostalgia', 'Awe', 'Hopeful', 'Euphoria', 'Elation', 'Enthusiasm', 'Pride', 'Determination', 'Playful', 'Inspiration', 'Hope', 'Happiness', 'Inspired', 'Empowerment', 'Proud', 'Grateful', 'Thrill', 'Overwhelmed', 'Compassionate', 'Reflection', 'Enchantment', 'Admiration', 'Reverence', 'Fulfillment', 'Compassion', 'Tenderness', 'Amusement', 'Arousal', 'Adventure', 'Satisfaction', 'Wonder', 'Accomplishment', 'Creativity', 'Harmony', 'Kind', 'Love', 'Confident', 'Free-spirited', 'Empathetic', 'Resilient', 'Rejuvenation', 'Relief', 'Creative Inspiration', 'Celestial Wonder', "Nature's Beauty", 'Thrilling Journey', 'Winter Magic', 'Culinary Adventure', 'Mesmerizing', 'Vibrancy', 'Imagination', 'Joy in Baking', 'Breakthrough', 'Solace', 'Celebration', 'Renewed Effort', 'Challenge', 'Mindfulness', 'Energy', 'Melodic', 'Motivation', 'Culinary Odyssey', 'Artistic Burst', 'Adrenaline', 'Dazzle', 'Freedom', 'Inner Journey', 'Festive Joy', 'Joyful Reunion', 'Grandeur', 'Blessed', 'Appreciation', 'Confidence', 'Wonderment', 'Optimism', 'Pensive', 'Playful Joy', 'Elegance', 'Immersion', 'Spark', 'Marvel', 'Overjoyed', 'DreamChaser', 'Romance', 'Amazement', 'Success', 'Friendship', 'Kindness', 'Positivity', 'Solitude', 'Coziness', 'Whimsy', 'Contemplation']
negative_sentiments =['Despair', 'Grief', 'Sad', 'Loneliness', 'Embarrassed', 'Regret', 'Frustration', 'Ambivalence', 'Melancholy', 'Numbness', 'Bad', 'Hate', 'Surprise', 'Bitterness', 'Frustrated', 'Betrayal', 'Disgust', 'Shame', 'Jealousy', 'Sorrow', 'Loss', 'Mischievous', 'Disappointment', 'Isolation', 'Coziness', 'Intimidation', 'Anxiety', 'Helplessness', 'Envy', 'Anger', 'Zest', 'Yearning', 'Apprehensive', 'Fear', 'Sadness', 'Enjoyment', 'Adoration', 'Affection', 'Disappointed', 'Engagement', 'Obstacle', 'Heartwarming', 'Triumph', 'Suspense', 'Touched', 'Devastated', 'Heartbreak', 'Ruins', 'Desperation', 'Darkness', 'Exhaustion', 'Lost Love', 'Emotional Storm', 'Suffering', 'Bittersweet', 'Intrigue']
neutral_sentiments = ['Acceptance', 'Confusion', 'Indifference', 'Engaged', 'Dismissive', 'Confident', 'Runway Creativity', 'Sympathy', 'Iconic', 'Connection', 'Hypnotic', 'Colorful', 'Ecstasy', 'Charm', 'Journey', 'Pressure', 'Ocean\'s Freedom', 'Relief', 'Winter Magic', 'Culinary Adventure', 'Mesmerizing', 'Vibrancy', 'Imagination', 'Envisioning History', 'Joy in Baking', 'Breakthrough', 'Solace', 'Celebration', 'Miscalculation', 'Renewed Effort', 'Whispers of the Past', 'Challenge', 'Mindfulness', 'Energy', 'Melodic', 'Motivation', 'Culinary Odyssey', 'Artistic Burst', 'Adrenaline', 'Dazzle', 'Freedom', 'Inner Journey', 'Festive Joy', 'Joyful Reunion', 'Grandeur', 'Blessed', 'Appreciation', 'Confidence', 'Wonderment', 'Optimism', 'Pensive', 'Playful Joy', 'Elegance', 'Immersion', 'Spark', 'Marvel', 'DreamChaser', 'Romance', 'Amazement', 'Success', 'Friendship', 'Kindness', 'Positivity', 'Solitude', 'Coziness', 'Whimsy', 'Contemplation']

#function for filtering emotions by category and counting them
def filter_and_sum(dataframe, sentiments):
    selected_rows = dataframe[dataframe['Sentiment'].isin(sentiments)]
    if not selected_rows.empty:
        sum_counts = selected_rows['Count'].sum()
        st.text(sum_counts)
        return sum_counts
    else:
        st.text("No matching rows found.")
        return 0
    
#counting sentiments by category
st.text('Number of posts with positive sentiment:') 
positive_sum = filter_and_sum(sentiment_counts_df, positive_sentiments)
st.text('Number of posts with negative sentiment:') 
negative_sum = filter_and_sum(sentiment_counts_df, negative_sentiments)
st.text('Number of posts with neutral sentiment:') 
neutral_sum = filter_and_sum(sentiment_counts_df, neutral_sentiments)

sentiment_categories_df = pd.DataFrame({
    'Sentiment': ['Positive', 'Negative', 'Neutral'],
    'Count': [positive_sum, negative_sum, neutral_sum]
})

#constructing a pie chart by sentiment categories
fig, ax = plt.subplots(figsize=(6, 6))
sentiment_categories_series = pd.Series([positive_sum, negative_sum, neutral_sum], index=['Positive', 'Negative', 'Neutral'])
ax.pie(sentiment_categories_series, labels=sentiment_categories_series.index, autopct='%1.1f%%', startangle=140,colors=['green','red','yellow'])
ax.set_title('Distribution of Sentiment Categories in Posts')
st.pyplot(fig)

#creating a column chart of number of posts by platform
platform_counts = cleaned_df['Platform'].value_counts()
platform_counts_df = platform_counts.to_frame().reset_index()
platform_counts_df.columns = ['Platform', 'Count']
st.dataframe(platform_counts_df)
st.bar_chart(platform_counts_df.set_index('Platform'))
plt.figure(figsize=(8, 4))
plt.bar(platform_counts_df['Platform'], platform_counts_df['Count'], color=['blue', 'green', 'red'])
plt.xlabel('Platform')
plt.ylabel('Number of posts')
plt.title('Number of posts on social media')
plt.show()

#filtering dataFrame for posts from the UK
sentiment_by_country = cleaned_df.groupby(['Country', 'Sentiment']).size().reset_index(name='Count')
sentiment_counts_df = pd.merge(sentiment_counts_df, sentiment_by_country, how='left', on='Sentiment')
st.dataframe(sentiment_counts_df)
uk_data = sentiment_counts_df[sentiment_counts_df['Country'] == 'UK']

#constructing a histogram of the distribution of sentiment in the UK
fig, ax = plt.subplots()
sentiments = uk_data['Sentiment']
counts = uk_data['Count_y']
ax.bar(sentiments, counts)
ax.set_xlabel('Sentiment')
ax.set_ylabel('Count')
ax.set_title('Distribution of Sentiments in the UK')
plt.setp(ax.get_yticklabels(), rotation=45, ha='right',fontsize=5)
ax.set_xticks(np.arange(len(sentiments)))
plt.setp(ax.get_xticklabels(), rotation=45, ha='right',fontsize=5)
st.pyplot(fig)

#visualization of top 10 hashtags
hashtags = cleaned_df[['Hashtag1', 'Hashtag2']].values.flatten()
#сounting the number of times each hashtag appears
top_hashtags = pd.Series(hashtags).value_counts().head(10)
st.dataframe(top_hashtags)
fig, ax = plt.subplots(figsize=(10, 6))
top_hashtags.plot(kind='bar', ax=ax, color='pink')
ax.set_title('Top 10 Hashtags')
ax.set_xlabel('Hashtag')
ax.set_ylabel('Hashtag usage number')
st.pyplot(fig)

#calculate the change in average likes for all years
average_likes_by_year = cleaned_df.groupby('Year')['Likes'].mean()
average_likes_change = average_likes_by_year.diff()
average_retweets_by_year = cleaned_df.groupby('Year')['Retweets'].mean()
average_retweets_change = average_retweets_by_year.diff()
fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(average_likes_by_year.index, average_likes_change, label='Changing likes', marker='o', linestyle='-', color='b')
ax.plot(average_retweets_by_year.index, average_retweets_change, label='Changing retweets', marker='o', linestyle='-', color='r')
ax.set_title('Change in average likes and retweets for all years')
ax.set_xlabel('Year')
ax.set_ylabel('Average value')
ax.legend()
ax.grid(True)
st.pyplot(fig)

#сorrelation calculation and display
correlation = cleaned_df[['Likes', 'Retweets']].corr().iloc[0, 1]
st.title('Correlation between likes and retweets')
st.write("Let's look at the correlation between the number of likes and the number of retweets for each post in 2023.")
st.write('Correlation between likes and retweets:', correlation)
st.write('Likes and retweets chart:')
fig, ax = plt.subplots()
sns.scatterplot(x='Likes', y='Retweets', data=cleaned_df, ax=ax)
st.pyplot(fig)

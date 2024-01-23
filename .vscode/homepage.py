import streamlit as st
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
import io
import os
import re

st.set_page_config(
    page_title="Sentiments in social media",
    page_icon="👩",
    #layout="wide",
)

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
st.dataframe(sentiment_df.head(10))

#display statistical information
st.text('Statistical information')
st.dataframe(sentiment_df.describe().T)

#display the types of data
st.text('The types of data')
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
 #In our case, remove what does not convey the mood in the texts, this includes pronouns, prepositions, and articles. We will also remove emoticons and signs.
#remove the first two columns, since they do not contain useful information, but only index rows

st.text("First, let’s remove the columns that don’t make sense.")
#Exploring the columns of a dataset
st.text('The columns of a dataset')
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

# Функция для очистки хэштегов
def clean_hashtags(hashtag_string):
    # Разбиваем строку по пробелам и убираем дубликаты
    hashtags = list(set(hashtag_string.split()))
    # Убираем символ '#' из каждого хэштега
    return ' '.join(hashtags)

cleaned_df['Hashtags'] = sentiment_df['Hashtags'].apply(clean_hashtags)

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
    if not selected_rows.empty:
        sum_counts = selected_rows['Count'].sum()
        st.text(sum_counts)
        return sum_counts
    else:
        st.text("No matching rows found.")
        return 0
    # Sum of values in the 'Counts' column
    sum_counts = selected_rows['Count'].sum()
    #st.text("Numbers of chosen sentiments:")
    st.text(sum_counts)

st.text('Number of posts with positive sentiment:') 
positive_sum = filter_and_sum(sentiment_counts_df, positive_sentiments)
st.text('Number of posts with negative sentiment:') 
negative_sum = filter_and_sum(sentiment_counts_df, negative_sentiments)
st.text('Number of posts with neutral sentiment:') 
neutral_sum = filter_and_sum(sentiment_counts_df, neutral_sentiments)

#st.text(f'Type of neutral_sum: {type(neutral_sum)}')

sentiment_categories_df = pd.DataFrame({
    'Sentiment': ['Positive', 'Negative', 'Neutral'],
    'Count': [positive_sum, negative_sum, neutral_sum]
})

st.dataframe(sentiment_categories_df)

#constructing a pie chart by sentiment categories
fig, ax = plt.subplots(figsize=(6, 6))
sentiment_categories_series = pd.Series([positive_sum, negative_sum, neutral_sum], index=['Positive', 'Negative', 'Neutral'])
ax.pie(sentiment_categories_series, labels=sentiment_categories_series.index, autopct='%1.1f%%', startangle=140)
ax.set_title('Post types by sentiment categories')
#Sentiment Distribution

# Displaying the chart in Streamlit
st.pyplot(fig)

# Построение столбчатой диаграммы
fig, ax = plt.subplots(figsize=(30, 12))
sentiment_counts_df.groupby('Sentiment')['Count'].sum().plot(kind='bar', ax=ax, color=['green', 'red', 'blue'])
ax.set_title('Column chart for the number of posts with different sentiments')
ax.set_xlabel('Sentiments')
ax.set_ylabel('Numbers of posts')
plt.legend(sentiment_counts_df['Sentiment'])
st.pyplot(fig)  

#
# sentiment_categories_df=pd.concat([sentiment_categories_df, cleaned_df['Timestamp']])
# st.dataframe(sentiment_categories_df)
# # Предполагая, что вы загрузили свой набор данных в DataFrame под названием cleaned_df
# cleaned_df['Timestamp'] = pd.to_datetime(cleaned_df['Timestamp'])
# cleaned_df.set_index('Timestamp', inplace=True)
# # Создание графика
# fig, ax = plt.subplots()
# sns.lineplot(x=cleaned_df.index, y=sentiment_categories_df['Sentiment'], data=cleaned_df)
# plt.title('Настроение с течением времени')
# # Отображение графика в Streamlit
# st.pyplot(fig)

# Предполагая, что вы загрузили свой набор данных в DataFrame под названием cleaned_df

#ГРАФИК 3 СДЕЛАТЬ ПО ХЭШТЕГАМ
#ГРАФИК 4 СДЕЛАТЬ ПО СТРАНАМ (ЛИБО ПОПУЛЯРНЫЕ, ГДЕ БОЛЬШЕ ПОСТЯТ И НЕ ПОПУЛЯРНЫЕ, ЛИБО ПО КОНТИНЕНТАМ ИЛИ СТРАНАМ)
#ГРАФИК 5 СДЕЛАТЬ ПО ВРЕМЕНИ (ВЫЯСНИТЬ В КАКОЙ ПЕРИОД БЫЛИ САМЫЕ ПОЗИТИВНЫЕ И НЕГАТИВНЫЕ)

#ДОБАВИТЬ ОКНА И ВЫПАДАЮЩИЕ ШТУКИ 

popular_hashtags_df = cleaned_df.sort_values(by='Likes', ascending=False)
st.dataframe(popular_hashtags_df[['Hashtags', 'Likes']].head())
fig, ax = plt.subplots(figsize=(10, 6))
ax.bar(popular_hashtags_df['Hashtags'], popular_hashtags_df['Likes'])
# #Для большей наглядоности разделим эмоции на три категории 
# fig, ax = plt.subplots(figsize=(10, 6))
# ax.bar(popular_posts['Sentiment'], popular_posts['Likes'], color=plt.cm.viridis.colors)
ax.set_xlabel('Hashtags')
ax.set_ylabel('Likes')
ax.set_title('Самые популярные хэштеги')
plt.xticks(rotation=45, ha='right')  # Повернуть подписи на оси x для лучшей читаемости

# # Вывести диаграмму в Streamlit
st.pyplot(fig)

cleaned_df['Hashtags_count'] = cleaned_df['Hashtags'].apply(lambda x: len(x.split()))

# Создание бар-графика
fig, ax = plt.subplots()
sns.barplot(x='Sentiment', y='Hashtags_count', data=cleaned_df, ci=None)
plt.title('Количество хэштегов по настроению')

# Отображение бар-графика в Streamlit
st.pyplot(fig)

#подсчитать кол по постов по соцтесям 

platform_counts = cleaned_df['Platform'].value_counts()
#Convert Series to DataFrame
platform_counts_df = platform_counts.to_frame().reset_index()
platform_counts_df.columns = ['Platform', 'Count']
st.dataframe(platform_counts_df)


def sum_platform(dataframe, platform):
    selected_platform = dataframe[dataframe['Platform'].isin([platform])]
    if not selected_platform.empty:
        platform_sum = selected_platform['Count'].sum()
        st.text(f"{platform} sum: {platform_sum}")
        return platform_sum
    else:
        st.text(f"No matching rows found for {platform}.")
        return 0

# Пример использования для Twitter
twitter_sum = sum_platform(platform_counts_df, 'Twitter')
st.text(twitter_sum)
import streamlit as st

# Ваш DataFrame
platform_counts_df = pd.DataFrame({
    'Platform': ['Instagram', 'Twitter', 'Facebook'],
    'Count': [258, 243, 231]
})
# c помощью следующего графика можем понять какая соцсеть более популярная
# Создание столбчатой диаграммы
plt.figure(figsize=(8, 4))
plt.bar(platform_counts_df['Platform'], platform_counts_df['Count'], color=['blue', 'green', 'red'])
plt.xlabel('Платформа')
plt.ylabel('Сумма постов')
plt.title('Сумма постов для каждой платформы')
plt.show()
# Вывод столбчатой диаграммы в Streamlit
st.bar_chart(platform_counts_df.set_index('Platform'))



#Настроения, на какую тему было больше всего постов и меньше всего постов

#Отображение графика в Streamlit
st.pyplot(fig)

# popular_posts = cleaned_df[cleaned_df['Likes'] >= 80]

# # Выводим настроение для этих постов
# if not popular_posts.empty:
#     st.write("Самые популярные посты по количеству лайков:")
#     st.write(popular_posts[['Sentiment', 'Likes']])
# else:
#     st.write("Нет постов с более чем 40 лайками.")
# #Для большей наглядоности разделим эмоции на три категории 
# fig, ax = plt.subplots(figsize=(10, 6))
# ax.bar(popular_posts['Sentiment'], popular_posts['Likes'], color=plt.cm.viridis.colors)
# ax.set_xlabel('Настроение')
# ax.set_ylabel('Количество лайков')
# ax.set_title('Самые популярные посты по количеству лайков')
# plt.xticks(rotation=45, ha='right')  # Повернуть подписи на оси x для лучшей читаемости

# # Вывести диаграмму в Streamlit
# st.pyplot(fig)
# # Группируем данные по настроению и суммируем лайки
# sentiment_likes_sum = popular_posts.groupby('Sentiment')['Likes'].sum()

# # Построить круговую диаграмму
# fig, ax = plt.subplots(figsize=(8, 8))
# ax.pie(sentiment_likes_sum, labels=sentiment_likes_sum.index, autopct='%1.1f%%', startangle=140, colors=plt.cm.viridis.colors)
# ax.set_title('Распределение лайков по настроению')

# # Вывести диаграмму в Streamlit
# st.pyplot(fig)


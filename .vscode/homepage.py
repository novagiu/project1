import streamlit as st
import pandas as pd
import numpy as np
import os
import re

st.title("""Sentiments in social media""")
st.write('Social networks have become an integral part of our lives. They influence many areas: social life, business, marketing, etc. There people express their emotions and opinions, which are further reflected in these areas. Using real data from popular social networks as an example, we will try to analyze them and draw conclusions.')

st.write("""First, let's get acquainted with the data with which we will work. Here is collected information from social networks Facebook, Twitter(X) and Instagram for a period of time""")

#reading a csv file and checking if a csv is available at a given address
csv_path="C:\\Users\\Xiaomi\\Desktop\\project\\.vscode\\sentimentdataset.csv"

if os.path.exists(csv_path):
    sentiment_df=pd.read_csv(csv_path, header=0)
    st.write(sentiment_df)
    
else:(f"File '{csv_path}' not found. Please check the path.")
#first few rows of the data set
st.write(sentiment_df.head())

#display all the dataset information 
st.write(sentiment_df.info())

#Data cleaning
st.title('Data cleaning')
st.write('Для того, чтобы нам было удобнее работь с данными, необходимо их сперва очистить. В нашем случае удалить то, что не передает настроение в текстах, этим язляются местоимения, предлоги, артикли. Таже удалим смайлики и знаки.')

#remove the first two columns, since they do not contain useful information, but only index rows
st.write(sentiment_df.describe().T)

#Exploring the columns of a dataset
st.text(sentiment_df.columns)

##removing unnamed columns
cleaned_df=(sentiment_df.drop(columns=['Unnamed: 0.1','Unnamed: 0']))
st.write(cleaned_df)

#Check for missing values in a dataset
st.text(cleaned_df.isnull().sum())

#data cleaning function
def data_cleaning(Text):
    Text = Text.lower()
    Text = re.sub(r'[^\w\s]','',Text)
    return Text

#comparing the original text with the cleaned text

cleaned_df['Cleaned_Text'] = cleaned_df['Text'].apply(data_cleaning)
st.write(cleaned_df[['Text', 'Cleaned_Text']].iloc[0:4])







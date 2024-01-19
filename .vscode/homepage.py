import streamlit as st
import pandas as pd
import numpy as pandas
import os

st.title("""Sentiments in social medias""")
st.write('Социальные сети стали неотъемлемой частью нашей жизни. Они влияют на многие сферы: социальная жизнь, бизнес, маркетинг и др. Там люди выражают свои эмоции и мнения, которые вдальнейшем отражаются на этих сферах. На примере реальных данных из популярных социальных сетей мы попробуем проанализоровать их и сделать выводы.')

st.write('Для начала ознакомимся с данными, с которыми будет работать. Здесь собрана информация из социальных сетей Facebook, Twitter(X) и Instagram за период времени...')

#чтение csv файла и проверка наличия csv по данному адресу
csv_path="C:\\Users\\Xiaomi\\Desktop\\project\\.vscode\\sentimentdataset.csv"

if os.path.exists(csv_path):
    df=pd.read_csv(csv_path)
    st.write(df)
else:(f"File '{csv_path}' not found. Please check the path.")











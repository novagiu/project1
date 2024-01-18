import numpy as np
import pandas as pd 
import streamlit as st

list_of_numbers = [23,12,13, 1, -2]
series_of_numbers = pd.Series(list_of_numbers)
v_1 = [8, 12, -3, 78, -1]
v_2 = [90, 23, -212, 0, 21]

vector_1 = np.array(v_1)
vector_2 = np.array(v_2)

st.write("""
# My first app
Hello *world!*
""")



print(vector_1)
print(vector_2)
print(series_of_numbers)
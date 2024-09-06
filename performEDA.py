#By using this code u can perform EDA on any Dataset
import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import ydata_profiling
from streamlit_pandas_profiling import  st_profile_report
st.title('By using this app u can perform EDA on any Dataset')

df = sns.load_dataset('titanic')
pr = df.profile_report()

st_profile_report(pr)

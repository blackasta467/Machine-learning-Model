#importing libraries
import pip
pip.main(['install','seaborn'])
import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
from ydata_profiling import  ProfileReport
from streamlit_pandas_profiling import  st_profile_report
#webapp ka title
st.title("Exploratary Data Analysis")
#how to  upload file 
uploaded_file = st.file_uploader("Choose a CSV file", type=["csv"])
df = sns.load_dataset('titanic')
st.markdown("[Example csv file](https://github.com/blackasta467/Titanic-Dataset/blob/main/Titanic-Dataset.csv)")
#profiling reports for pandas

if  uploaded_file is not None:
    # Load CSV file
    def load_csv():
        
        csv = pd.read_csv(uploaded_file)
        return csv
    df = load_csv()
    pr = ProfileReport(df, explorative=True)
    st.header('**Input DF')
    st.write(df)
    st.write('---')
    st.header("**Profiling report with pandas")
    st_profile_report(pr)
else:
    st.info("waitnig for the file ")
    if st.button('Press to use data '):

        def load_data():
            a = pd.DataFrame(np.random.rand(100,5), 
                             columns=['age', 'banana', 'cat', 'dog', ' elephant'])
            return a 
        df =  load_data()
        pr = ProfileReport(df, explorative=True)
        st.header('**Input DF')
        st.write(df)
        st.write('---')
        st.header("**Profiling report with pandas")
        st_profile_report(pr)
        


            

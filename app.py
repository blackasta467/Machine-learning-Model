#importing libraires
import streamlit as st
import seaborn as sns
import pandas as pd 
import plotly.express as  px
from sklearn.ensemble import  RandomForestRegressor
from sklearn.metrics  import mean_absolute_error,r2_score,mean_squared_error



#make containers
header = st.container()
datasets = st.container()
features = st.container()
model_trainig = st.container()
plotly =  st.container()

with header:
    st.title("SHIP APP")
    st.write("In this project we will work on ship dataset")
    
with datasets:
    st.header("Importing dataset")
    df =  sns.load_dataset('titanic')
    df = df.dropna()
    st.write(df)
    #plots
    st.subheader("Different age of people in titanic")
    
    st.line_chart(df['age'].value_counts())
    
    st.subheader("Number of males and females  in titanic")
    
    st.bar_chart(df['sex'].value_counts())
    
    st.subheader("Difference in prices according to  class")

    st.bar_chart(df['pclass'].value_counts())

with features:
    st.header("These are  my app features")
    st.markdown('1.  **Data Visualization**: This app will have different plots to visualize the data')

    
with model_trainig:
    st.header("Model Training")
    #making columns
    input, display = st.columns(2)
    #making slider
    max_depth = input.slider("How many people u know?" , min_value=1 , max_value=100 , value =1 , step=1)

#n_estimaotrs
n_estimators = input.selectbox("How many trees should be there in RF?" ,options=[50,100,200,300,400,500,'No Limit'])
#adding list of features
# input.write(df.columns)
#input features from user
input_features = input.selectbox("Which feature should be used for prediction?" , options=df.columns)

#Machine learning model
model = RandomForestRegressor(max_depth=max_depth,n_estimators=n_estimators)

#define X and Y
X = df[[input_features]]
y = df[['fare']]
#fit model
model.fit(X,y)
pred = model.predict(X)
#display metrices
display.subheader("Mean absolute error of the model is: ")
display.write(mean_absolute_error(y,pred))
display.subheader("Mean squred error of the model is: ")
display.write(mean_squared_error(y,pred))
display.subheader("R_2 score of the model is: ")
display.write(r2_score(y,pred))

with plotly:
    st.title("Import another dataset")
    df1 = px.data.gapminder()
    st.write(df1)
    st.write(df1.columns)
    #summary stat
    # df1.describe()
    st.write(df1.describe())
    #data mangement
    year_option = df1['year'].unique().tolist()
    year =  st.selectbox("Select year", year_option, 0)
    # df1 =  df1[df1['year'] == year]
    #plotting 
    fig = px.scatter(df1 , x='gdpPercap', y='lifeExp', size='pop' , color='country', hover_name='country' , log_x=True, size_max=55, range_x=[100,100000] ,range_y=[20,90]
                     , animation_frame='year', animation_group='country')
    st.write(fig)


    

    

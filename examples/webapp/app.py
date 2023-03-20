from cgi import print_exception
from re import template
from tokenize import group
from unicodedata import name
from webbrowser import get
from pycaret.regression import load_model, predict_model
import streamlit as st
from readline import set_pre_input_hook
from sys import setprofile
import pandas as pd
import plotly as py
import plotly.figure_factory as ff
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
import altair as alt
import time
import requests
import streamlit as st
from streamlit_lottie import st_lottie
from streamlit_lottie import st_lottie_spinner
from st_aggrid import AgGrid
import streamlit.components.v1 as components


# Page config for the app, some basic configuration for the app
st.set_page_config(page_title='Diamond Price predictor', page_icon='diamond_shape_with_a_dot_inside', layout='wide',  initial_sidebar_state="expanded")


# This i will use to print Lottie file ( Lottie is a JSON-based animation file format that can be used in webapp applications)
def load_lottieurl(url: str):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()

   
# Predict function, this function will use the model to predict , i use as a funtion because we have 2 models in same app
def predictor(model, df): 
    
    predictions_data = predict_model(estimator = model, data = df) 
    return predictions_data['Label'][0] 

 #""" -------------------------------------------------------------------------------------------------HEADER-------------------------------------------------------------------------------- """

# Header configuration, this is the header of the app, lottie + title 
#first defined how much columns we need
col1, col2,  = st.columns(2)

#first column, this is the lottie file
with col1:
    lottie_url_hello = "https://assets1.lottiefiles.com/packages/lf20_krxbz1ww.json"
    lottie_hello = load_lottieurl(lottie_url_hello)
    st_lottie(lottie_hello, key="hello", height=150, width=150, loop = True)

#second column, this is the title
with col2:
    st.title("Jeweler's Smiles And Sparkles")


#"""--------------------------------------------------------------------------------------- MODEL CONFIGURATION-------------------------------------------------------------------------------- """

model = load_model('diamond_model')
model_quality = load_model('diamond_model_quality')


#"""--------------------------------------------------------------------------------------- SIDEBARD ATRIBUTES -------------------------------------------------------------------------------- """

price = st.sidebar.slider(label = "price", min_value =100.1, max_value = 99999.1 , value = 2.0, step = 10.0)
carat = st.sidebar.slider(label = 'carat', min_value =0.1, max_value = 5.1 , value = 2.0, step = 0.1)
cut = st.sidebar.selectbox(label = 'cut', options = ['fair', 'Good', 'Very Good', 'Premium', 'Ideal'])
color = st.sidebar.selectbox(label = 'color', options = ["D", "E", "F", "G", "H", "I", "J"])
clarity = st.sidebar.selectbox(label = 'clarity', options = ['I1', 'IF', 'VVS1', 'VVS2', 'VS1', 'SI2', 'VS2', 'SI1'])
depth = st.sidebar.slider(label = 'depth', min_value = 42.1, max_value = 80.1 , value = 2.0, step = 0.10)
table = st.sidebar.slider(label = 'table', min_value = 42.1, max_value = 99.1 , value = 2.0, step = 0.10)
x_dimension = st.sidebar.slider(label = 'x', min_value = 0.1, max_value = 5.1 , value = 2.0, step = 0.10)
y_dimension = st.sidebar.slider(label = 'y', min_value = 0.1, max_value = 5.1 , value = 2.0, step = 0.10)
z_dimension = st.sidebar.slider(label = 'z', min_value = 0.1, max_value = 5.1 , value = 2.0, step = 0.10)


#"""--------------------------------------------------------------------------------------- SELECTIVE TABS -------------------------------------------------------------------------------- """


tabs = st.tabs(["Information about our Diamonds Sets", "Calculate Price", "Calculate Carat", "Interactive catalogue"])


#"""--------------------------------------------------------------------------------------- TAB2 -------------------------------------------------------------------------------- """

tab_plots = tabs[1] #this is the second tab
with tab_plots:

    st.title('How much can you sell your diamond?')
    st.subheader('Based on the characteristics of your diamond, we will predict the price you can sell it for.')

    st.write('NOT FOR COMERCIAL USE, IS A TEST') 

    # This is the dataframe that will be used to predict the price
    features = {'carat': carat, 'cut': cut, 'color': color, 'clarity': clarity, 'depth': depth, 'table': table, 'x': x_dimension, 'y': y_dimension, 'z': z_dimension}

    df = pd.DataFrame(features, index = [0])
    prediction = predictor(model, df)       
    features_df  = pd.DataFrame([features])


    st.table(features_df)
    
    # This is the plot that will be used to predict the price if the user change the characteristics of the diamond and click on the button
    if st.button('Predict Price'):    
        prediction = predictor(model, features_df)    
        st.write('We will pay u   '+ str(prediction.round(2)) + '$')





#"""--------------------------------------------------------------------------------------- TAB3 -------------------------------------------------------------------------------- """

tab_plots = tabs[2] #this is the third tab
with tab_plots:

    st.title('How much carats have your diamond?')
    st.subheader('Based on the characteristics of your diamond, we will predict the carats.')

    st.write('NOT FOR COMERCIAL USE, IS A TEST') 


    features = {'price': price, 'cut': cut, 'color': color, 'clarity': clarity, 'depth': depth, 'table': table, 'x': x_dimension, 'y': y_dimension, 'z': z_dimension}

    df = pd.DataFrame(features, index = [0])
    prediction = predictor(model_quality, df)       
    features_df  = pd.DataFrame([features])


    st.table(features_df)

    if st.button('Predict Carats'):    
        prediction = predictor(model_quality, features_df)    
        st.write('Using the info inputing, the diamond have   '+ str(prediction.round(2)) + 'Carats')




#"""--------------------------------------------------------------------------------------- TAB1 -------------------------------------------------------------------------------- """


tab_plots = tabs[0] #this is the first tab
with tab_plots:

    st.title('Information about our diamonds sets')
    st.subheader('The distribution of the characteristics of our diamonds.')

    st.write('NOT FOR COMERCIAL USE, IS A TEST') 

    diamond_sets = pd.read_csv('diamonds.csv') # read data from diamonds.csv
    diamond_sets = diamond_sets.drop(['Unnamed: 0'], axis=1)


    cols = st.columns(2)
    with cols[0]:
        st.write("General information about our diamonds sets:")
        st.dataframe(diamond_sets.describe())

    with cols[1]:
        #Count each cut type
        #Plotting distribution of cut types
        cut = diamond_sets.groupby('cut').size().sort_values(ascending=False)
        st.write("The distribution of the cut types:")
        st.bar_chart(cut)

    
    cols = st.columns(2)
    with cols[0]:
        #pie chart of color
        color = diamond_sets.groupby('color').size().sort_values(ascending=False)
        st.write("Dstribution of the color types:")
        fig = px.pie(color, labels=color.index, values=color, hover_data=[color.index])
        st.plotly_chart(fig)

    with cols[1]:
        #plot correlation between price and carat
        st.write("Correlation between price and carat:")
        fig = px.scatter(diamond_sets, x='price', y='carat', size='depth', color='cut', template='plotly_dark', title='Price vs Carat')
        st.plotly_chart(fig)

           
        
#"""--------------------------------------------------------------------------------------- TAB4 -------------------------------------------------------------------------------- """
   
    tab_plots = tabs[3] #this is the fourth tab
    with tab_plots:
    
         components.iframe("https://datastudio.google.com/embed/reporting/dde3b825-cdae-49cc-a5c4-a09076e53e77/page/wWxyC", width=1200, height=900)
         
         AgGrid(diamond_sets)








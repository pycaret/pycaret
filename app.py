import streamlit as st
from pycaret.regression import *
import matplotlib.pyplot as plt
from PIL import Image
import pandas as pd

#import os.path
#import sys
#sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

st.balloons()

add_selectbox = st.sidebar.selectbox(
    'Menu',
    ('About', 'Upload Data', 'Base Models', 'Model Analysis', 'Predictions'))

image = Image.open('logo.png')
st.image(image)
st.warning("PyCaret 'UI' is a beta release announced with PyCaret version 1.0.1. (Not for production)")
st.markdown("## **What is PyCaret?**")
st.markdown("PyCaret is an open source low-code machine learning library in Python that aims to reduce the hypothesis to insights cycle time in a ML experiment. It enables data scientists to perform end-to-end experiments quickly and efficiently. In comparison with the other open source machine learning libraries, PyCaret is an alternate low-code library that can be used to perform complex machine learning tasks with only few lines of code.")
st.markdown("## **What is PyCaret UI?**")
st.markdown("PyCaret UI is front-end interface designed on streamlit. This is ideal for data professionals who prefers a no-code solution.")
st.markdown("## **Important Links**")
st.markdown("Official : https://www.pycaret.org")
st.markdown("GitHub : https://www.github.com/pycaret/pycaret")
st.markdown("YouTube : https://www.youtube.com/channel/UCxA1YTYJ9BEeo50lxyI_B3g")
st.markdown("LinkedIn : https://www.linkedin.com/company/pycaret")






#st.title('PyCaret AutoML Interface')

from pycaret.datasets import get_data
data = get_data('boston')

#st.title("Preprocessing")

s = setup(data, target = 'medv', session_id=123, silent=True, html=False)
'preprocess', s[0]

#st.title('Baseline Models')
lr = create_model('lr')
s[-2][0]


#best = regression.compare_models(blacklist=['catboost'])

#regression.plot_model(best)
#st.pyplot()
import pandas as pd
import pickle
import streamlit as st
st.set_page_config(page_title="House_Price_Predictor 🏠")
st.header("Welcome to Bangalore House Price Prediction💰💰")
df=pd.read_csv("cleaned_data.csv")
with open('RFmodel.pkl', 'rb') as f:
    model = pickle.load(f)
with st.container(border=True):
    col1, col2 = st.columns(2)
    loc=col1.selectbox("location",options=df['location'].unique())
    sqft=col2.number_input("Total Square Feet",min_value=300.0)
    bath=col1.number_input("Number of Bathrooms",max_value=6,min_value=1)
    bhk=col2.number_input("Number of Bedrooms",max_value=6,min_value=1)
    location=list(df['location'].unique())
    location.sort()
    input_values=[location.index(loc),sqft,bath,bhk]
    if st.button('Predict Price'):
        out=model.predict([input_values])
        st.subheader(f"The predicted price of the house is: ₹ {out[0]*1e5:.2f}")

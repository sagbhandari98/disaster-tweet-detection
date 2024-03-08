import streamlit as st
import numpy as np
import joblib

with open('rf_pipeline.pkl', 'rb') as f:
    loaded_pipeline = joblib.load(f) 

st.title('Disaster Tweets Predictor')

# Input text box for entering a tweet
tweet_input = st.text_input('Enter a tweet:')
submit_button = st.button('Submit')

if submit_button and tweet_input:

    processed_tweet = np.array([tweet_input])

    # Make predictions using the loaded pipeline
    prediction = loaded_pipeline.predict(processed_tweet)

    # Display the prediction
    st.subheader('Prediction:')
    if prediction == 0:
        st.success('Not a Disaster Tweet')
    else:
        st.error('Disaster Tweet')

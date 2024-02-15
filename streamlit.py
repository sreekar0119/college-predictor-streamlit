import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestClassifier
import streamlit_authenticator as stauth
def main():
    names = ['Hemanth', 'Sreekar']
    usernames = ['Hemanth', 'Sreekar']
    passwords = ['123', '456']

    hashed_passwords = stauth.Hasher(passwords).generate()

    authenticator = stauth.Authenticate(names, usernames, hashed_passwords,
        'some_cookie_name', 'some_signature_key', cookie_expiry_days=30)

    name, authentication_status, username = authenticator.login('Login', 'main')

    if authentication_status == False:
        st.error("Username/Password is incorrect")
        
    if authentication_status == None:
        st.warning("please enter your username and password")
    
    if authentication_status:

#Title of the app

        st.title('College Predictor Using the Scores')

#load dataset
        data = pd.read_csv('Admission_Predict.csv')
        df = pd.DataFrame(data)

#input features
        st.sidebar.subheader('input features')
        GRE = st.sidebar.slider('GRE_Score', 290, 340, 295)
        CGPA = st.sidebar.slider('CGPA_Score', 7.0, 9.9, 8.5)

        X = df[['GRE Score', 'CGPA']]
        y = df['university_name']

# Split the data into a training set and a testing set
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train a Random Forest Classifier
        model = RandomForestClassifier()
        model.fit(X_train, y_train)

# Predict university names for test data
        predictions = model.predict(X_test)

# Evaluate the model (You can use appropriate evaluation metrics)
        accuracy = (predictions == y_test).mean()

        a = GRE
        b = CGPA
        x = 1
        y = 1
# Use the model to predict university names for new applicants
        new_applicant = pd.DataFrame({'GRE Score': [a], 'CGPA': [b]})
        if  a == 340 and b == 9.9:
             image = Image.open('C:\Users\sreek\Desktop\ucla-cr-alamy.webp')
             st.image(image)
        predicted_university = model.predict(new_applicant) 
        st.write(predicted_university[0])
    authenticator.logout("Logout", "sidebar")
if __name__ == '__main__':
	main()


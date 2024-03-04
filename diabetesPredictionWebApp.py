# -*- coding: utf-8 -*-
"""
Created on Sun Mar  3 23:54:39 2024

@author: 14699
"""
## import librabries
import numpy as np
import pickle ## loading saved model
import streamlit as st ## run web app

## loading the saved model
## .load() --> Loads the model
## 'rb' --> binary data
loaded_model = pickle.load(open('trained_model.sav', 'rb')) ##C:\Sai_Donepudi/MachineLearning/Diabetes/

## creating a function for prediction
def diabetes_prediction(input_data):

    # changing the input_data to numpy array
    input_data_as_numpy_array = np.asarray(input_data)

    # reshape the array as we are predicting for one instance
    input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)

    prediction = loaded_model.predict(input_data_reshaped)
    print(prediction)

    if (prediction[0] == 0):
      return 'The person is not diabetic'
    else:
      return 'The person is diabetic'
  
def main():
    
    ## giving a title
    st.title('Diabetes Prediction Web App')
    
    ## getting the input data from the user
    Pregnancies = st.text_input('Number of pregnancies')
    Glucose = st.text_input('Glucose value')
    BloodPressure = st.text_input('Blood pressure value')
    SkinThickness = st.text_input('Skin thickness value')
    Insulin = st.text_input('Insulin value')
    BMI = st.text_input('BMI value')
    DiabetesPedigreeFunction = st.text_input('Diabetes pedigreee function value')
    Age = st.text_input('Age of the person')
    
    
    ## code for prediction
    diagnosis = ''
    
    ## creating a button for prediction
    if st.button('Check for Diabetes'):
        diagnosis = diabetes_prediction([Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age])
        
    st.success(diagnosis)
    
if __name__=='__main__':
    main()
    
    
    
    
    
    
    
    
    
    
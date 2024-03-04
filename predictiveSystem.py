# -*- coding: utf-8 -*-
"""
Created on Sun Mar  3 18:57:54 2024

@author: 14699
"""

import numpy as np
import pickle

## loading the saved model
## .load() --> Loads the model
loaded_model = pickle.load(open('C:\Sai_Donepudi/MachineLearning/Diabetes/trained_model.sav', 'rb'))

input_data = (4,110,92,0,0,37.6,0.191,30)

## changing the input data to a numpy array
input_data_as_numpy_array = np.asarray(input_data)

## reshape the array as we are predicting for one instance(input_data)
input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)

## standardize the input_data
##std_data = scaler.transform(input_data_reshaped)
##print(std_data)

prediction = loaded_model.predict(input_data_reshaped)
print(prediction)

if (prediction[0] == 0):
  print('The person is not diabetic!')
else:
  print('The person is diabetic!')
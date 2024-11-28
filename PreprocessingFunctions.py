import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn import preprocessing

#Encode the Make column as numeric so it can be processed
#using the OrdinalEncoder from Sklearn
def encodeMakeCol(carData):
    coder = preprocessing.OrdinalEncoder()
    cols = ['Make', 'Price']
    carData['Make'] = coder.fit_transform(carData[cols])
    return carData

#Encode the Color column as numeric so it can be processed
#using the OrdinalEncoder from Sklearn
def encodeColorCol(carData):
    coder = preprocessing.OrdinalEncoder()
    cols = ['Color','Price']
    carData['Color'] = coder.fit_transform(carData[cols])
    return carData

def getTransmissionAsFeature(types):
    feature = []
    for type in types:
        if type == 'Automatic':
            feature.append(True)
        else:
            feature.append(False)
    return pd.DataFrame(feature, columns=['Transmission Type'])

#Fill in missing values for Fuel Tank Capacity, Engine, and Seating Capacity
#using the SimpleImputer from Sklearn
def imputeValues(carData):
    imputer = SimpleImputer(strategy='median')
    missingValCol = ['Fuel Tank Capacity', 'Engine', 'Seating Capacity']
    carData[missingValCol] = imputer.fit_transform(carData[missingValCol])
    return carData
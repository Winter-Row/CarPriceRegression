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
#Takes the transmission column and turns into into numeric data
def getTransmissionAsFeature(types):
    feature = []
    for type in types:
        if type == 'Automatic':
            feature.append(1)
        else:
            feature.append(0)
    return pd.DataFrame(feature, columns=['Transmission Type'])

#Fill in missing values for Fuel Tank Capacity, Engine, and Seating Capacity
#using the SimpleImputer from Sklearn
def imputeValues(carData):
    imputer = SimpleImputer(strategy='median')
    missingValCol = ['Fuel Tank Capacity', 'Engine', 'Seating Capacity']
    carData[missingValCol] = imputer.fit_transform(carData[missingValCol])
    return carData
#Unused function due to the feature having little importance
def getOwnerNum(carData):
    f1 = []
    for owner in carData['Owner']:
        if(owner == 'First'):
            f1.append(1)
        elif(owner == 'Second'):
            f1.append(2)
        else:
            f1.append(0)
    carData['Owner'] = f1
    return carData
#gets thee age of the car from the yer column and subtracting it for the year the data was last updated
def getCarAge(carData):
    carData.insert(0, "Age", 2022 - carData['Year'])
    carData.drop('Year', axis=1)
    return carData

#Used for getting the outliers for the given data
#Outlier Help from https://www.kaggle.com/code/farzadnekouei/polynomial-regression-regularization-assumptions/notebook
def findOutliers(carData):
    outliers_indexes = []
    features = ['Age', 'Kilometer', 'Fuel Tank Capacity', 'Engine', 'Seating Capacity', 'Make', 'Color', 'Transmission', 'Price']
    carData = carData[features]

    for col in carData.columns:
        q1 = carData[col].quantile(0.30)
        q3 = carData[col].quantile(0.70)
        iqr = q3-q1
        maximum = q3 + (1.5 * iqr)
        minimum = q1 - (1.5 * iqr)
        outlier_samples = carData[(carData[col] < minimum) | (carData[col] > maximum)]
        outliers_indexes.extend(outlier_samples.index.tolist())
    
    outliers_indexes = list(set(outliers_indexes))
    return outliers_indexes
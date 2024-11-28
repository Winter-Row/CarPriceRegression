import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer

def getCarMakerAsNumeric(carData):
    f1 = []
    makers = carData['Make']
    carMakeList = makers
    carMakeList = list(dict.fromkeys(carMakeList))
    #print(len(carMakeList))
    for make in makers:
        for num in carMakeList:
            if make == num:
                f1.append(carMakeList.index(num))
    carData['Make'] = f1
    return carData

def getCarColorAsFeature(colors):
    feature = []
    colorList = list(dict.fromkeys(colors))
    for color in colors:
        for num in colorList:
            if color == num:
                feature.append(colorList.index(num))
    return feature

def getTransmissionAsFeature(types):
    feature = []
    for type in types:
        if type == 'Automatic':
            feature.append(True)
        else:
            feature.append(False)
    return pd.DataFrame(feature, columns=['Transmission Type'])

def imputeValues(carData):
    imputer = SimpleImputer(strategy='median')
    missingValCol = ['Fuel Tank Capacity', 'Engine', 'Seating Capacity']
    carData[missingValCol] = imputer.fit_transform(carData[missingValCol])
    return carData
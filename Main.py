import os, sys
from featureFunctions import *
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn import linear_model
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

features = []
car_X = []
car_Y = []
carMakeList = []

carDetailsFilePath = open(os.path.join(os.path.dirname(sys.argv[0]) + "/car-details.csv"))
carData = pd.read_csv(carDetailsFilePath)

car_Y = carData['Price']
car_X = carMakerFeature(carData['Make'])
car_X.insert(1,'Year',carData['Year'])
car_X.insert(2, 'KM', carData['Kilometer'])
print(car_X)

car_X_train, car_X_test, car_Y_train, car_Y_test = train_test_split(car_X, car_Y, test_size=0.3)

regr = linear_model.LinearRegression()
regr.fit(car_X_train, car_Y_train)
car_y_pred = regr.predict(car_X_test)

print("MSE: %.2f" % mean_squared_error(car_Y_test, car_y_pred))
print("R^2: %.2f\n" % r2_score(car_Y_test, car_y_pred))
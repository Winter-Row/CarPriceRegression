import os, sys
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from featureFunctions import *
from sklearn import linear_model
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

features = []
car_X = pd.DataFrame()
car_Y = pd.DataFrame()
carMakeList = []

carDetailsFilePath = open(os.path.join(os.path.dirname(sys.argv[0]) + "/car-details.csv"))
carData = pd.read_csv(carDetailsFilePath)

car_Y = carData['Price']

car_X = getCarMakerAsFeature(carData['Make'])
car_X.insert(0,'Year', carData['Year'])
car_X.insert(1, 'KM', carData['Kilometer'])
car_X.insert(2, 'Color', getCarColorAsFeature(carData['Color']))

carData['Fuel Tank Capacity'] = np.where(carData['Fuel Tank Capacity'].isna(), getAvg(carData['Fuel Tank Capacity']), carData['Fuel Tank Capacity'])
car_X.insert(3, 'Fuel Capacity', carData['Fuel Tank Capacity'])

carData['Seating Capacity'] = np.where(carData['Seating Capacity'].isna(), 5, carData['Seating Capacity'])
car_X.insert(4, 'Number of Seats', carData['Seating Capacity'])

car_X.insert(5, 'Transmission Type', getTransmissionAsFeature(carData['Transmission']))

carData['Engine'] = np.where(carData['Engine'].isna(), getAvg(carData['Engine']), carData['Engine'])
car_X.insert(6, 'Engine Power in CC', carData['Engine'])

print(car_X)

car_X_train, car_X_test, car_Y_train, car_Y_test = train_test_split(car_X, car_Y, test_size=0.3)

regr = linear_model.LinearRegression()
regr.fit(car_X_train, car_Y_train)
car_y_pred = regr.predict(car_X_test)

RandForestRegr = RandomForestRegressor()
RandForestRegr.fit(car_X_train, car_Y_train)
car_y_pred_RandForest = RandForestRegr.predict(car_X_test)

print("Linear Regression")
print("MSE: %.2f" % mean_squared_error(car_Y_test, car_y_pred))
print("R^2: %.2f\n" % r2_score(car_Y_test, car_y_pred))

print("Random Forest Regression")
print("MSE: %.2f" % mean_squared_error(car_Y_test, car_y_pred_RandForest))
print("R^2: %.2f\n" % r2_score(car_Y_test, car_y_pred_RandForest))
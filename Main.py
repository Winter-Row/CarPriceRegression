import os, sys
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PreprocessingFunctions import *
from sklearn import linear_model
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

features = []
car_X = pd.DataFrame()
car_Y = ()

#getting the file path for the CSV
carDetailsFilePath = open(os.path.join(os.path.dirname(sys.argv[0]) + "/car-details.csv"))
#Reading the data from a CSV using pandas
carData = pd.read_csv(carDetailsFilePath)

#Preprocessing
carData = imputeValues(carData)
carData = getCarMakerAsNumeric(carData)
carData['Color'] = getCarColorAsFeature(carData['Color'])
carData['Transmission'] = getTransmissionAsFeature(carData['Transmission'])

car_Y = carData['Price']

#getting the features for the model from carData
features = ['Year', 'Kilometer', 'Fuel Tank Capacity', 'Engine', 'Seating Capacity', 'Make', 'Color', 'Transmission']

car_X = carData[features]

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

# plt.plot(car_X['Kilometer'], car_Y, 'o')
# m, b = np.polyfit(car_X['Kilometer'], car_Y, 1)
# plt.plot(car_X['Kilometer'], m*car_X['Kilometer']+b)
# plt.show()

plt.style.use('default')
plt.style.use('ggplot')

#This plotting from https://aegis4048.github.io/mutiple_linear_regression_and_visualization_in_python
fig, ax = plt.subplots(figsize=(8, 4))

ax.plot(car_X_test['Kilometer'], car_y_pred_RandForest, color='k', label='Regression model')
ax.scatter(car_X_test['Kilometer'], car_y_pred_RandForest, edgecolor='k', facecolor='grey', alpha=0.7, label='Sample data')
ax.legend(facecolor='white', fontsize=11)
ax.set_title('$R^2= %.2f$' % r2_score(car_Y_test, car_y_pred_RandForest), fontsize=18)

fig.tight_layout()

plt.show()
import os, sys
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PreprocessingFunctions import *
from GraphFunctions import *
from sklearn import linear_model
from sklearn.metrics import mean_squared_error, r2_score, f1_score, explained_variance_score, max_error
from sklearn.ensemble import RandomForestRegressor, ExtraTreesClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

features = []
car_X = pd.DataFrame()
car_Y = []

#getting the file path for the CSV
carDetailsFilePath = open(os.path.join(os.path.dirname(sys.argv[0]) + "/car-details.csv"))
#Reading the data from a CSV using pandas
carData = pd.read_csv(carDetailsFilePath)

#Preprocessing
carData = imputeValues(carData)
carData = encodeMakeCol(carData)
carData = encodeColorCol(carData)
carData = getCarAge(carData)
#carData = getOwnerNum(carData)
carData['Transmission'] = getTransmissionAsFeature(carData['Transmission'])
car_Y = carData['Price']
#getting the features for the model from carData
features = ['Age', 'Kilometer', 'Fuel Tank Capacity', 'Engine', 'Seating Capacity', 'Make', 'Color', 'Transmission']
car_X = carData[features]

print(car_X)

car_X_train, car_X_test, car_Y_train, car_Y_test = train_test_split(car_X, car_Y, test_size=0.3)


#getting impact of the features
#gotten from https://www.kaggle.com/code/alimohammedbakhiet/forward-feature-selection
model = ExtraTreesClassifier()
model.fit(car_X_train, car_Y_train)
car_y__pred_Tree = model.predict(car_X_test)
feat_importances = pd.Series(model.feature_importances_, index=car_X_train.columns)
feat_importances = feat_importances.sort_values()
plotFeatureImportance(feat_importances)

regr = linear_model.LinearRegression()
regr.fit(car_X_train, car_Y_train)
car_y_pred = regr.predict(car_X_test)

RandForestRegr = RandomForestRegressor()
RandForestRegr.fit(car_X_train, car_Y_train)
car_y_pred_RandForest = RandForestRegr.predict(car_X_test)

print("Outliers Found: %d\n" % len(findOutliers(carData)))

print("Linear Regression")
print("MSE: %.2f" % mean_squared_error(car_Y_test, car_y_pred))
print("R^2: %.2f" % r2_score(car_Y_test, car_y_pred))
print("Max Error: %.2f\n" % max_error(car_Y_test, car_y_pred))

print("Random Forest Regression")
print("MSE: %.2f" % mean_squared_error(car_Y_test, car_y_pred_RandForest))
print("R^2: %.2f" % r2_score(car_Y_test, car_y_pred_RandForest))
print("Max Error: %.2f\n" % max_error(car_Y_test, car_y_pred_RandForest))

print("Extra Tree Classifier")
print("MSE: %.2f" % mean_squared_error(car_Y_test, car_y__pred_Tree))
print("R^2: %.2f" % r2_score(car_Y_test, car_y__pred_Tree))
print("Max Error: %.2f\n" % max_error(car_Y_test, car_y__pred_Tree))

plotCorrelationMap(carData)
#using linear regression
plotActualVsPredictedPrices(car_Y_test, car_y_pred)
#using random forest regression
plotActualVsPredictedPricesRandomForest(car_Y_test, car_y_pred_RandForest)
#plotting outliers
plotOutlierBoxPlot(carData)
#plotting graphs on price vs other features
plotDescriptionGrids(carData)
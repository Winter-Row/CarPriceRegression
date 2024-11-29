import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from PreprocessingFunctions import *

"""
Graphing help from 
https://www.kaggle.com/code/farzadnekouei/polynomial-regression-regularization-assumptions/notebook
https://www.kaggle.com/code/martejaz/04-car-price-prediction-model/notebook
"""

def plotActualVsPredictedPrices(test, pred):
    plt.scatter(test, pred)
    plt.xlabel("Actual Car Price")
    plt.ylabel("Predicted Car Price")
    plt.title("Actual Car Price vs. Predicted Car Price Linear Regression Test Data")
    plt.show()

def plotOutlierBoxPlot(carData):
    features = ['Age', 'Kilometer', 'Fuel Tank Capacity', 'Engine', 'Seating Capacity', 'Make', 'Color', 'Transmission', 'Price']
    carData = carData[features]
    sns.set_style('darkgrid')
    colors = ['#0055ff', '#ff7000', '#23bf00']
    sns.set_palette(sns.color_palette(colors))
    OrderedCols = np.concatenate([carData.columns.values])

    ax = plt.subplots(3, 4, figsize=(15,7),dpi=100)

    for i,col in enumerate(OrderedCols):
        x = i//4
        y = i%4
        if i<6:
            sns.boxplot(data=carData, y=col, ax=ax[x,y])
            ax[x,y].yaxis.label.set_size(15)
        else:
            sns.boxplot(data=carData, x=col, y='Price', ax=ax[x,y])
            ax[x,y].xaxis.label.set_size(15)
            ax[x,y].yaxis.label.set_size(15)

    plt.tight_layout()    
    plt.show()



def plotDescriptionGrids(carData):
    target = 'Price'
    carData['label'] = 'Normal'
    carData.loc[findOutliers,'label'] = 'Outlier'
    print(carData) 
    # Plot
    features = ['Age', 'Kilometer', 'Fuel Tank Capacity', 'Engine', 'Seating Capacity', 'Make', 'Color', 'Transmission']
    colors = ['#0055ff','#ff7000','#23bf00']
    sns.set_palette(sns.color_palette(colors))
    ax = plt.subplots(nrows=3 ,ncols=3, figsize=(10,10), dpi=100)

    for i in range(len(features)):
        x=i//3
        y=i%3
        sns.scatterplot(data=carData, x=features[i], y=target, hue='label', ax=ax[x,y])
        ax[x,y].set_title('{} vs. {}'.format(target, features[i]), size = 8)
        ax[x,y].set_xlabel(features[i], size = 6)
        ax[x,y].set_ylabel(target, size = 6)
        ax[x,y].grid()

    ax[2, 2].axis('off')
    plt.tight_layout()
    plt.show()
import pandas as pd
import numpy as np
def carMakerFeature(makers):
    f1 = []
    carMakeList = makers
    carMakeList = list(dict.fromkeys(carMakeList))
    #print(len(carMakeList))
    for make in makers:
        for num in carMakeList:
            if make == num:
                f1.append(carMakeList.index(num))
    f1 = pd.DataFrame(f1, columns=['MakerNumbers'])
    return f1

def carColorFeatures(colors):
    feature = []
    colorList = list(dict.fromkeys(colors))
    for color in colors:
        for num in colorList:
            if color == num:
                feature.append(colorList.index(num))
    return pd.DataFrame(feature, columns=['Color'])

def getAvgFuelCapacity(fuelCapacities):
    fuelList = list(fuelCapacities)
    avg = 0
    for num in fuelList:
        if(not np.isnan(num)):
            avg += num
    return round(avg / len(fuelList), 2)
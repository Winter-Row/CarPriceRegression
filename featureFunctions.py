import pandas as pd
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
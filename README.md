**COMP 4112 – Introduction to Data Science**

**Impact of Attributes on Car Price’s**

**Regression Model**

**By Hunter Roseborough**

**Abstract**

The purpose of this project was to find what aspects of a car have the biggest effect on price and create a predictive model to determine the accuracy of those features impacting price. The aim was to use different regression models to find out what features will most accurately depict a car’s price. The dataset used was a car details csv taken from Kaggle.com for which the model was trained and tested on. The methods used were splitting the data into a training and testing set and using three supervised learning techniques Linear regression, Random Forest regression, and Extra Tree Classifier. These were all used for the creation and training of the regression models. The model with the highest accuracy was the Random Forest Regressor that had a R2 score of 0.83. Next was the Linear Regression model that came back with a score of 0.57. and finally, the Extra Tree Classifier that came back with an accuracy score of 0.51. In conclusion, only one model came back with a reasonable accuracy score. The other two models’ score could use some improvement to get a better measurement of what is impacting the overall price of vehicles. The reason for this could be the dataset itself as it contains a sufficient number of out liars around 700. Or the features of the vehicles themselves in the dataset do not have very much impact on the price of a car. The model still provides good information on how some features impact the price of a car more than other ones.

**Introduction**

Understanding cars can be challenging for some people, especially when it comes to buying them, there’s a lot of choices and so much going on that it can be difficult to know why they are so expensive. This report looks at what is the cause of the pricey automobile by showing what has the most impact on a vehicle’s price. Also create a predictive model to test the accuracy and performance of these features so you can know what has the most effect. With this model it will be able to help you further understand car prices so the next time you go and buy a car you have further knowledge of vehicles and their features that contribute to their prices.

**Methods**

**Data Collection**

Kaggle is a data science platform containing hundreds of thousands of datasets specifically for data science. It contains the Vehicle dataset which has 4 csv this report focused on the car details v4. The dataset was made from web scraping and its main purpose is to be used for price prediction for linear regression. It consists of 20 columns and a little over 2000 rows with various attributes of cars like, “make”, “model”, “year”, “price”, “kilometers”, “fuel type”, “transmission type”, etc. The main focus for this report was attributes related to the price of cars. There are 8 attributes specific that were chosen. Make, year, kilometers, fuel capacity, number of Seats, color, transmission type, and engine power in ccs. All these attributes are converted to integers in preprocessing if they were not in integer format already.

**Data Preprocessing**

The dataset was already given as a csv file so there was no need to convert the whole file to a proper format, but the data did need to be brought into python for preprocessing. The data was then brought into python by converting the csv into a pandas data frame using the pandas library. The data set did contain missing values, so those values were handled by using a simple imputer from the scikit learn library. The simple imputer replaces missing values with a descriptive statistic, for this report I used the mean statistic to fill in any missing values for fuel tank capacity, engine power in ccs, and seating capacity as these were the only columns used that had missing values. Next, not all columns were in a numerical format. The “make”, “color”, and “transmission” columns were in a text format, so these needed to be converted. For the “make” and “color” columns I used the ordinal encoder from scikit learn. This encoder takes an input and returns the features as a column of integers ranging from 0 to n categories – 1. The “transmission” column there was only 2 options, so the “automatic” value was assigned 1 and “manual” value was assigned 0. All other used columns were already in numerical formats able to be used for feature construction right away.

**Feature Engineering**

The features chosen for the model were mostly selected right from the data frame of car data. The “year” attribute was modified a bit to give the age of car from 2022 as this is when the data was last updated. The other features used were picked out of the data frame using forward selection picking the attributes that had the greatest impact on the R2 score and had the most importance on “price”. Each column in the data frame contained information related to the price of the car. The data was mostly in integer format apart from a few columns which needed to be converted to integers in preprocessing. Aswell some columns contained missing data, so some information needed to be imputed into the data frame.

**Techniques**

The report focuses on two regression techniques and one classification technique to train the model on how the features impact “price”. The two regressors are Linear Regression and Random Forest regression, and the classification is the Extra Tree Classifier. The reason for using the different techniques is to see which regression model would perform the best and produce the most accurate results. Extra Tree Classifier is used to determine which feature has the greatest impact on “price” while also testing its accuracy and performance.

**Technique descriptions**

**Linear Regression:** Determines the relationship between two or more variables and creates a line of best fit to look at the data. One variable must be the dependent variable that is the scaler response variable, and the other variables are the independent variable or referred to as the explanatory variable this can be one or multiple variables used to explain the dependent variable.

**Random Forest Regression:** An estimator that fits multiple decisions trees regressors on different sub samples of the dataset. Does this by using averaging to improve accuracy and control the over fitting of the model.

**Extra Tree Classifier:** Similar to Random Forest Regression but uses a number of randomized decision trees on the various sub samples to improve the accuracy of the model and control the over fitting as well.

**Training**

The training process was done by taking the data from the used features and splitting the into a training set and a testing set. The training set is used for model fitting and the testing set is used to see how accurate the model is on the training data. The split was setup as a 70% training set and 30% testing set as this is a standard split in machine learning. Which is needed for creating a generalized model of unseen data. All the techniques used to test for the accuracy and the model performance, used the same testing and training data with the same split.

**Evaluation**

**R-Squared Score:** This metric is frequently used in regression models to help determine the percentage in which the dependent variables variation on which independent variables contribute too. R square is useful for seeing the overall effectiveness of a regression model.

**Mean Squared Error:** Used frequently in regression to assess how well the model’s prediction works. It measures the square root of the average differences between the data sets actual values and its predicted ones.

**Results**

**Sample Size**

The datasets total row count is 2060. Splitting this into the 70% training set and the 30% testing size gives us a training set of 1441 with a testing size of 618.

**Performance of models**

The results for each training metric using the selected models Linear Regression, Random Forest Regression, and Extra Tree Classifier did not provide ideal scores back. Specifically, MSE came back with quite high numbers for its results while the R2 scores were mainly mediocre results. But the information provided was still relevant to the effects of the features had on “price”. The results will be discussed below:

**Linear Regression**

* **R-Squared:** Linear regression came back with an R2 score of 0.57. This indicates that the model has about 57% effectiveness on the features involved in the testing set. So around half the features do have an impact on “price”.
* **Mean Squared Error:** MSE came back with quite a high score of 3470139565483.93. I am not sure why this score was so high as different measures were taken to locate the reason for this. This while be discussed more in the discussion section.

**Random Forest Regression**

* **R-Squared:** Random Forests R2 score came back with score of 0.83. Indicating a higher score so this model has a greater effectiveness rate of 83% making it a better performing model for the data.
* **Mean Squared Error:** Random Forest also came back with a high number for MSE with a score of 879486346316.73. Note that it is a lower number being that the Random Forest Regressor did perform better overall.

**Extra Tree Classifier**

* **R-Squared:** Extra Tree Classifier came back with a similar score as to Linear regression being 0.51. Indicating a mediocre score of 51%. So again, the feature effectiveness was about half.
* **Mean Squared Error:** The MSE score for the Extra Tree had a similar number to the Linear Regression being quite high. The score came back with 2072832160993.62.

**Discussion**

**Analysis of Results**

The results provided from the Linear Regression, Random Forest Regression, and Extra Tree Classifier models offered a good insight into how the aspects of a car can influence the price point of a vehicle. Random Forest Regression displayed the best score of 83% on the R2 score showing that it performed the best on the data used. Whereas Linear Regression and Extra Tree Classifier performed poorly in comparison only around 50%. A main concern with the results in the MSE scores as this score came back with quite high numbers. A few actions were taken to see if the results could be adjusted like rounding the target “price” column, trying to leave out certain features but these attempts did not affect the score. Rounding the “price” down did reduce the number slightly but did not have a great enough effect. The conclusion I came to is the problem lies in the data itself that was used. Even with this issue the results provided still hold significant value in showing how these attributes affect the “price” point of cars.

**What Could be done Next**

A few ways to enhance the regression model on the car prices could be to try and explore further in feature engineering. Looking for more features that better impact car prices or developing more in-depth features like the car’s size, fuel usage, and other features like this could help enhance the model. Another way to enhance the model could be to look at other data sets that contain different attributes or more up to date states to help create more refined features. Normalizing the data could also be done to see how it affects the overall score the model produces and if there are any improvements to it. Finally different techniques could be used for encoding textural data or even for imputing missing data from the data sets. This could also lead to having a more enhanced model which may yield better results.

**Conclusion**

Overall, I was able to develop a model that can help describe how certain attributes of a car contribute to the price point of that vehicle. Which could help people understand a bit more about why vehicles cost so much and aspects to look out for when purchasing a car. The Random Forest Regressor showed the best performance out of the other models and Extra Tree Classifier had the poorest performance. Linear Regression was not much better but still provided good information about the data. There is still room to improve the model with more in-depth features and trying different datasets but overall, the data used provided good features for determining the effectiveness of the attributes on the price of vehicles.

**Graphs**

![image](https://github.com/user-attachments/assets/3a867e25-6717-496c-935f-49e344592d7a)
![image](https://github.com/user-attachments/assets/8fb79de5-e02f-4a37-9b28-94b973916d8e)
![image](https://github.com/user-attachments/assets/e33b4c93-6745-49aa-8360-e7076061172f)
![image](https://github.com/user-attachments/assets/06b63d61-1a0c-42a5-a803-b0807aad7dd2)
![image](https://github.com/user-attachments/assets/e3d23b64-cb4a-44eb-8fac-462fe8b9eff8)
![image](https://github.com/user-attachments/assets/4a920828-aaaa-4d0a-844f-e2e38eb5001a)



**Tables**

**Features**

| **Feature** | **Data Type** |
| --- | --- |
| Age | Numeric |
| Kilometer | Numeric |
| Fuel Tank Capacity | Numeric |
| Engine Power in CC | Numeric |
| Seating Capacity | Numeric |
| Make | Textural (Encoded to Numeric) |
| Color | Textural (Encoded to Numeric) |
| Transmission | Textural (Encoded to Numeric) |

**Scores**

**R2**

| **Model** | **Score** |
| --- | --- |
| Linear Regression | 0.57 |
| Random Forest Regression | 0.83 |
| Extra Tree Classifier | 0.51 |

**MSE**

| **Model** | **Score** |
| --- | --- |
| Linear Regression | 3470139565483.93 |
| Random Forest Regression | 879486346316.73 |
| Extra Tree Classifier | 2072832160993.62 |

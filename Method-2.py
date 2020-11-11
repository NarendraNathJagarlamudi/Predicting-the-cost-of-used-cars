#MachineHack
#Importing Libraries 
import numpy as np
import matplotlib.pyplot as plt 
import pandas as pd
import statistics 
# import re
from sklearn.metrics import make_scorer
from sklearn.model_selection import cross_val_score


dataset = pd.read_excel('Data_Train.xlsx')
testdata = pd.read_excel('Data_Test.xlsx')

#"""Splitting name into 2 features, brand and model"""

#Training Set
names = list(dataset.Name)
brand = []
model = []
for i in range(len(names)):
   try:
       brand.append(names[i].split(" ")[0].strip())
       try:
           model.append(" ".join(names[i].split(" ")[1:]).strip())
       except:
           pass
   except:
       print("ERR ! - ", names[i], "@" , i)
dataset["Brand"] =  brand
dataset["Model"] = model
dataset.drop(labels = ['Name'], axis = 1, inplace = True)

#Test Set
names = list(testdata.Name)
brand = []
model = []
for i in range(len(names)):
   try:
       brand.append(names[i].split(" ")[0].strip())
       try:
           model.append(" ".join(names[i].split(" ")[1:]).strip())
       except:
           pass
   except:
       print("ERR ! - ", names[i], "@" , i)
testdata["Brand"] =  brand
testdata["Model"] = model
testdata.drop(labels = ['Name'], axis = 1, inplace = True)


#""" Removing the  texts and converting to integer''"""

# Training Set
'''mileage = list(dataset.Mileage)
for i in range(len(mileage)):
   try :
       mileage[i] = float(mileage[i].split(" ")[0].strip())
   except:
       mileage[i] = np.nan
dataset['Mileage'] = mileage'''

#Missing Values
dataset.apply(lambda dataset: sum(dataset.isnull()))
testdata.apply(lambda testdata: sum(testdata.isnull()))
#New_Price feature has a huge number of null values
#It is also possible to fill the nulls with zeros or unit values to check its significance in predictions
del dataset['New_Price']
del testdata['New_Price']

#Re-ordering the columns
dataset = dataset[['Brand', 'Model', 'Location', 'Year', 'Kilometers_Driven', 'Fuel_Type', 'Transmission',
      'Owner_Type', 'Mileage', 'Engine', 'Power', 'Seats', 'Price']]
testdata = testdata[['Brand', 'Model', 'Location', 'Year', 'Kilometers_Driven', 'Fuel_Type', 'Transmission',
      'Owner_Type', 'Mileage', 'Engine', 'Power', 'Seats']]

#Finding all unique categories
#'Brand', 'Model', 'Location','Fuel_Type', 'Transmission', 'Owner_Type'

all_brands = list(set(list(dataset.Brand) + list(testdata.Brand)))
all_models = list(set(list(dataset.Model) + list(testdata.Model)))
all_locations = list(set(list(dataset.Location) + list(testdata.Location)))
all_fuel_types = list(set(list(dataset.Fuel_Type) + list(testdata.Fuel_Type)))
all_transmissions = list(set(list(dataset.Transmission) + list(testdata.Transmission)))
all_owner_types = list(set(list(dataset.Owner_Type) + list(testdata.Owner_Type)))

del dataset['Model']
del testdata['Model'] 

# Categorical Variable
from sklearn.preprocessing import LabelEncoder
labelencoder_x = LabelEncoder()
dataset['Brand'] = labelencoder_x.fit_transform(dataset['Brand'])
#dataset['Model'] = labelencoder_x.fit_transform(dataset['Model'])
dataset['Location'] = labelencoder_x.fit_transform(dataset['Location'])
dataset['Fuel_Type'] = labelencoder_x.fit_transform(dataset['Fuel_Type'])
dataset['Transmission'] = labelencoder_x.fit_transform(dataset['Transmission'])
dataset['Owner_Type'] = labelencoder_x.fit_transform(dataset['Owner_Type'])


#For testdata 
testdata['Brand'] = labelencoder_x.fit_transform(testdata['Brand'])
#testdata['Model'] = labelencoder_x.fit_transform(testdata['Model'])
testdata['Location'] = labelencoder_x.fit_transform(testdata['Location'])
testdata['Fuel_Type'] = labelencoder_x.fit_transform(testdata['Fuel_Type'])
testdata['Transmission'] = labelencoder_x.fit_transform(testdata['Transmission'])
testdata['Owner_Type'] = labelencoder_x.fit_transform(testdata['Owner_Type'])



x = dataset.iloc[:,0 : -1]
y = dataset.iloc[:, -1]

#Dealing with missing value 
#mean
x["Mileage"] = x["Mileage"].replace(np.nan, x["Mileage"].mean())
x["Engine"] = x["Engine"].replace(np.nan, x["Engine"].mean())
testdata["Engine"] = testdata["Engine"].replace(np.nan, testdata["Engine"].mean())
x["Power"] = x["Power"].replace(np.nan, x["Power"].mean())
testdata["Power"] = testdata["Power"].replace(np.nan, testdata["Power"].mean())

#Mode
x["Seats"] = x["Seats"].replace(np.nan, statistics.mode(x["Seats"]))
testdata["Seats"] = testdata["Seats"].replace(np.nan, statistics.mode(testdata["Seats"]))

#Checking for missing value
x.apply(lambda x: sum(x.isnull()))
testdata.apply(lambda testdata: sum(testdata.isnull()))

#a new column depicting the years of operation of a store.
x['Years_Old'] = 2019 - x['Year']
testdata['Years_Old'] = 2019 - testdata['Year']

del x['Year']
del testdata['Year']

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
x = sc_X.fit_transform(x)
testdata = sc_X.transform(testdata)

#Another way of feature scaling on some selected variables

x['Kilometers_Driven'] = (x['Kilometers_Driven'] - x['Kilometers_Driven'].mean())/(statistics.stdev(x['Kilometers_Driven']))
x['Mileage'] = (x['Mileage'] - x['Mileage'].mean())/(statistics.stdev(x['Mileage']))
x['Engine'] = (x['Engine'] - x['Engine'].mean())/(statistics.stdev(x['Engine']))
x['Power'] = (x['Power'] - x['Power'].mean())/(statistics.stdev(x['Power']))

testdata['Kilometers_Driven'] = (testdata['Kilometers_Driven'] - testdata['Kilometers_Driven'].mean())/(statistics.stdev(testdata['Kilometers_Driven']))
testdata['Mileage'] = (testdata['Mileage'] - testdata['Mileage'].mean())/(statistics.stdev(testdata['Mileage']))
testdata['Engine'] = (testdata['Engine'] - testdata['Engine'].mean())/(statistics.stdev(testdata['Engine']))
testdata['Power'] = (testdata['Power'] - testdata['Power'].mean())/(statistics.stdev(testdata['Power']))


#We will test this later 
#One Hot Coding:
x = pd.get_dummies(x, columns=['Brand','Location','Fuel_Type','Transmission','Owner_Type'])

#Taking care of dummy variablr trap
del x['Brand_0']
del x['Brand_29']
del x['Brand_30']

del x['Location_0']
del x['Fuel_Type_0']
del x['Fuel_Type_4']

del x['Transmission_0']
del x['Owner_Type_0']

#One Hot Coding:
testdata = pd.get_dummies(testdata, columns=['Brand','Location','Fuel_Type','Transmission','Owner_Type'])

del testdata['Brand_0']
del testdata['Location_0']
del testdata['Fuel_Type_0']
del testdata['Transmission_0']
del testdata['Owner_Type_0']



#Visualising Data
import numpy as np
# import matplotlib.pyplot as plt

# Create data
N =6019
xaxis = x['Power']
yaxis = np.random.rand(N)
colors = (0,0,0)
area = np.pi*5

# Plot
plt.scatter(xaxis,yaxis, s=area, c=colors, alpha=0.5)
plt.title('Scatter plot')
plt.xlabel('xaxis')
plt.ylabel('yaxis')
plt.show()



#Splitting training data
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 0)

# Applying PCA 
#Will test later, also we will include New_Price variable 

from sklearn.decomposition import PCA
pca = PCA(n_components = None)
X_train = pca.fit_transform(X_train)
X_test = pca.transform(X_test)
explained_variance = pca.explained_variance_ratio_
# Do not worked well in this solution

# Applying LDA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
lda = LDA(n_components = None)
X_train = lda.fit_transform(X_train, y_train)
X_test = lda.transform(X_test)
explained_variance = lda.explained_variance_ratio_
# Do not worked well in this solution


#Model Building



#Linear Regression Model 
#Scaling do not effect Linear Regression
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)

y_pred = regressor.predict(X_test)

#Calculating Score
import numpy as np
# import math
# from keras import backend as K
# import tensorflow as tf

def rmsle(y, y0):
    assert len(y) == len(y0)
    return np.sqrt(np.mean(np.power(np.log1p(y)-np.log1p((y0** 2) ** 0.5), 2)))

error = rmsle(y_test, y_pred)
score = 1- error
score
# Score = 0.5216324697600452
# Public Score = .7741 

# Fitting Polynomial Regression to the dataset
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
poly_reg = PolynomialFeatures(degree = 3)
X_poly = poly_reg.fit_transform(X_train)
poly_reg.fit(X_poly, y_train)
lin_reg_2 = LinearRegression()
lin_reg_2.fit(X_poly, y_train)

y_pred = lin_reg_2.predict(poly_reg.fit_transform(X_test))

#Calculating Score
import numpy as np


def rmsle(y, y0):
    assert len(y) == len(y0)
    return np.sqrt(np.mean(np.power(np.log1p(y)-np.log1p((y0** 2) ** 0.5), 2)))

error = rmsle(y_test, y_pred)
score = 1- error
score
#score with degree 2 = 0.6684419139232086
#score with PCA3 with degree 2 = 0.6286835130541
#Public score with degree 2 = 0.7899
#score with degree 3 = 0.674275028384

#predicting the testset result 
y_pred = lin_reg_2.predict(poly_reg.fit_transform(testdata))
y_pred = pd.DataFrame( y_pred)
y_pred.rename(columns={ 0: 'Price'}, inplace=True)
y_pred.to_excel("Polyregwithdegree3.xlsx", index = False)



#SVR model 
from sklearn.svm import SVR
regressor = SVR(kernel = 'rbf')
regressor.fit(X_train,y_train)
y_pred = regressor.predict(X_test)
#Calculating Score
import numpy as np
# import math
# from keras import backend as K
# import tensorflow as tf

def rmsle(y, y0):
    assert len(y) == len(y0)
    return np.sqrt(np.mean(np.power(np.log1p(y)-np.log1p((y0** 2) ** 0.5), 2)))

error = rmsle(y_test, y_pred)
score = 1- error
score
#score with poly = 0.7001132874844693
#public score poly = 0.8355
#score with rbf = 0.7624903786222172 
#public score rbf = 0.8826


y_pred = regressor.predict(testdata)
y_pred = pd.DataFrame( y_pred)
y_pred.rename(columns={ 0: 'Price'}, inplace=True)
y_pred.to_excel("SVRrbf.xlsx", index = False)

# Decision Tree Regression
from sklearn.tree import DecisionTreeRegressor
regressor = DecisionTreeRegressor()
regressor.fit(X_train,y_train)

y_pred = regressor.predict(X_test)
#Calculating Score
import numpy as np
# import math
# from keras import backend as K
# import tensorflow as tf

def rmsle(y, y0):
    assert len(y) == len(y0)
    return np.sqrt(np.mean(np.power(np.log1p(y)-np.log1p((y0** 2) ** 0.5), 2)))

error = rmsle(y_test, y_pred)
score = 1- error
score
#score = 0.7468277340399
#Public = 0.8655

y_pred = regressor.predict(testdata)
y_pred = pd.DataFrame( y_pred)
y_pred.rename(columns={ 0: 'Price'}, inplace=True)
y_pred.to_excel("tree.xlsx", index = False)

#Random Forest
from sklearn.ensemble import RandomForestRegressor
regressor = RandomForestRegressor(n_estimators = 512)

regressor.fit(X_train,y_train) # if running cross val score then don't run this cell

y_pred = regressor.predict(X_test)

#Calculating Score
import numpy as np
# import math
# from keras import backend as K
# import tensorflow as tf
# from sklearn.metrics import make_scorer

def rmsle(y, y0):
    assert len(y) == len(y0)
    return np.sqrt(np.mean(np.power(np.log1p(y)-np.log1p((y0** 2) ** 0.5), 2)))

rmsle_score = make_scorer(rmsle)

error = rmsle(y_test, y_pred)
score = 1- error
score
#Score with n = 8 = 0.8069701860119882
#Score with n = 16 = 0.8029621552432834
#Score with n = 128 = 0.8187293703773127
#Score with n = 128 = 0.8208515600899219

#Score public 8 = 0.8912

# from sklearn.model_selection import cross_val_score
error = cross_val_score(estimator = regressor, X = X_train, y = y_train, cv = 10, scoring = rmsle_score)
error.mean()
score = 1- error
score.mean()


#predicting the testset result 
y_pred = regressor.predict(testdata)
y_pred = pd.DataFrame( y_pred)
y_pred.rename(columns={ 0: 'Price'}, inplace=True)
y_pred.to_excel("forest2048.xlsx", index = False)


# Fitting XGBoost to the Training set
from xgboost import XGBRegressor
regressor = XGBRegressor()
regressor.fit(X_train, y_train)
y_pred = regressor.predict(X_test)

#Score
def rmsle(y, y0):
    assert len(y) == len(y0)
    return np.sqrt(np.mean(np.power(np.log1p(y)-np.log1p((y0** 2) ** 0.5), 2)))

error = rmsle(y_test, y_pred)
score = 1- error

#predicting the testset result 
y_pred = regressor.predict(testdata)
y_pred = pd.DataFrame( y_pred)
y_pred.rename(columns={ 0: 'Price'}, inplace=True)
y_pred.to_excel("forestx.xlsx", index = False)


# Importing the Keras libraries and packages
# import keras
from keras.models import Sequential
from keras.layers import Dense

# Initialising the ANN
regressor = Sequential()

# Adding the input layer and the first hidden layer
regressor.add(Dense(output_dim = 6, init = 'uniform', activation = 'relu', input_dim = 12))

# Adding the second hidden layer
regressor.add(Dense(output_dim = 6, init = 'uniform', activation = 'relu'))
# Adding the second hidden layer
regressor.add(Dense(output_dim = 6, init = 'uniform', activation = 'relu'))

# Adding the output layer
regressor.add(Dense(output_dim = 1, init = 'uniform', activation = 'linear'))

# Compiling the ANN
regressor.compile(optimizer = 'adam', loss = 'mean_absolute_error', metrics = ['mean_absolute_error'])

# Fitting the ANN to the Training set
regressor.fit(X_train, y_train, batch_size = 10, nb_epoch = 100)

# Predicting the Test set results
y_pred = regressor.predict(X_test)
y_pred.reshape([1204,]) 
y_pred.shape 

#Calculating Score
import numpy as np
# import math
# from keras import backend as K
# import tensorflow as tf
from sklearn.metrics import make_scorer

def rmsle(y, y0):
    assert len(y) == len(y0)
    return np.sqrt(np.mean(np.power(np.log1p(y)-np.log1p((y0** 2) ** 0.5), 2)))

error = rmsle(y_test, y_pred)
score = 1- error

# Public = 0.8753
#predicting the testset result 
y_pred = regressor.predict(testdata)
y_pred = pd.DataFrame( y_pred)
y_pred.rename(columns={ 0: 'Price'}, inplace=True)
y_pred.to_excel("NNhidden21.xlsx", index = False)











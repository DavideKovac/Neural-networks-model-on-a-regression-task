#!/usr/bin/env python
# coding: utf-8

# <h1>Neural Network Regression Project</h1>
# 
# This is a world real data set from the historical housing data in King County,USA (where Seattle is)
# The data is from this Kaggle link: https://www.kaggle.com/harlfoxem/housesalesprediction
# <h3><b>Feature Columns</h3></b>
# 
# * id - Unique ID for each home sold
# * date - Date of the home sale
# * price - Price of each home sold
# * bedrooms - Number of bedrooms
# * bathrooms - Number of bathrooms, where .5 accounts for a room with a toilet but no shower
# * sqft_living - Square footage of the apartments interior living space
# * sqft_lot - Square footage of the land space
# * floors - Number of floors
# * waterfront - A dummy variable for whether the apartment was overlooking the waterfront or not
# * view - An index from 0 to 4 of how good the view of the property was
# * condition - An index from 1 to 5 on the condition of the apartment,
# * grade - An index from 1 to 13, where 1-3 falls short of building construction and design, 7 has an average level of construction and design, and 11-13 have a high quality level of construction and design.
# * sqft_above - The square footage of the interior housing space that is above ground level
# * sqft_basement - The square footage of the interior housing space that is below ground level
# * yr_built - The year the house was initially built
# * yr_renovated - The year of the house’s last renovation
# * zipcode - What zipcode area the house is in
# * lat - Lattitude
# * long - Longitude
# * sqft_living15 - The square footage of interior housing living space for the nearest 15 neighbors
# * sqft_lot15 - The square footage of the land lots of the nearest 15 neighbors

#import helpful libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('whitegrid')

#Read the data
df=pd.read_csv('kc_house_data.csv')

#Find Null value in every columns
df.isnull().sum()

#get info about the data
df.describe()

df.info()

df.head()

#EDA

#Check house price distribution(HistPrice.png)

plt.figure(figsize=(12,6))
sns.distplot(df['price'])

#Check number of bedroom distrubution(BedHist.png)
plt.figure(figsize=(12,6))
sns.countplot(df['bedrooms'])

#Check correlation with price

df.corr()['price'].sort_values()

#Scatter plot what is correlated with highly correlated with price
#Price vs Sqftliving(Sqftpricescat.png)
plt.figure(figsize=(12,6))
sns.scatterplot(x='price',y='sqft_living',hue='grade',data=df,palette="deep")

#Price vs grade(gradepricescat.png)
plt.figure(figsize=(12,6))
sns.scatterplot(x='price',y='grade',data=df)

#check grade distibution (gradedis.png)
plt.figure(figsize=(12,6))
sns.countplot(df['grade'])

#Bedroom vs Price(bedroompricebox.png)
plt.figure(figsize=(12,6))
sns.boxplot(x='bedrooms',y='price',data=df)

#distibution of prices related to longitude(pricelong.png)
plt.figure(figsize=(12,6))
sns.scatterplot(x='price',y='long',data=df)

#distibution of prices related to longitude(pricelat.png)
plt.figure(figsize=(12,6))
sns.scatterplot(x='price',y='lat',data=df)

# # Map of King County
# 
# ![title](king-wa-county-map.jpg)

#Scatter plot lan long and see where it is expensive in the map(longlatscat.png)
plt.figure(figsize=(12,6))
sns.scatterplot(x='long',y='lat',hue='price',data=df)

#Explore the data to clean price outliers

df.sort_values('price',ascending=False).head(20)


#delete top 1% of houses (Most expensive outliers)
len(df)*0.01

df_non_top1=df.sort_values('price',ascending=False).iloc[216:]

#cleaned lang long scatter(cleanedlatlong.png)
plt.figure(figsize=(12,8))
sns.scatterplot(x='long',y='lat',hue='price',data=df_non_top1,palette='RdYlGn')

#using the map we can see that the outlier are on the waterfront
#we can plot to see that(boxwater.png)
plt.figure(figsize=(12,6))
sns.boxplot(x='waterfront',y='price',data=df_non_top1)

#drop useless columns(clean and reorganise data) or change columns

df=df.drop('id',axis=1)

df['date']=pd.to_datetime(df['date'])

df['year']=df['date'].apply(lambda date:date.year)
df['month']=df['date'].apply(lambda date:date.month)

df

#EDA on new extracted data
#plot month per price(monthpricebox.png)
plt.figure(figsize=(12,6))
sns.boxplot(x='month',y='price',data=df)

#hard to tell with this plot(monthplot.png)
plt.figure(figsize=(12,6))
df.groupby('month').mean()['price'].plot()

#Year plotting against price(yearplot.png)
plt.figure(figsize=(12,6))
df.groupby('year').mean()['price'].plot()

df=df.drop('date',axis=1)

df=df.drop('zipcode',axis=1)

df

#create the train and test data
X=df.drop('price',axis=1).values
y=df['price'].values

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=101)

from sklearn.preprocessing import MinMaxScaler
scaler=MinMaxScaler()
X_train=scaler.fit_transform(X_train)
X_test=scaler.transform(X_test)

#Create the model(layer dense mean that all the neurons betweeen layer are all connected to each other)

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

model=Sequential()

model.add(Dense(19,activation='relu'))
model.add(Dense(19,activation='relu'))
model.add(Dense(19,activation='relu'))
model.add(Dense(19,activation='relu'))

model.add(Dense(1))

model.compile(optimizer='adam',loss='mse')

#fit the model
model.fit
model.fit(x=X_train,y=y_train,verbose=3,validation_data=(X_test,y_test),batch_size=128,epochs=400)

#model evaluation(lossevaluation.png)
loss=pd.DataFrame(model.history.history)
loss.plot()

#prediction of the model
from sklearn.metrics import mean_squared_error,mean_absolute_error,explained_variance_score

pred=model.predict(X_test)


#error and evaluation
mean_squared_error(y_test,pred)

np.sqrt(mean_squared_error(y_test,pred))

#avg error
mean_absolute_error(y_test,pred)

#explain the variance score from 0 to 1(best)
explained_variance_score(y_test,pred)

#plot the prediction vs the result(predvsresultscat.png)
plt.figure(figsize=(12,8))
plt.scatter(y_test,pred)
plt.plot(y_test,y_test,'r')

#predict a new value
new_house=df.drop('price',axis=1).iloc[0].values.reshape(-1,19)
new_house

new_house=scaler.transform(new_house)

model.predict(new_house)

df.head(1)

#Create a new model without the 1%
df_non_top1.head()
#clean , reshape and change the data
df_non_top1['date']=pd.to_datetime(df_non_top1['date'])
df_non_top1['year']=df_non_top1['date'].apply(lambda date:date.year)
df_non_top1['month']=df_non_top1['date'].apply(lambda date:date.month)

df_non_top1=df_non_top1.drop('id',axis=1)
df_non_top1=df_non_top1.drop('zipcode',axis=1)
df_non_top1=df_non_top1.drop('date',axis=1)

df_non_top1


#create the train and test data
X1=df_non_top1.drop('price',axis=1).values
y1=df_non_top1['price'].values


#Create and scale the train and test data
from sklearn.model_selection import train_test_split
X_train1, X_test1, y_train1, y_test1 = train_test_split(X1, y1, test_size=0.30, random_state=101)

from sklearn.preprocessing import MinMaxScaler
scaler1=MinMaxScaler()
X_train1=scaler1.fit_transform(X_train1)
X_test1=scaler1.transform(X_test1)

#Create the model
model_new=Sequential()

model_new.add(Dense(19,activation='relu'))
model_new.add(Dense(19,activation='relu'))
model_new.add(Dense(19,activation='relu'))
model_new.add(Dense(19,activation='relu'))

model_new.add(Dense(1))


model_new.compile(optimizer='adam',loss='mse')

#fit the model
model_new.fit
model_new.fit(x=X_train1,y=y_train1,validation_data=(X_test1,y_test1),batch_size=128,epochs=400)

#model evaluation(lossevaluation_new.png)
loss_new=pd.DataFrame(model_new.history.history)
loss_new.plot()

pred_new=model_new.predict(X_test1)

#error and evaluation
mean_squared_error(y_test1,pred_new)

np.sqrt(mean_squared_error(y_test1,pred_new))

#avg error
mean_absolute_error(y_test1,pred_new)

#explain the variance score from 0 to 1(best)
explained_variance_score(y_test1,pred_new)

#plot the prediction vs the result(predvsresultscat_new.png)
plt.figure(figsize=(12,8))
plt.scatter(y_test1,pred_new)
plt.plot(y_test1,y_test1,'r')


#Model comparison 100% data vs 99% data
#Mean Squared Error = 26164322071.025368 vs 21366683303.731194
#Root mean squared = 161753.89352663312 vs 
#Mean Absolute error = 101335.41279779129 vs 97410.89180144193
#Explained variance =0.803964677725989 vs 0.7418654828301325

#Conclusion and possible future development
#The new model (99%) is able to reduce the error compare to the first model(100%) trading off a lower explained variance 
#and a lower range of data trained on and suitable for the model.

#A possible further implementation could be to delete even mre outlier and train a new model on just the batch of data with
#more similar attributes and too see if the model improve it's efficency


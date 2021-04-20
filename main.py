import pandas as pd
from sklearn.metrics import mean_squared_error

import LinearRegressionGradientDescent as lrgd
import graphicView as gW
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm
from sklearn.linear_model import LinearRegression

# loading data
pd.set_option('display.max_columns', 12)
pd.set_option('display.width', None)
data = pd.read_csv("house_prices_train.csv")

# printing first and last five elements
print("First five elements: ")
print(data.head())
print("Last five elements: ")
print(data.tail())

# statistics information about table
data.info()
print(data.describe())

# feature engineering
year = np.array(data.loc[:, ['Year_built']])
area = np.array(data.loc[:, ['Area']])
bath_no = np.array(data.loc[:, ['Bath_no']])
bedroom = np.array(data.loc[:, ['Bedroom_no']])
price = data['Price']
price = price / 1000

data_train = pd.DataFrame(data=year, columns=['Year'])
data_train.insert(value=area, column='Area', loc=1)
data_train.insert(value=bath_no, column='Bath_no', loc=2)
data_train.insert(value=bedroom, column='Bedroom_no', loc=3)

data_train1 = data_train.copy(deep=True)

# graphic view of dependency between input and output values
gW.draw_dependency(year, price, 'year')
gW.draw_dependency(area, price, 'area')
gW.draw_dependency(bath_no, price, 'bath_no')
gW.draw_dependency(bedroom, price, 'bedroom_no')

lr_model = LinearRegression()
lr_model.fit(data_train, price)
predict = lr_model.predict(data_train)
data_train.insert(loc=4, column='Price', value=predict)

print("After prediction")
print("First five elements")
print(data_train.head())
print("Last five elements")
print(data_train.tail())

# the coefficients
print("Print the coefficients")
print(lr_model.coef_)

# the mean squared error
print('Mean squared error: %.2f' % mean_squared_error(price, predict))

# the coefficients of determination
# print('Coefficient of predictions: %.2f'
#      % lr_model.score(data_train, data))

# gradient descent implemented

lr = lrgd.LinearRegressionGradientDescent()
lr.fit(data_train1, price)
lr.perform_gradient_descent(0.00008, 50)
predict = lr.predict(data_train1)
data_train1.insert(loc=4, value=predict, column='Price')

print("After prediction of implemented algorithm")
print("First five elements")
print(data_train1.head())
print("Last five elements")
print(data_train1.tail())

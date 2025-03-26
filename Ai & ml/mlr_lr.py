# -*- coding: utf-8 -*-
"""
Created on Mon Mar 17 20:21:31 2025

@author: ADITYA SHOME"""
import numpy as np
import sklearn as sk
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge,LinearRegression
import matplotlib.pyplot as plt
import seaborn as sns

path="C:/Users/ADITYA SHOME/OneDrive/Desktop/"
data = pd.read_csv(path+"50_Startups.csv")

data.drop(columns=['State'], inplace=True)

# Separate features and target variable
x= data[['R&D Spend']]
y = data[['Profit']]



X_train,X_test,Y_train,Y_test=train_test_split(x, y,test_size=0.3,random_state=42)
reg=LinearRegression()
reg=reg.fit(X_train, Y_train)
y_pred=reg.predict(X_test)
mse=np.sqrt(mean_squared_error(Y_test, y_pred))
print(mse)

plt.figure(figsize=(8, 5))
sns.regplot(x=X_train, y=Y_train, scatter_kws={'alpha': 0.5}, line_kws={"color": "red"})
plt.xlabel("R&D Spend")
plt.ylabel("Profit")
plt.title("R&D Spend vs Profit with Regression Line")
plt.show()
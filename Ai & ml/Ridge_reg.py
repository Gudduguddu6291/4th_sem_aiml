# -*- coding: utf-8 -*-
import numpy as np
import sklearn as sk
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import pandas as pd
from sklearn.linear_model import Ridge,LinearRegression
path="C:/Users/ADITYA SHOME/OneDrive/Desktop/"
data = pd.read_csv(path+"BostonHousing.csv")
z=pd.DataFrame(data.corr().round(2))
x=data["rm"]
y=data["medv"]
x=pd.DataFrame(x)
y=pd.DataFrame(y)
x=np.reshape(x,(len(x),1))
X_train,X_test,Y_train,Y_test = train_test_split(x, y,test_size=0.1)
Y_test = np.reshape(Y_test, (-1,1))
reg =LinearRegression()
reg = reg.fit(X_train, Y_train)
Y_pred = reg.predict(X_test)
mean_sq_er=np.sqrt(mean_squared_error(Y_test,Y_pred))
r2_sqare = reg.score(Y_test,Y_pred)
Ridge_reg_class = Ridge()
Ridge_reg_class.fit(X_train,Y_train)
Y_pred_ridge = Ridge_reg_class.predict(X_test)
mean_sq_er_ridge = np.sqrt(mean_squared_error(Y_test, Y_pred_ridge))
print("Linear Regression"+str(mean_sq_er))
print("Ridge Regression"+str(mean_sq_er_ridge))
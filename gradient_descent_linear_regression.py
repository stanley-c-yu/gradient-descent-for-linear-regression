# -*- coding: utf-8 -*-
"""
Created on Thu Oct 31 13:08:51 2019

@author: stany
"""

import pandas as pd 
import numpy as np
import statsmodels.api as sm 


from sklearn import preprocessing 


train_data = pd.read_csv("train.csv") 
test_data = pd.read_csv("test.csv") 


#------------------------------------------------------------------------------#
##Create Training Dataframe##
df_train_predictors = pd.DataFrame({"bedrooms": train_data.bedrooms,
                                   "bathrooms": train_data.bathrooms,
                                   "sqft_living": train_data.sqft_living,
                                    "sqft_lot": train_data.sqft_lot,
                                   "floors": train_data.floors,
                                   "waterfront": train_data.waterfront,
                                   "view": train_data.view, 
                                   "condition": train_data.condition, 
                                   "grade": train_data.grade, 
                                   "sqft_above": train_data.sqft_above,
                                   "sqft_basement": train_data.sqft_basement, 
                                   "yr_built": train_data.yr_built, 
                                   "yr_renovated": train_data.yr_renovated,
                                   "lat": train_data.lat, 
                                   "long": train_data.long, 
                                   "sqft_living15": train_data.sqft_living15,
                                   "sqft_lot15": train_data.sqft_lot15})
#print(df_train_predictors[:10])

df_train_response = pd.DataFrame({"price": train_data.price})
#print(df_train_response[:10])


#------------------------------------------------------------------------------#
#------------------------------------------------------------------------------#
##Create Testing Dataframes## 
df_test_predictors = pd.DataFrame({"bedrooms": test_data.bedrooms,
                                   "bathrooms": test_data.bathrooms,
                                   "sqft_living": test_data.sqft_living,
                                    "sqft_lot": test_data.sqft_lot,
                                   "floors": test_data.floors,
                                   "waterfront": test_data.waterfront,
                                   "view": test_data.view, 
                                   "condition": test_data.condition, 
                                   "grade": test_data.grade, 
                                   "sqft_above": test_data.sqft_above,
                                   "sqft_basement": test_data.sqft_basement, 
                                   "yr_built": test_data.yr_built, 
                                   "yr_renovated": test_data.yr_renovated,
                                   "lat": test_data.lat, 
                                   "long": test_data.long, 
                                   "sqft_living15": test_data.sqft_living15,
                                   "sqft_lot15": test_data.sqft_lot15})

df_test_response = pd.DataFrame({"price": test_data.price})


#columns = ['id', 'date', 'zipcode']
#df.drop(columns, inplace=True, axis=1)
#------------------------------------------------------------------------------#

#Assign response data and predictor data to variables 
X = df_train_predictors 
y = df_train_response 

#preprocessing.StandardScaler().fit() computes the mean and standard deviation for future scaling
#Test data should be scaled using training data's mean.

std_scaleX = preprocessing.StandardScaler().fit(X) 
std_scaley = preprocessing.StandardScaler().fit(y)

X_train_scaled = std_scaleX.transform(X)
y_train_scaled = std_scaley.transform(y)

#------------------------------------------------------------------------------# 
class linear_regression:
    '''
    Employs gradient descent for linear regression to return predictions.  
    '''
    
    def __init__(self,X,y,learning_rate,max_iter,training_set):
        self.X = X 
        self.y = y 
        self.learning_rate = learning_rate
        self.max_iter = max_iter 
        self.training_set = training_set
    
    def loss_function(self,X,y,theta):  
        N = len(y)
        MSE = (1/N) * np.sum(np.square(np.matmul(X, theta) - y))
        return(MSE)
        
    def gradient_descent(self):
        X1 = np.matrix(sm.add_constant(self.X))
        m,n = np.shape(X1)
        N = len(self.y)
        theta = np.ones(n)
        theta = np.reshape(theta, (n,1))
        for i in range(0, self.max_iter):
            loss = self.loss_function(X1,self.y,theta)
            predictions = np.matmul(X1,theta)
            step=((self.learning_rate*2)/N)*(X1.transpose().dot(predictions-self.y))
            theta=theta-step
        if self.training_set == True: 
            print("\nTraining Set: True")
        else: 
            print("\nTraining Set: False")
        print("Theta: \n", theta, " \nMSE: %f" % (loss), 
              "\nAlpha = %s" % self.learning_rate, "\nIterations: %s" % self.max_iter)
        return(theta)
    
    def predictor(self):
        X1 = np.matrix(sm.add_constant(self.X)) 
        theta = self.gradient_descent()
        predictions = np.matmul(X1,theta)
        return(predictions)
    
linreg = linear_regression(X_train_scaled,y_train_scaled,0.01,1000,training_set=True)    
predictions = linreg.predictor()
print(predictions)
print(y_train_scaled)
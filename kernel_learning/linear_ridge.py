#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: almert
"""

import numpy as np
from scipy.linalg import solve
from sklearn.base import BaseEstimator
from sklearn.metrics import accuracy_score

class RidgeClassifier(BaseEstimator):
    '''
    RidgeClassifier produces a linear predictor for multiclass classification by performing Ridgre Regression on
    an indicator matrix.
    
    Arguments:
    rho: (float) The value of the ridge regularizer.
    '''
    
    def __init__(self,rho=None):
        self.rho = rho;
        
    def fit(self,X,y):
        '''
        Arguments:
        X: (array) The training data.
        y: (array) The training labels.
        '''
        
        self.classes_ = np.unique(y);
        mu = X.mean(axis=0);
        Xbar = X-mu;
        Sbar = np.dot(Xbar.T,Xbar);
        np.fill_diagonal(Sbar, Sbar.diagonal()+self.rho);
        
        Nc = np.zeros((len(self.classes_)));
        Sbh = np.zeros([Sbar.shape[0],len(self.classes_)]);
        for i in range(len(self.classes_)):
            ssIdx = np.argwhere(y==self.classes_[i]).squeeze();
            Nc[i] = len(ssIdx);
            Sbh[:,i] += np.sum(Xbar[ssIdx,:],axis=0);
        
        self.coef_ = solve(Sbar,Sbh,sym_pos=True,overwrite_a=True,overwrite_b=True).T;
        self.intercept_ = Nc/len(X)-np.dot(self.coef_,mu);
        return self;

    def predict(self,X):
        '''
        Arguments:
        X: (array) The data to perform predictions on.
        
        Returns:
        An array storing the predicted class labels.
        '''
        
        scores = np.dot(X,self.coef_.T)+self.intercept_;
        predictions = self.classes_[np.argmax(scores,axis=1)];
        return predictions;
        
    def score(self,X,y):
        '''
        Arguments:
        X: (array) The test data to perform predictions on.
        y: (array) The true class labels.
        
        Returns:
        The ratio of samples whose class was predicted correctly.
        '''
        
        return accuracy_score(y,self.predict(X));

class RidgeRegressor(BaseEstimator):
    '''
    RidgeRegressor performs linear regression with an l_2 penalty.
    
    Arguments:
    rho: (float) The value of the ridge regularizer.
    '''
    
    def __init__(self,rho=None):
        self.rho = rho;
        
    def fit(self,X,y):
        '''
        Arguments:
        X: (array) The training data.
        y: (array) The training labels.
        '''
        
        mu = X.mean(axis=0);
        Xbar = X-mu;
        Sbar = np.dot(Xbar.T,Xbar);
        np.fill_diagonal(Sbar, Sbar.diagonal()+self.rho);
                
        self.coef_ = solve(Sbar,np.dot(Xbar.T,y),sym_pos=True,overwrite_a=True,overwrite_b=True).T;
        self.intercept_ = np.mean(y,axis=0)-np.dot(self.coef_,mu);
        return self;

    def predict(self,X):
        '''
        Arguments:
        X: (array) The test data to perform predictions on.
        
        Returns:
        An array storing the predictions.
        '''
        
        predictions = np.dot(X,self.coef_.T)+self.intercept_;
        return predictions;
        
    def score(self,X,y):
        '''
        Arguments:
        X: (array) The test data to perform predictions on.
        y: (array) The correct predictions.
        
        Returns:
        The R-Squared value, i.e., the explained variance.
        '''
        
        residual_sum = np.sum((y-self.predict(X))**2);
        total_sum = np.sum((y-np.mean(y,axis=0))**2);
        return 1. - residual_sum/total_sum;
        
    def rmse_score(self,X,y):
        '''
        Arguments:
        X: (array) The test data to perform predictions on.
        y: (array) The correct predictions.
        
        Returns:
        The Root Mean Squared Error (RMSE) value.
        '''
        
        return np.sqrt(np.mean((y-self.predict(X))**2));

class Binary(BaseEstimator):
    def __init__(self,rho=None):
        self.rho = rho;
        
    def fit(self,X,y):
        self.classes_ = np.unique(y);
        if len(self.classes_) != 2:
            print('This class only supports binary classification!');
            return self;
        mu = X.mean(axis=0);
        Xbar = X-mu;
        Sbar = np.dot(Xbar.T,Xbar);
        np.fill_diagonal(Sbar, Sbar.diagonal()+self.rho);
        
        Nc = [];
        Idx = np.argwhere(y==self.classes_[0]).squeeze();
        Nc.append(len(Idx));
        Sbh = -np.sum(Xbar[Idx,:],axis=0);
        Idx = np.argwhere(y==self.classes_[1]).squeeze();
        Nc.append(len(Idx));
        Sbh += np.sum(Xbar[Idx,:],axis=0);
        
        self.coef_ = solve(Sbar,Sbh,sym_pos=True,overwrite_a=True,overwrite_b=True);
        self.intercept_ = (Nc[1]-Nc[0])/len(X)-np.dot(self.coef_,mu);
        return self;

    def predict(self,X):
        scores = np.dot(X,self.coef_)+self.intercept_;
        predictions = self.classes_[(scores>0).astype(int)];
        return predictions;
        
    def score(self,X,y):
        return accuracy_score(y,self.predict(X));

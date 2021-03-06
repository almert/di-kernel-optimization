#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: almert
"""

import numpy as np
from scipy.linalg import solve, eigh
from sklearn.metrics import pairwise
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelBinarizer
from sklearn.base import BaseEstimator

class KernelRidgeClassifier(BaseEstimator):
    '''
    KernelRidgeClassifier produces a kernel predictor for multiclass classification by performing Kernel Ridgre Regression 
    on an indicator matrix.
    
    Arguments:
    rho: (float) The value of the ridge regularizer.
    kernel: (string) The type of kernel to be used. Currently 'linear', 'poly', 'rbf', 'laplacian', 'chi2', 
    'additive_chi2' and 'sigmoid' are supported.
    gamma: (float) The scale parameter of the kernel. Equals 1/sqrt(2 bandwidth^2) for kernels with a bandwidth parameter.
    degree: (float) The degree for polynomial kernels.
    coef0: (float) An additive parameter for 'poly' and 'sigmoid' kernels.
    '''
    
    def __init__(self,rho=0.0001,kernel='rbf',gamma=1.,degree=3,coef0=1.):
        self.rho = rho;
        self.kernel = kernel;
        self.gamma = gamma;
        self.degree = degree;
        self.coef0 = coef0;
        
    def fit(self,X,y,solver='chol'):
        '''
        Arguments:
        X: (array) The training data.
        y: (array) The training labels.
        '''
        
        self.X = X;
        self.N = len(X);
        
        self.lb = LabelBinarizer(neg_label=0,pos_label=1);
        self.lb.fit(y);
        if self.lb.y_type_ == 'binary':
            self.lb.neg_label = -1; 

        yb = self.lb.transform(y).astype(float);
        mu_y = np.mean(yb,axis=0);
        yb -= mu_y;
        
        K = get_kernel_matrix(X,None,self.kernel,self.gamma,self.degree,self.coef0);
        mu_K = np.mean(K,axis=0);
        K -= mu_K;
        K -= np.mean(K,axis=1)[:,None];
        np.fill_diagonal(K, K.diagonal()+self.rho);

        if solver == 'chol':
            self.dual_coef_ = solve(K,yb,sym_pos=True,overwrite_a=True,overwrite_b=True);

        self.dual_coef_ -= np.mean(self.dual_coef_,axis=0);
        self.dual_coef_ = self.dual_coef_.T;
        self.intercept_ = mu_y-np.dot(self.dual_coef_.T,mu_K)
        return self;
        
    def predict(self,X):
        '''
        Arguments:
        X: (array) The data to perform predictions on.
        
        Returns:
        An array storing the predicted class labels.
        '''
        
        K = get_kernel_matrix(X,self.X,self.kernel,self.gamma,self.degree,self.coef0)
        scores = np.dot(K,self.dual_coef_.T)+self.intercept_;
        if self.lb.y_type_ == 'binary':
            return self.lb.classes_[(scores>=0).astype(int)];
        else:
            return self.lb.classes_[np.argmax(scores,axis=1)];
    
    def score(self,X,y):
        '''
        Arguments:
        X: (array) The test data to perform predictions on.
        y: (array) The true class labels.
        
        Returns:
        The ratio of samples whose class was predicted correctly.
        '''
        
        return accuracy_score(y,self.predict(X));

class KernelRidgeRegressor(BaseEstimator):
    '''
    KernelRidgeRegressor performs kernel regression with an l_2 penalty.
    
    Arguments:
    rho: (float) The value of the ridge regularizer.
    kernel: (string) The type of kernel to be used. Currently 'linear', 'poly', 'rbf', 'laplacian', 'chi2', 
    'additive_chi2' and 'sigmoid' are supported.
    gamma: (float) The scale parameter of the kernel. Equals 1/sqrt(2 bandwidth^2) for kernels with a bandwidth parameter.
    degree: (float) The degree for polynomial kernels.
    coef0: (float) An additive parameter for 'poly' and 'sigmoid' kernels.
    '''
    
    def __init__(self,rho=0.0001,kernel='rbf',gamma=1.,degree=3,coef0=1.):
        self.rho = rho;
        self.kernel = kernel;
        self.gamma = gamma;
        self.degree = degree;
        self.coef0 = coef0;
        
    def fit(self,X,y,solver='chol'):
        '''
        Arguments:
        X: (array) The training data.
        y: (array) The training labels.
        '''
        
        self.X = X;
        self.N = len(X);
        
        mu_y = np.mean(y,axis=0);
        y -= mu_y;
        
        K = get_kernel_matrix(X,None,self.kernel,self.gamma,self.degree,self.coef0);
        mu_K = np.mean(K,axis=0);
        K -= mu_K;
        K -= np.mean(K,axis=1)[:,None];
        np.fill_diagonal(K, K.diagonal()+self.rho);

        if solver == 'chol':
            self.dual_coef_ = solve(K,y,sym_pos=True,overwrite_a=True,overwrite_b=True);

        self.dual_coef_ -= np.mean(self.dual_coef_,axis=0);
        self.dual_coef_ = self.dual_coef_.T;
        self.intercept_ = mu_y-np.dot(self.dual_coef_.T,mu_K)
        return self;
        
    def predict(self,X):
        '''
        Arguments:
        X: (array) The test data to perform predictions on.
        
        Returns:
        An array storing the predictions.
        '''
        
        K = get_kernel_matrix(X,self.X,self.kernel,self.gamma,self.degree,self.coef0)
        predictions = np.dot(K,self.dual_coef_.T)+self.intercept_;
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
    
def NystromRidgeRegressor(X,y,X_rep,rho=1e-4,kernel = 'rbf',gamma = 0.01,degree = 3,
                          coef0 = 1,normalize=False,rcond = None):
    '''
    NystromRidgeRegressor uses a representative set of samples to compute the corresponding dual coefficients and
    the intercept term directly, when a Ridge Regressor is trained on the Nystrom kernel features.
    
    Arguments:
    X: (array) The training data.
    y: (array) The training labels.
    X_rep: (array) The representative data points.
    rho: (float) The value of the ridge regularizer.
    kernel: (string) The type of kernel to be used. Currently 'linear', 'poly', 'rbf', 'laplacian', 'chi2', 
    'additive_chi2' and 'sigmoid' are supported.
    gamma: (float) The scale parameter of the kernel. Equals 1/sqrt(2 bandwidth^2) for kernels with a bandwidth parameter.
    degree: (float) The degree for polynomial kernels.
    coef0: (float) An additive parameter for 'poly' and 'sigmoid' kernels.
    normalize: (bool) Whether the columns of y should be scaled to be unit norm. This makes the regression produce an optimal 
    Kernel Discriminant Analysis Solution.
    rcond: The condition number to be imposed on the kernel matrix, used to determine rank.
    '''
    
    if normalize:
        yn = y/np.norm(y,axis=0);
    else:
        yn = y;
        
    K = get_kernel_matrix(X,X_rep,kernel,gamma,degree,coef0);
    mu_K = np.mean(K,axis=0);
    K -= mu_K;
    
    Kw = np.dot(K.T,K)+rho*get_kernel_matrix(X_rep,None,kernel,gamma,degree,coef0);
    Ky = np.dot(K.T,yn);
        
    if rcond is None:
        rcond = len(X_rep)*np.finfo(K.dtype).eps;

    Keig, Kvec = eigh(Kw,overwrite_a=True);
    mask = Keig >= Keig[-1]*rcond;
    Keig = Keig[mask]; Kvec = Kvec[:,mask];
    A = np.dot(Kvec*(1./Keig),(Kvec.T).dot(Ky));
    b = np.mean(y,axis=0) - np.dot(A.T,mu_K);
    return (A,b)

def get_kernel_matrix(X1, X2=None, kernel='rbf',gamma = 1, degree = 3, coef0=1):
    #Obtain N1xN2 kernel matrix from N1xM and N2xM data matrices
    if kernel == 'rbf':
        K = pairwise.rbf_kernel(X1,X2,gamma = gamma);
    elif kernel == 'poly':
        K = pairwise.polynomial_kernel(X1,X2,degree = degree, gamma = gamma,
                                       coef0 = coef0);
    elif kernel == 'linear':
        K = pairwise.linear_kernel(X1,X2);
    elif kernel == 'laplacian':
        K = pairwise.laplacian_kernel(X1,X2,gamma = gamma);
    elif kernel == 'chi2':
        K = pairwise.chi2_kernel(X1,X2,gamma = gamma);
    elif kernel == 'additive_chi2':
        K = pairwise.additive_chi2_kernel(X1,X2);
    elif kernel == 'sigmoid':
        K = pairwise.sigmoid_kernel(X1,X2,gamma = gamma,coef0 = coef0);
    else:
        print('[Error] Unknown kernel');
        K = None;
    return K;

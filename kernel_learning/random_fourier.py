#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: almert
"""

import numpy as np
from scipy.stats import cauchy
from sklearn.base import BaseEstimator

class RandomFourier(BaseEstimator):
    '''
    RandomFourier approximates a kernel mapping using the methodology introduced in Rahimi and Recht [1].
    [1] https://people.eecs.berkeley.edu/~brecht/papers/07.rah.rec.nips.pdf
    
    Arguments: 
    n: (int) The dimensionality of the kernel feature map.
    kernel: (string) The type of kernel to be used. Only 'rbf' and 'laplacian' are supported.
    gamma: (float) The scale parameter of the kernel. Equals 1/sqrt(2 bandwidth^2).
    '''
    
    def __init__(self,n_dims=1000,kernel='rbf',gamma=0.01):
        self.J = n_dims;
        self.kernel = kernel;
        self.gamma = gamma;
        
    def fit(self,X):
        '''
        Arguments: 
        
        X: The training data, only the number of columns is used to produce the kernel mapping.
        '''
        
        M = np.shape(X)[1];
        if self.kernel == 'rbf':
            self.W = np.sqrt(2*self.gamma)*np.random.normal(size=(M,self.J));
        elif self.kernel == 'laplacian':
            self.W = cauchy.rvs(scale=self.gamma,size=(M,self.J));
        else:
            print('Kernel type \''+str(self.kernel)+'\' is not supported!');
            return self;
        self.b = np.random.uniform(low=.0,high=2*np.pi,size=(self.J));
        return self;
        
    def transform(self,X):
        '''
        Arguments:
        X: (array) The data to apply the kernel mapping to.
        
        Returns:
        An array representing the kernel mapping applied to X.
        '''
        
        return np.sqrt(2./self.J)*np.cos(np.dot(X,self.W)+self.b);
        

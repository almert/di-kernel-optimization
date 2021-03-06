#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: almert
"""

import sys
sys.path.append('..')
from numbers import Number
import tensorflow as tf
import numpy as np
import tf_kernel
from scipy.linalg import eigh
from scipy.stats import cauchy
from sklearn.metrics import pairwise


class KernelLayer():
    '''
    KernelLayer produces a tensorflow compatible kernel mapping by performing a Nystrom approximation.
    
    Arguments:
    n: (int) The number of samples to be selected in random to produce the Nystrom approximation.
    k: (int) The output dimensionality of the kernel feature map.
    kernel: (string) The type of kernel to be used. Currently, only 'rbf' and 'laplacian' are supported.
    gamma: (float) The scale parameter of the kernel. Equals 1/sqrt(2 bandwidth^2).
    '''
    
    def __init__(self,n=1024,k=None,kernel='rbf',gamma=0.01):
        
        assert (isinstance(n,int) or np.issubdtype(n, np.integer)) and n > 0
        assert k is None or ((isinstance(k,int) or np.issubdtype(k, np.integer)) and k > 0)
        assert kernel in ['rbf','laplacian']
        assert isinstance(gamma,Number) and gamma > 0
        self.n = n
        self.kernel = kernel
        self.gamma = gamma
        
        if k is None or k >= n:
            k = n;
        self.k = k;
            
        if self.kernel == 'rbf':
            self.kernel_func = tf_kernel.GaussianKernel(1./np.sqrt(2*self.gamma))
        elif self.kernel == 'laplacian':
            self.kernel_func = tf_kernel.LaplacianKernel(1./self.gamma)
        
    def initialize(self,X,dtype=tf.float64,x_trainable=True,w_trainable=True):
        '''
        Arguments:   
        X: (array) The training data from which n landmark samples are selected to create a kernel mapping.
        dtype: (object) The tensorflow data type.
        x_trainable: (bool) Whether the landmark samples trainable or not.
        w_trainable: (bool) Whether the projection matrix is trainable or not.
        '''
        
        assert isinstance(X, np.ndarray) and X.ndim == 2
        assert isinstance(x_trainable,bool) and isinstance(w_trainable,bool)

        X_set = X[np.random.permutation(len(X))[:self.n]]
        mapper = LocalNystrom(kernel=self.kernel,gamma=self.gamma)
        mapper.fit(X_set,k=self.k)
        
        self.X_rep = tf.Variable(X_set,dtype=dtype,trainable=x_trainable)
        self.W_rep = tf.Variable(mapper.A,dtype=dtype,trainable=w_trainable)
        
    def layer_func(self,X):
        '''
        Arguments:
        X: (2d tensor) The data to apply the kernel mapping to.
        
        Returns:
        A tensor representing the kernel mapping applied to X.
        '''
        
        return tf.matmul(self.kernel_func(X, self.X_rep),self.W_rep)


class SubspaceKDI():
    '''
    SubspaceKDI produces a tensorflow compatible kernel mapping and produces the Kernel Discriminant Information (KDI)
    metric for evaluating/optimizing the kernel mapping.
    
    Arguments:
    n: (int) The number of samples to be selected in random to produce the Nystrom approximation.
    k: (int) The output dimensionality of the kernel feature map.
    kernel: (string) The type of kernel to be used. Currently, only 'rbf' and 'laplacian' are supported.
    gamma: (float) The scale parameter of the kernel. Equals 1/sqrt(2 bandwidth^2).
    rho: (float) The ridge regularizer to be used in the KDI objective.
    '''
    
    def __init__(self,n=1024,kernel='rbf',gamma=0.01,rho=1e-4):
        assert (isinstance(n,int) or np.issubdtype(n, np.integer)) and n > 0
        assert kernel in ['rbf','laplacian']
        assert isinstance(gamma,Number) and gamma > 0
        assert isinstance(rho,Number) and rho > 0
        self.n = n
        self.kernel = kernel
        self.gamma = gamma
        self.rho = rho
            
        if self.kernel == 'rbf':
            self.kernel_func = tf_kernel.GaussianKernel(1./np.sqrt(2*self.gamma))
        elif self.kernel == 'laplacian':
            self.kernel_func = tf_kernel.LaplacianKernel(1./self.gamma)
            
    def initialize(self,X,dtype=tf.float64,x_trainable=True):
        '''
        Arguments:        
        X: (array) The training data from which n landmark samples are selected to initialize the kernel mapping.
        dtype: (object) The tensorflow data type.
        x_trainable: (bool) Whether the landmark samples trainable or not. Set to True in order to optimize the 
        kernel map via KDI.
        '''
        
        assert isinstance(X, np.ndarray) and X.ndim == 2
        assert isinstance(x_trainable,bool)

        X_set = X[np.random.permutation(len(X))[:self.n]]        
        self.X_rep = tf.Variable(X_set,dtype=dtype,trainable=x_trainable)
        self.K_rep = self.kernel_func(self.X_rep)
        
    def layer_func(self,X):
        '''
        Arguments:
        X: (2d tensor) The data to apply the kernel mapping to.
        
        Returns:
        A tensor representing the kernel mapping applied to X.
        '''
        
        return self.kernel_func(X, self.X_rep)
        
    def KDI(self,X,Y,normalize=False,epsilon=1e-10):
        '''
        Arguments:
        X: (2d tensor) The data to compute the KDI with.
        Y: (2d tensor) The training labels to compute KDI with.
        normalize: (bool) Whether to scale the columns of Y to be unit norm. This makes KDI match Fisher's Discriminant
        Analysis objective.
        epsilon: (float) A small ridge regularizer added for numerical stability.
        
        Returns:
        A tensorflow scalar representing the value of the KDI objective.
        '''
        
        if normalize:
            Yn = Y/(tf.norm(Y,axis=0)+1e-10)
        else:
            Yn = Y
        
        K = self.kernel_func(X, self.X_rep)
        Kbar = K-tf.reduce_mean(K, axis=0)
        Kw = tf.matmul(Kbar,Kbar,transpose_a=True) + self.rho*self.K_rep + \
                epsilon*tf.eye(self.n,dtype=X.dtype)
        #Kw_inv = tf.matrix_inverse(Kw)
        
        Kb_half = tf.matmul(Kbar,Yn,transpose_a=True)
        KinvKb = tf.linalg.solve(Kw,Kb_half)
        objective = tf.reduce_sum(Kb_half*KinvKb)
        return objective
        

class RandomFourierLayer():
    '''
    RandomFourierLayer produces a tensorflow compatible kernel mapping by performing a Random Fourier approximation.
    
    Arguments:
    n: (int) The dimensionality of the kernel feature map.
    kernel: (string) The type of kernel to be used. Only 'rbf' and 'laplacian' are supported.
    gamma: (float) The scale parameter of the kernel. Equals 1/sqrt(2 bandwidth^2).
    '''

    
    def __init__(self,n=1024,kernel='rbf',gamma=0.01):
        
        assert (isinstance(n,int) or np.issubdtype(n, np.integer)) and n > 0
        assert kernel in ['rbf','laplacian']
        assert isinstance(gamma,Number) and gamma > 0
        self.n = n
        self.kernel = kernel
        self.gamma = gamma
        
    def initialize(self,X,dtype=tf.float64,W_init=None,b_init=None,
                   w_trainable=True,b_trainable=True):
        '''
        Arguments: 
        X: (array) The training data, only the number of columns is used to produce the kernel mapping.
        dtype: (object) The tensorflow data type.
        W_init: (array) The initial value of the projection matrix. If None, initialization is done randomly 
        based on the chosen kernel.
        b_init: (array) The initial value of the bias vector. If None, initialization is done randomly 
        based on the chosen kernel.
        w_trainable: (bool) Whether the projection matrix is trainable or not.
        b_trainable: (bool) Whether the bias vector is trainable or not.
        '''        
        
        assert isinstance(X, np.ndarray) and X.ndim == 2
        assert isinstance(w_trainable,bool) and isinstance(b_trainable,bool)

        m = np.shape(X)[1];

        if W_init is None:
            if self.kernel == 'rbf':
                self.W = tf.Variable(np.sqrt(2*self.gamma)*np.random.normal(size=(m,self.n)),
                                     dtype=dtype,trainable=w_trainable)
            elif self.kernel == 'laplacian':
                self.W = tf.Variable(cauchy.rvs(scale=self.gamma,size=(m,self.n)),
                                     dtype=dtype,trainable=w_trainable);
        else:
            self.W = tf.Variable(W_init,dtype=dtype,trainable=w_trainable)
           
        if b_init is None:                          
            self.b = tf.Variable(np.random.uniform(low=.0,high=2*np.pi,size=(self.n)),
                                 dtype=dtype,trainable=b_trainable)
        else:
            self.b = tf.Variable(b_init,dtype=dtype,trainable=b_trainable)
               
    def layer_func(self,X):
        '''
        Arguments:
        X: (2d tensor) The data to apply the kernel mapping to.
        
        Returns:
        A tensor representing the kernel mapping applied to X.
        '''
        
        return np.sqrt(2./self.n)*tf.cos(tf.matmul(X,self.W)+self.b);


class LocalNystrom():
    def __init__(self,kernel='rbf',gamma=1.,degree=3,coef0=1.):
        self.kernel = kernel;
        self.gamma = gamma;
        self.degree = degree;
        self.coef0 = 1.;
      
    def fit(self,X_rep,rcond=None,seed=None,k=None):
        n = len(X_rep);
        K = get_kernel_matrix(X_rep,X_rep,self.kernel,self.gamma,
                                        self.degree,self.coef0)
        Keig, Kvec = eigh(K);
                 
        if (k is None) or (k>n):
            k = n;
        if rcond is None:
            rcond = n*np.finfo(K.dtype).eps;
        mask = Keig >= Keig[-1]*rcond;
        mask[:-k] = False;
        Keig = Keig[mask]; Kvec = Kvec[:,mask];
        self.A = Kvec*(1./np.sqrt(Keig));
        return self;
    
    def transform(self,X,X_rep):
        Ktest = get_kernel_matrix(X,X_rep,self.kernel,self.gamma,
                                        self.degree,self.coef0);
        return np.dot(Ktest,self.A);


def DI(X,Y,rho=1e-4,normalize=False):
    '''
    DI computes the linear Discriminant Information criterion on the data and labels. This can be applied to a non-linear 
    mapping such as a Random Fourier layer to produce a supervised training objective.
    
    Arguments:
    X: (2d tensor) The data to compute the DI with.
    Y: (2d tensor) The training labels to compute DI with.
    normalize: (bool) Whether to scale the columns of Y to be unit norm. This makes DI match Fisher's Discriminant
    Analysis objective.
    rho: The ridge regularizer used in the DI objective.
        
    Returns:
    A tensorflow scalar representing the value of the DI objective.
    '''
    
    if normalize:
        Yn = Y/(tf.norm(Y,axis=0)+1e-10)
    else:
        Yn = Y
    
    Xbar = X - tf.reduce_mean(X,axis=0)
    Sbar = tf.matmul(Xbar,Xbar,transpose_a=True)+rho*tf.eye(tf.shape(X)[1],dtype=X.dtype)
    #S_inv = tf.matrix_inverse(Sbar)
    
    Sb_half = tf.matmul(Xbar,Yn,transpose_a=True)
    objective = tf.reduce_sum(Sb_half*tf.linalg.solve(Sbar,Sb_half))
    return objective
    

def get_kernel_matrix(X1,X2=None,kernel='rbf',gamma = 1,degree = 3,coef0=1):
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
                

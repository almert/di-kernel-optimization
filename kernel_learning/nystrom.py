#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan  3 14:08:59 2018

@author: merta
"""

import numpy as np
from scipy.linalg import eigh
from sklearn.metrics import pairwise, accuracy_score
from sklearn.cluster import MiniBatchKMeans
from sklearn.base import BaseEstimator
from sklearn.utils.extmath import randomized_svd
from sklearn.svm import LinearSVC
from linear_ridge import Binary, RidgeClassifier, RidgeRegressor

class Nystrom(BaseEstimator):
    '''
    Nystrom produces a low-dimensional kernel feature mapping via the Nystrom approximation method.
    
    Arguments:
    kernel: (string) The type of kernel to be used. Currently 'linear', 'poly', 'rbf', 'laplacian', 'chi2', 
    'additive_chi2' and 'sigmoid' are supported.
    gamma: (float) The scale parameter of the kernel. Equals 1/sqrt(2 bandwidth^2) for kernels with a bandwidth parameter.
    degree: (float) The degree for polynomial kernels.
    coef0: (float) An additive parameter for 'poly' and 'sigmoid' kernels.
    n: (int) Number of random data samples used to produce a Nystrom approximation. If n=len(X), no approximation takes place
    and the full-dimensional kernel mapping is computed.
    k: (int) Number of subspace dimensions used. If k=n, the full kernel feature space is used.
    approx: (string) Whether Standard Nystrom ('Nys') or K-Means Nystrom ('KNys') approximation is used.
    kmeans_params: (dict) A dictionary containing the k-means parameters. Only used if approx='KNys'.
    kmeans_N: (int) The maximum number of samples to perform K-Means clustering with.
    rand_svd: (bool) Whether to use randomized SVD to obtain the Nystrom kernel approximation. Can result in speed up when k is
    much smaller than n.
    over_sampling: The over_sampling parameter for randomized SVD.
    power: The power parameter for randomized SVD.
    '''
    
    def __init__(self,kernel='rbf',gamma=1.,degree=3,coef0=1.,n=None,k=None,
                 approx='Nys',kmeans_params=None,kmeans_N=20000,rand_svd=False,
                 over_sampling=10,power=2):
        self.kernel = kernel;
        self.gamma = gamma;
        self.degree = degree;
        self.coef0 = coef0;
        self.n = n;
        self.k = k;
        self.approx = approx;
        self.kmeans_params = kmeans_params;
        self.kmeans_N = kmeans_N;
        self.rand_svd = rand_svd;
        self.over_sampling = over_sampling;
        self.power = power;
        
    def fit(self,X,rcond=None,seed=None):
        '''
        Arguments:
        X: (array) The training data.
        rcond: (float) The condition number to be imposed on the kernel matrix.
        seed: (float) The seed to control randomization.
        '''
        
        self.N = len(X);
        self._Xrep = None;
        if seed is not None:
            np.random.seed(seed=seed);

        if (self.n is None) or (self.n>self.N):
            self.n = self.N;
        if (self.k is None) or (self.k>=self.n):
            self.k = self.n;
            self.rand_svd = False;
            
        if self.approx == 'Nys':
            Inds = np.random.permutation(self.N)[:self.n];
            self._Xrep = X[Inds,:];
        elif self.approx == 'KNys':
            if self.kmeans_params == None:
                self.kmeans_params = {'n_clusters': self.n,'max_iter': 20,
                                 'compute_labels': False,'init_size': 3*self.n};
            KM = MiniBatchKMeans(**self.kmeans_params);
            if self.N > self.kmeans_N:
                Inds = np.random.permutation(self.N)[:self.kmeans_N];
                KM.fit(X[Inds,:]);
            else:
                KM.fit(X);
            self._Xrep = KM.cluster_centers_;
        else:
            print('Invalid approximation method.');
            return None;
        K = get_kernel_matrix(self._Xrep,None,self.kernel,self.gamma,
                                        self.degree,self.coef0);
                              
        if rcond is None:
            rcond = self.n*np.finfo(K.dtype).eps;
                              
        if self.rand_svd:
            Kvec, Keig = RSVD(K,self.k,self.over_sampling,self.power);
            mask = Keig >= Keig[0]*rcond;
        else:
            Keig, Kvec = eigh(K,overwrite_a=True);
            mask = Keig >= Keig[-1]*rcond;
            mask[:-self.k] = False;
        Keig = Keig[mask]; Kvec = Kvec[:,mask];
        self.A = Kvec*(1./np.sqrt(Keig));
    
    def transform(self,X):
        '''
        Arguments:
        X: (array) The data to apply the kernel mapping to.
        
        Returns:
        An array representing the kernel mapping applied to X.
        '''
       
        Ktest = get_kernel_matrix(X,self._Xrep,self.kernel,self.gamma,
                                        self.degree,self.coef0);
        Phi = np.dot(Ktest,self.A);
        return Phi;

class EnsembleNystrom(BaseEstimator):
    '''
    EnsembleNystrom produces a low-dimensional kernel feature mapping by combining smaller scale Nystrom mappings.
    
    Arguments:
    kernel: (string) The type of kernel to be used. Currently 'linear', 'poly', 'rbf', 'laplacian', 'chi2', 
    'additive_chi2' and 'sigmoid' are supported.
    gamma: (float) The scale parameter of the kernel. Equals 1/sqrt(2 bandwidth^2) for kernels with a bandwidth parameter.
    degree: (float) The degree for polynomial kernels.
    coef0: (float) An additive parameter for 'poly' and 'sigmoid' kernels.
    n: (int) Number of random data samples used to produce each Nystrom approximation. If n=len(X), no approximation takes place
    and the full-dimensional kernel mapping is computed.
    k: (int) Number of subspace dimensions used in individual Nystrom mappings. If k=n, the full kernel feature space is used.
    p: (int) Number of Nystrom feature maps to be combined.
    weights: (string) The weighting scheme for the Nystrom maps. Currently, only 'uniform' is supported. Alternatively a regression
    could be fitted on a hold-out sample to optimize the weights.
    rand_svd: (bool) Whether to use randomized SVD to obtain the Nystrom kernel approximation. Can result in speed up when k is
    much smaller than n.
    over_sampling: The over_sampling parameter for randomized SVD.
    power: The power parameter for randomized SVD.
    '''
    
    def __init__(self,kernel='rbf',gamma=1.,degree=3,coef0=1.,n=None,k=None,p=1,
                 weights='uniform',rand_svd=False,over_sampling=10,power=2):
        self.kernel = kernel;
        self.gamma = gamma;
        self.degree = degree;
        self.coef0 = coef0;
        self.n = n;
        self.k = k;
        self.p = p;
        self.weights = weights;
        self.rand_svd = rand_svd;
        self.over_sampling = over_sampling;
        self.power = power;
        
    def fit(self,X,rcond=None,seed=None):
        self.N = len(X);
        self._Xrep = [[] for i in range(self.p)];
        self.A = [];
        self.dims = [];
        if seed is not None:
            np.random.seed(seed=seed);
    
        if self.n is None:
            self.n = self.N//self.p;           
        if rcond == None:
            rcond = self.n*np.finfo('float64').eps;
            
        Inds = np.random.permutation(self.N)[:self.n*self.p];
        if (self.k is None) or (self.k>=self.n):
            self.k = self.n;
            self.rand_svd = False;
            
        start = 0;
        end = self.n;
        i = 0;
        while end <= self.n*self.p:
            self._Xrep[i] = X[Inds[start:end]];
            K = get_kernel_matrix(self._Xrep[i],None,self.kernel,self.gamma,
                                        self.degree,self.coef0);
            
            if self.rand_svd:
                Kvec, Keig = RSVD(K,self.k,self.over_sampling,self.power);
                mask = Keig >= Keig[0]*rcond;
            else:
                Keig, Kvec = eigh(K,overwrite_a=True);
                mask = Keig >= Keig[-1]*rcond;
                mask[:-self.k] = False;
            Keig = Keig[mask]; Kvec = Kvec[:,mask];
            self.A.append(Kvec*(1./np.sqrt(Keig)));
            self.dims.append(self.A[i].shape[1]);

            i += 1;
            start += self.n;
            end += self.n;
                       
        if self.weights == 'uniform':
            self.mu = [np.sqrt(1./self.p)]*self.p;

        for i in range(self.p):
            self.A[i] = self.A[i]*self.mu[i];
        
        return self;
        
    def transform(self,X):
        Phi = np.zeros((len(X),sum(self.dims)));
        start = 0;
        for i in range(self.p):
            K = get_kernel_matrix(X,self._Xrep[i],self.kernel,self.gamma,
                                        self.degree,self.coef0);
            Phi[:,start:start+self.dims[i]] = np.dot(K,self.A[i]);
            start += self.dims[i];
        return Phi;
        
        
class NystromClassifier(BaseEstimator):
    def __init__(self,kernel='rbf',gamma=1.,degree=3,coef0=1.,n=1000,k=None,
                 rand_svd = False,clf='svc',C=1.,rho=1e-4):
        self.kernel = kernel;
        self.gamma = gamma;
        self.degree= degree;
        self.coef0 = coef0;
        self.n = n;
        self.k = k;
        self.rand_svd = rand_svd;
        self.clf = clf;
        self.C = C;
        self.rho = rho;
        
    def fit(self,X,y):
        mapper = Nystrom(kernel=self.kernel,gamma=self.gamma,degree=self.degree,
                         coef0=self.coef0,n=self.n,k=self.k,rand_svd=self.rand_svd);
        mapper.fit(X);
        self._Xrep = mapper._Xrep;

        if self.clf == 'svc':
            clf = LinearSVC(dual=False,C=self.C);
        elif self.clf == 'ridge':
            if len(np.unique(y)>2):
                clf = RidgeClassifier(rho=self.rho);
            else:
                clf = Binary(rho=self.rho);
        else:
            clf = self.clf;
            
        clf.fit(mapper.transform(X),y);
        self.dual_coef_ = np.dot(mapper.A,clf.coef_.T).T;
        self.intercept_ = clf.intercept_;
        self.classes_ = clf.classes_;
        return
        
    def predict(self,X):
        Ktest = get_kernel_matrix(X,self._Xrep,self.kernel,self.gamma,
                                  self.degree,self.coef0);
        scores = np.dot(Ktest,self.dual_coef_.T)+self.intercept_;
        preds = self.classes_[np.argmax(scores,axis=1)];
        return preds;

    def score(self,X,y):
        return accuracy_score(y,self.predict(X));
        
class NystromRegressor(BaseEstimator):
    def __init__(self,kernel='rbf',gamma=1.,degree=3,coef0=1.,n=1000,k=None,
                 rand_svd = False,rho=1e-4):
        self.kernel = kernel;
        self.gamma = gamma;
        self.degree= degree;
        self.coef0 = coef0;
        self.n = n;
        self.k = k;
        self.rand_svd = rand_svd;
        self.rho = rho;
        
    def fit(self,X,y):
        mapper = Nystrom(kernel=self.kernel,gamma=self.gamma,degree=self.degree,
                         coef0=self.coef0,n=self.n,k=self.k,rand_svd=self.rand_svd);
        mapper.fit(X);
        self._Xrep = mapper._Xrep;

        regressor = RidgeRegressor(rho=self.rho);            
        regressor.fit(mapper.transform(X),y);
        self.dual_coef_ = np.dot(mapper.A,regressor.coef_.T).T;
        self.intercept_ = regressor.intercept_;
        return
        
    def predict(self,X):
        Ktest = get_kernel_matrix(X,self._Xrep,self.kernel,self.gamma,
                                  self.degree,self.coef0);
        predictions = np.dot(Ktest,self.dual_coef_.T)+self.intercept_;
        return predictions;

    def score(self,X,y):
        residual_sum = np.sum((y-self.predict(X))**2);
        total_sum = np.sum((y-np.mean(y,axis=0))**2);
        return 1. - residual_sum/total_sum;
        
    def rmse_score(self,X,y):
        return np.sqrt(np.mean((y-self.predict(X))**2));


def RSVD(W,k,over_sampling=10,power=2):
    [U,S,_V] = randomized_svd(W,k,n_oversamples=over_sampling,n_iter=power);
    return (U,S);
      
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

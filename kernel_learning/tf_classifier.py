#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: almert
"""

from numbers import Number
import numpy as np
import tensorflow as tf
from sklearn.base import BaseEstimator
from sklearn.metrics import pairwise
from nystrom import Nystrom

class TFClassifier(BaseEstimator):
    '''
    TFClassifier implements a linear predictor for classification tasks and uses Tensorflow libraries to train 
    the predictor via stochastic gradient methods.
    
    Arguments:
    loss: (string) The type of loss function to be minimized by the predictor. Currently "hinge", "squared_hinge",
    'logistic' and 'squared' losses are supported.
    alpha: (float) The l_2 penalty to be applied to the predictor weights.
    epoch_lim: (int) The maximum number of epochs to train the predictor.
    optimizer: (string or object) The name of the gradient based optimizer ("SGD" or "Adam") or an object from
    tensorflow.train.
    '''
    
    def __init__(self,loss='squared_hinge',alpha=1e-4,epoch_lim=250,optimizer='Adam'):
        
        assert loss in ['hinge','squared_hinge','logistic','squared']
        assert isinstance(alpha,Number) and alpha >= 0
        assert isinstance(epoch_lim,int) or np.issubdtype(epoch_lim, np.integer) 
        self.loss = loss
        self.optimizer = optimizer
        self.alpha = alpha
        self.epoch_lim = epoch_lim
        
    def fit(self,X,y,batch_size=100,patience=10,lr_patience=5,verbose=False):
        '''
        Arguments:
        X: (array) The training data, must have two dimensions.
        y: (array) The training labels, must be one-hot encoded with two dimensions. To produce training labels in 
        this format, one might use sklearn.preprocessing.LabelBinarizer.
        batch_size: (int) The mini-batch size for computing the stochastic gradients.
        patience: (int) Number of epochs without a decrease in average loss before training is stopped.
        lr_patience: (int) Number of epochs without a decrease in average loss before learning rate is reduced.
        verbose: (bool) Whether to print out update messages during predictor training.
        '''
        
        x_in = tf.placeholder(tf.float64,shape=[None,X.shape[1]])
        y_in = tf.placeholder(tf.float64,shape=[None,y.shape[1]])
        lr = tf.placeholder(tf.float64)
        
        W = tf.Variable(tf.zeros([X.shape[1],y.shape[1]],tf.float64))
        b = tf.Variable(tf.constant(0.1,dtype=tf.float64,shape=[y.shape[1]]))
        y_out = tf.matmul(x_in,W)+b
    
        
        if self.loss == 'hinge':
            loss_func = tf.reduce_mean(tf.reduce_sum(tf.maximum(1-(2.*y_in-1)*y_out,0),axis=1))+\
                            self.alpha*tf.reduce_sum(tf.square(W))
        elif self.loss == 'squared_hinge':
            loss_func = tf.reduce_mean(tf.reduce_sum(tf.square(tf.maximum(1-(2.*y_in-1)*y_out,0)),axis=1))+\
                            self.alpha*tf.reduce_sum(tf.square(W))
        elif self.loss == 'logistic':
            if y.shape[1] > 1:
                loss_func = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_in,logits=y_out))+\
                            self.alpha*tf.reduce_sum(tf.square(W))
            else:
                loss_func = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=y_in,logits=y_out))+\
                            self.alpha*tf.reduce_sum(tf.square(W))
        else:
            loss_func = tf.reduce_mean(tf.reduce_sum(tf.square(y_in-y_out),axis=1))+\
                            self.alpha*tf.reduce_sum(tf.square(W))
        
            
        if self.optimizer == 'SGD':
            train_step = tf.train.GradientDescentOptimizer(lr).minimize(loss_func,var_list=[W,b]);
            lr_now = 0.01;
        elif self.optimizer == 'Adam':
            train_step = tf.train.AdamOptimizer(lr).minimize(loss_func,var_list=[W,b]);
            lr_now = 0.001;
        else:
            train_step = self.optimizer(lr).minimize(loss_func,var_list=[W,b]);
            lr_now = 0.001;
                            
        sess = tf.Session();
        sess.run(tf.global_variables_initializer());

        min_avg_loss = np.inf;
        for i in range(self.epoch_lim):
            start = 0;
            end = batch_size;
            n_batch = 1;
            avg_loss = 0;
    
            perm = np.random.permutation(len(X))
            while True:
                b_inds = perm[start:end];
                lval, _ = sess.run([loss_func,train_step],
                    feed_dict={x_in: X[b_inds], y_in: y[b_inds],lr: lr_now});
                avg_loss = ((n_batch-1)*avg_loss + lval)/n_batch;
      
                if end >= len(X):
                    break;
                n_batch += 1;  
                start = start+batch_size;
                end = min(len(X),end+batch_size);

            if verbose:
                print('Epoch %d with average loss %f'%(i,avg_loss));
            
            if avg_loss < min_avg_loss:
                min_avg_loss = avg_loss;
                stop_cnt = 0;
                lr_cnt = 0;
            else:
                stop_cnt += 1;
                lr_cnt += 1;
                if lr_cnt >= lr_patience:
                    lr_now = lr_now*0.1;
                    lr_cnt = 0;
                if stop_cnt >= patience:
                    break;
        self.coef_ = np.transpose(sess.run(W));
        self.intercept_ = np.array(sess.run(b));
        sess.close();
        
    def predict(self,X):
        '''
        Arguments:
        X: (array) The data to perform predictions on.
        
        Returns:
        An array storing the class predictions before thresholding or argmax.
        '''
        
        return np.dot(X,self.coef_.T)+self.intercept_
        
    def score(self,X,y):
        '''
        Arguments:
        X: (array) The test data to perform predictions on.
        y: (array) The true class labels.
        
        Returns:
        The ratio of samples whose class was predicted correctly.
        '''
        
        if len(self.intercept_) > 1:
            return np.mean(np.argmax(y,axis=1)==np.argmax(self.predict(X),axis=1))
        else:
            return np.mean((y >= 0.5) == (self.predict(X) >= 0.5))
            
        
class TFNystromClassifier(BaseEstimator):
    '''
    TFNystromClassifier implements a kernel based predictor for classification tasks and uses Tensorflow libraries to 
    train the predictor via stochastic gradient methods.
    
    Arguments:
    kernel: (string) The type of kernel to be used. Currently 'linear', 'poly', 'rbf', 'laplacian', 'chi2', 
    'additive_chi2' and 'sigmoid' are supported.
    gamma: (float) The scale parameter for the kernel. Equals 1/sqrt(2 bandwidth^2) for kernels with a bandwidth parameter.
    degree: (float) The degree for polynomial kernels.
    coef0: (float) An additive parameter for 'poly' and 'sigmoid' kernels.
    n: (int) Number of random data samples used to produce a Nystrom approximation. If n=len(X), no approximation takes place
    and the full-dimensional kernel mapping is used.
    k: (int) Number of subspace dimensions used. If k=n, the full kernel feature space is used to train the predictor.
    rand_svd: (bool) Whether to use randomized SVD to obtain the Nystrom kernel approximation.
    loss: (string) The type of loss function to be minimized by the predictor. Currently, "hinge", "squared_hinge",
    'logistic' and 'squared' losses are supported.
    alpha: (float) The l_2 penalty to be applied to the predictor weights.
    epoch_lim: (int) The maximum number of epochs to train the predictor.
    optimizer: (string or object) The name of the gradient based optimizer ("SGD" or "Adam") or an object from
    tensorflow.train.
    '''    
    
    def __init__(self,kernel='rbf',gamma=1.,degree=3,coef0=1.,n=1000,k=None,
                 rand_svd=False,loss='squared_hinge',alpha=1.,optimizer='Adam'):
        self.kernel = kernel;
        self.gamma = gamma;
        self.degree= degree;
        self.coef0 = coef0;
        self.n = n;
        self.k = k;
        self.rand_svd = rand_svd;
        self.loss = loss;
        self.alpha = alpha;
        self.optimizer = optimizer;
        
    def fit(self,X,y,batch_size=100):
        '''
        Arguments:
        X: (array) The training data, must have two dimensions.
        y: (array) The training labels, must be one-hot encoded with two dimensions. To produce training labels in 
        this format, one might use sklearn.preprocessing.LabelBinarizer.
        batch_size: (int) The mini-batch size for computing the stochastic gradients.
        '''
        
        mapper = Nystrom(kernel=self.kernel,gamma=self.gamma,degree=self.degree,
                         coef0=self.coef0,n=self.n,k=self.k,rand_svd=self.rand_svd);
        mapper.fit(X);
        self._Xrep = mapper._Xrep;
        clf = TFClassifier(loss=self.loss,alpha=self.alpha,optimizer=self.optimizer)
            
        clf.fit(mapper.transform(X),y,batch_size=batch_size);
        self.dual_coef_ = np.dot(mapper.A,clf.coef_.T).T;
        self.intercept_ = clf.intercept_;
        return
        
    def predict(self,X):
        '''
        Arguments:
        X: (array) The data to perform predictions on.
        
        Returns:
        An array storing the class predictions before thresholding or argmax.
        '''
        
        Ktest = get_kernel_matrix(X,self._Xrep,self.kernel,self.gamma,
                                  self.degree,self.coef0);
        return np.dot(Ktest,self.dual_coef_.T)+self.intercept_;

    def score(self,X,y):
        '''
        Arguments:
        X: (array) The test data to perform predictions on.
        y: (array) The true class labels.
        
        Returns:
        The ratio of samples whose class was predicted correctly.
        '''
        
        return np.mean(np.argmax(y,axis=1)==np.argmax(self.predict(X),axis=1))
        
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
        

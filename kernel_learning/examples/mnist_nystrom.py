#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: almert
"""

import sys
sys.path.append('..')
import numpy as np
import tensorflow as tf
from tf_kernel_network import SubspaceKDI
from nystrom import Nystrom
from sklearn.preprocessing import LabelBinarizer
from tf_classifier import TFClassifier

n_dims = [100,500,1000]
n_epochs = 500
lr_epochs = 100
learning_rate = 0.001
gamma = 0.01

def load_mnist():
    # Replace this part with your own to load from your local directory.
    (Xtr, ytr), (Xte, yte) = tf.keras.datasets.mnist.load_data()
    lb = LabelBinarizer()
    ytr = lb.fit_transform(ytr)
    yte = lb.transform(yte)
    Xtr = Xtr/255.
    Xte = Xte/255.
    return Xtr, ytr, Xte, yte
    
class get_minibatch:
    
    def __init__(self,X,y):
        self.X = X
        self.y = y
        self.perm = np.random.permutation(len(X))
        self.start = 0
        
    def __call__(self,batch_size=100):
        end = self.start+batch_size
        if end > len(self.perm):
            self.perm = np.random.permutation(len(self.perm))
            self.start = 0
            end = min(batch_size,len(self.perm))
        
        batch_ind = self.perm[self.start:end]
        self.start += batch_size
        return self.X[batch_ind], self.y[batch_ind]

(Xtr, ytr, Xte, yte) = load_mnist()
get_next_batch = get_minibatch(Xtr,ytr)

for n in n_dims:
    batch_size = max(1000,2*n)
        
    tf.reset_default_graph() 
    
    lr = tf.placeholder(tf.float64)   
    x = tf.placeholder(tf.float64, shape=(None,Xtr.shape[1]));
    y = tf.placeholder(tf.float64, shape=(None,ytr.shape[1]));

    kernel_func = SubspaceKDI(n=n,
                              kernel='rbf',
                              gamma=gamma,
                              rho=1e-4)
                              
    kernel_func.initialize(Xtr,dtype=tf.float64)
    objective = kernel_func.KDI(x,y,normalize=False)
        
    train_vars = tf.trainable_variables()
    train_step = tf.train.AdamOptimizer(lr).minimize(-objective,
                                                     var_list=train_vars)

    sess = tf.InteractiveSession()
    tf.global_variables_initializer().run()

    lr_t = learning_rate
    min_avg_loss = np.inf
    avg_obj = 0.
    
    for t in range(1,n_epochs+1):
        
        for batch_num in range(1,len(Xtr)//batch_size+1):    
            Xb, yb = get_next_batch(batch_size)
            di_obj, _ = sess.run([objective,train_step],
                                 feed_dict={x: Xb, 
                                            y: yb,
                                            lr: lr_t})
            avg_obj = ((batch_num-1)*avg_obj + di_obj)/batch_num

        print('Epoch %d with average KDI %f'%(t,avg_obj));
        if t % lr_epochs == 0:
            lr_t = lr_t*0.1
      
    print('Fitting a predictor on the optimized Nystrom mapping.')        
    kmap = Nystrom(kernel='rbf',
                   gamma=gamma,
                   n=n)
    kmap.fit(sess.run(kernel_func.X_rep))
    sess.close()
    
    clf = TFClassifier(loss='logistic', 
                       alpha=0.)
    clf.fit(kmap.transform(Xtr),ytr)
    Accuracy = clf.score(kmap.transform(Xte),yte)
        
    print('Accuracy for feature dimensionality %d: %f'%(n,Accuracy))

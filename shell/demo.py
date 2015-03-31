#!/usr/bin/env python
# -*- coding: GB2312 -*-
# Last modified: 

"""docstring
"""

__revision__ = '0.1'

from sklearn.linear_model import LinearRegression as LM

from sklearn import linear_model, decomposition, datasets
from sklearn.cross_validation import KFold
from sklearn.metrics import mean_squared_error
import pandas as pd
import numpy as np
from random import shuffle

train = pd.read_csv('../data/train.csv')
test = pd.read_csv('../data/test.csv')

#calculate the age of each sample. fyi this only worked with pandas 0.14, not 0.13
train['Age']=2015-pd.DatetimeIndex(train['Open Date']).year
test['Age']=2015-pd.DatetimeIndex(test['Open Date']).year

#Extract the age and log transform it
X=np.log(train[['Age']].values.reshape((train.shape[0],1)))
Xt=np.log(test[['Age']].values.reshape((test.shape[0],1)))
y=train['revenue'].values

#randomize the order for cross validation
combined=zip(y,X)
shuffle(combined)
y[:], X[:] = zip(*combined)

import math
#Model Setup
clf=LM()
#clf= linear_model.LogisticRegression()
OUT_TYPE=2
def yparse(y):
    if OUT_TYPE == 1:
        return np.log(y)
    else:
        return np.sqrt(y)

def yout(y):
    if OUT_TYPE == 1:
        return np.exp(y)
    else:
        return y * y

scores=[]

#ss=KFold(len(y), n_folds=3,shuffle=True)
ss=KFold(len(y), n_folds=5)
for trainCV, testCV in ss:
    X_train, X_test, y_train, y_test= X[trainCV], X[testCV], y[trainCV], y[testCV]
    clf.fit(X_train, yparse(y_train))
    y_pred=yout(clf.predict(X_test))

    scores.append(mean_squared_error(y_test,y_pred))

#Average RMSE from cross validation
scores=np.array(scores)
print "CV Score:",np.mean(scores**0.5)

#Fit model again on the full training set
clf.fit(X,yparse(y))
#Predict test.csv & reverse the log transform
yp=yout(clf.predict(Xt))

#Write submission file
sub=pd.read_csv('../data/sampleSubmission.csv')
sub['Prediction']=yp
sub.to_csv('sub.csv',index=False)



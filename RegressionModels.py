# -*- coding: utf-8 -*-
"""
@author: Maria and Jack

Collaborated with Asha, Phil, and Aseel
"""
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import numpy.random as rnd
import sklearn.linear_model as lm
import os

filename = 'data-subset.tsv'
fileweb = 'https://raw.githubusercontent.com/dzinoviev/cmpsc-310/master/Project4/data-subset.tsv'

if not os.path.isfile(filename):
	data = pd.read_csv(fileweb, delimiter = "\t", header=(0), na_values = "-")
else:
	data = pd.read_csv(filename, delimiter = "\t", header=(0), na_values = "-")
data.dropna(inplace = True)

maleData = data[data["Gender"] == "M"]
maleAge = list(pd.Series(maleData['Age']))
maleGen = maleData['IGrade'] - maleData['OGrade']
maleGen = maleGen + (rnd.normal(0,0.1,len(maleGen)))

femData = data[data["Gender"] == "F"]
femAge = list(pd.Series(femData['Age']))
femGen = femData['IGrade'] - femData['OGrade']
femGen = femGen + (rnd.normal(0,0.1,len(femGen)))

d = {"M": 0, "F": 1}
data["Gender"] = data["Gender"].map(d)

selection = rnd.binomial(1, .70, size=len(data)).astype(bool)
testing = data[~selection]
training = data[selection]

#Histogram Generation
def createHistogram():
    plt.hist(femAge, alpha=0.75, label='Female')
    plt.hist(maleAge, alpha=0.75, label='Male')
    plt.legend(loc='upper right')
    plt.savefig('Histograms')
    plt.close()

#Scatterplot Generation
def createScatter():
    plt.scatter(maleData['Age'], maleGen, alpha=0.5, color = 'gold')
    plt.scatter(femData['Age'], femGen, alpha=0.5, color = 'green')
    plt.savefig('Scatter')
    plt.close()

#first model
olm = lm.LinearRegression()
X = training[["Gender", "Age", "OGrade"]]
y = training['IGrade']
olm.fit(X,y)         
model = [np.round(X) for X in olm.predict(X)]
score = olm.score(X,y)
def predict_linear(gender, age, stimulus):
    return olm.predict([gender, age, stimulus])
    
#second step
dummyStim = pd.get_dummies(training["OGrade"])
dummyResp = pd.get_dummies(training["IGrade"])

i_range = 6
log_regressions = [lm.LogisticRegression(C = 20.0) for x in range(i_range)]
X = pd.concat([training[["Gender", "Age"]], dummyResp], axis = 1)

#lost after here, mostly.... if we had some more time, might be able to fix it. 
#Still intend to work on it

#creating the 6 logistic regressions
for i in range(6):
    log_regressions[i].fit(X, dummyStim[i+1])
    print(log_regressions[i].predict(X))
    print(log_regressions[i].score(X, dummyStim[i+1]))
    #Not sure if the scores are too close to 1




import pandas as pd
from matplotlib import pyplot as plt
import numpy.random as rnd
import sklearn.linear_model as lm
import os
import urllib

filename = 'data-subset.tsv'

#if not os.path.isfile(filename):
data = pd.read_csv("https://raw.githubusercontent.com/dzinoviev/cmpsc-310/master/Project4/data-subset.tsv", delimiter = "\t", header=(0), na_values = "-")
#else:
#data = pd.read_csv(filename, delimiter = "\t", header=(0), na_values = "-")

data.dropna(inplace = True)

maleData = data[data["Gender"] == "M"]
maleAge = list(pd.Series(maleData['Age']))
maleGen = maleData['IGrade'] - maleData['OGrade']
maleGen = maleGen + (rnd.normal(0,0.1,len(maleGen)))
femData = data[data["Gender"] == "F"]
femAge = list(pd.Series(femData['Age']))
femGen = femData['IGrade'] - femData['OGrade']
femGen = femGen + (rnd.normal(0,0.1,len(femGen)))
selection = rnd.binomial(1, .70, size=len(data)).astype(bool)
training = data[selection]
testing = data[~selection]



#Histogram Generation
def createHistogram():
    plt.hist(femAge, alpha=0.75, label='Female')
    plt.hist(maleAge, alpha=0.75, label='Male')
    plt.legend(loc='upper right')
    #plt.show()
    plt.savefig('Histograms')
    plt.close()

#Scatterplot Generation
def createScatter():
    plt.scatter(maleData['Age'], maleGen, alpha=0.5, color = 'gold')
    plt.scatter(femData['Age'], femGen, alpha=0.5, color = 'green')
    #plt.show()
    plt.savefig('Scatter')
    plt.close()

def predict_linear(gender,age,stimulus):
    olm = lm.LinearRegression()
    x = training['OGrade']
    y = training['IGrade']
    olm.fit(x,y)
    



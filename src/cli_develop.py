"""Author: Nagabhushan S Baddi (InStep Intern)\nOrganization: Infosys Ltd.\nCLI app (module) to detect botnet traffic using Machine Learning"""

#import the modules
import dataset_load
import models
import threading
import time
import pickle

#load data set
file = open('../dataset/flowdata.pickle', 'rb')
sd = pickle.load(file)
X, Y, XT, YT = sd[0], sd[1], sd[2], sd[3]

def loadModel(modelName, fileName=None):
    """load the modelName ML model and test the accuracy"""
    global X, Y, XT, YT
    mlalgo = modelName
    if mlalgo == 'Decision Tree':
        model = models.DTModel(X, Y, XT, YT)
        model.start()
    elif mlalgo == 'Naive Bayes':
        model = models.NBModel(X, Y, XT, YT)
        model.start()
    elif mlalgo == 'SVM':
        model = models.SVMModel(X, Y, XT, YT)
        model.start()
    elif mlalgo == 'K Nearest Neighbours':
        model = models.KNNModel(X, Y, XT, YT)
        model.start()
    elif mlalgo == 'Logistic Regression':
        model = models.LogModel(X, Y, XT, YT)
        model.start()
    else:
        model = models.ANNModel(X, Y, XT, YT)
        model.start()
import pandas as pd
import numpy as np
import sklearn
import math
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.datasets import make_regression
from sklearn import tree, ensemble
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
import math
import matplotlib.pyplot as plt
from numpy import mean
import seaborn as sns
import scipy
import io
import csv

from decimal import Decimal

from sklearn.metrics import confusion_matrix,accuracy_score,recall_score,precision_score,f1_score,ConfusionMatrixDisplay 

class FuzzyKNNEC:
  def __init__(self):
    self.Xtrain = None
    self.Ytrain = None
    self.k      = None
    self.nclass = None
    self.kfold_Result = None

  def fit(self,train,test,k,nclass,cv):
    self.k      = k
    self.nclass = nclass
    self.kfold(train,test,cv)

  def predict(self,Xtest): 
    Ptrain = []
    for i in range(len(Xtest)):
      Ptrain.append(self.pred(Xtest.iloc[i]))
    return Ptrain

  def pred(self,Xtest):
    k      = self.k
    nclass = self.nclass
    Xtrain = self.Xtrain.copy()
    Ytrain = self.Ytrain.copy()
    jarak  = []
    d      = []
    for i in range(len(Xtrain)) :
      data = Xtrain.iloc[i]
      jarak2 = sum([(num1-num2)*(num1-num2) for num1, num2 in zip(list(data.values),list(Xtest.values))])**0.5
      jarak.append(jarak2)
      d.append(jarak[i]**(-2))


    Xtrain['jarak'] = pd.DataFrame(jarak)
    Xtrain['d'] = pd.DataFrame(d)
    Xtrain['RiskLevel'] = Ytrain

    S = []
    for i in range(nclass):
      train0 = Xtrain[Xtrain['RiskLevel'] == i]
      data=sorted(list(train0['d'].values), key = lambda x:float(x))
      data=list(pd.DataFrame(data)[0].dropna().values)
      S.append(sum(data[:k]))

    S = list(pd.DataFrame(S)[0].fillna(0).values)
    print(S)
    PS = [num1/sum(S) for num1 in S]
    print(PS,"=",PS.index(max(PS)))
    return PS.index(max(PS))

  def kfold(self,Xtrain,Ytrain,cv):
      dataDict = {}
      cv = int(len(Xtrain)/cv)
      index = 1
      for i in range(0,len(Xtrain),cv):
        if i+cv<len(Xtrain):
          dataDict['k'+str(index)]= Xtrain.loc[i:i+cv-1]
          dataDict['testk'+str(index)]= Ytrain.loc[i:i+cv-1]
          index+=1

      kfold_Result = []
      Xfiks        = 0
      Yfiks        = 0
      Resultfiks   = 0
      for i in dataDict.keys():
        if not i.__contains__('test'):
          traingrup = []
          testgrup = []
          for d in dataDict.keys():
            if d != i and not d.__contains__('test'):
              traingrup.append(dataDict[d])
              testgrup.append(dataDict['test'+d])

          self.Xtrain  = pd.concat(traingrup,axis=0).reset_index(drop=True)
          self.Ytrain  = pd.concat(testgrup).reset_index(drop=True)
          acuracy = accuracy_score(dataDict['test'+i].values, np.array(self.predict(dataDict[i])))*100
          kfold_Result.append(acuracy)

          if acuracy > Resultfiks:
            Xfiks        = pd.concat(traingrup,axis=0).reset_index(drop=True)
            Yfiks        = pd.concat(testgrup).reset_index(drop=True)
            Resultfiks   = acuracy

      self.Xtrain = Xfiks
      self.Ytrain = Yfiks
      self.kfold_Result = Resultfiks
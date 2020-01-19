"""
Created on Tue Dec 24 20:25 2019
@author: anne marie delaney
         eoin brophy
         
Script to generate a training and test data set of sine waves
"""


import pandas as pd
import numpy as np
import random

"""Create a training set of sine waves with 10000 records"""
a = np.arange(0.1,0.9,0.02)
x = np.arange(0,20,0.5)
r = np.arange(2,6.1,0.1)
count = 0
fs = len(x)
y = np.zeros((1,len(x)))

for n in range(10000):
  amp = a[random.randint(0,len(a)-1)]
  rad = r[random.randint(0,len(r)-1)]
  phase = random.uniform(-1,1)*np.pi
  y = np.append(y,amp*np.sin(((2*np.pi*rad*x)+phase)/fs).reshape((1,len(x))),axis = 0)
     
data = pd.DataFrame(y[1:][:])  
data.to_csv('./sinedata_v2.csv', header = False, index = False)

"""Creating a test set of sine waves with 3000 records"""
a = np.arange(0.1,0.9,0.02)
x = np.arange(0,20,0.5)
r = np.arange(2,6.1,0.1)
count = 0
fs = len(x)
y = np.zeros((1,len(x)))

for n in range(3000):
  amp = a[random.randint(0,len(a)-1)]
  rad = r[random.randint(0,len(r)-1)]
  phase = random.uniform(-1,1)*np.pi

  y = np.append(y,amp*np.sin(((2*np.pi*rad*x)+phase)/fs).reshape((1,len(x))),axis = 0)
  
data = pd.DataFrame(y[1:][:])  
data.to_csv('sinedata_test_v2.csv', header = False, index = False)
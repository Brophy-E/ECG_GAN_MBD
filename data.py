# -*- coding: utf-8 -*-
"""
Created on Tue Sep 17 11:09:33 2019

@author: anne marie delaney
         eoin brophy

Data Loading module for GAN training
------------------------------------

Creating the Training Set

Creating the pytorch dataset class for use with Data Loader to enable batch training of the GAN
"""
import torch
from torch.utils.data import Dataset
import pandas as pd

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

class ECGData(Dataset):
  #This is the class for the ECG Data that we need to load, transform and then use in the dataloader.
  def __init__(self,source_file,class_id, transform = None):
    self.source_file = source_file
    data = pd.read_csv(source_file, header = None)
    class_data = data[data[187]==class_id]
    self.data = class_data.drop(class_data.iloc[:,187],axis=1)
    self.transform = transform
    self.class_id = class_id
    
  def __len__(self):
    return self.data.shape[0]
    
  def __getitem__(self,idx):
    sample = self.data.iloc[idx]
    if self.transform:
        sample = self.transform(sample)
    return sample

"""Including the function that will transform the dataframe to a pytorch tensor"""

class PD_to_Tensor(object):
    def __call__(self,sample):
      return torch.tensor(sample.values).to(device)


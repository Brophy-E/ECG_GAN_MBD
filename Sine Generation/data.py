"""
Created on Tue Dec 24 20:25 2019
@author: anne marie delaney
         eoin brophy
Data Loading module for GAN training
------------------------------------
Creating the Training Set
Creating the pytorch dataset class for use with Data Loader to enable batch training of the GAN
"""


import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

class SineData(Dataset):
  #This is the class for teh ECG Data that we need to load, transform and then use in teh dataloader.
  def __init__(self,source_file,transform = None):
    self.source_file = source_file
    self.data  = pd.read_csv(source_file, header = None)
    self.transform = transform
    
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
      return torch.tensor(sample.values).cuda()


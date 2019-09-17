# -*- coding: utf-8 -*-
"""
Created on Tue Sep 17 11:16:34 2019

@author: anne marie delaney
         eoin brophy
         
Module of the GAN model for time series synthesis.

"""

import torch
import torch.nn as nn

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


""" 
NN Definitions
---------------
Defining the Neural Network Classes to be evaluated in this Notebook

Minibatch Discrimination
--------------------------
Creating a module for Minibatch Discrimination to avoid mode collapse as described:
https://arxiv.org/pdf/1606.03498.pdf
https://torchgan.readthedocs.io/en/latest/modules/layers.html#minibatch-discrimination

"""

class MinibatchDiscrimination(nn.Module):
   def __init__(self,input_features,output_features,minibatch_normal_init, hidden_features=16):
      super(MinibatchDiscrimination,self).__init__()
      
      self.input_features = input_features
      self.output_features = output_features
      self.hidden_features = hidden_features
      self.T = nn.Parameter(torch.randn(self.input_features,self.output_features, self.hidden_features))
      if minibatch_normal_init == True:
        nn.init.normal(self.T, 0,1)
      
   def forward(self,x):
      M = torch.mm(x,self.T.view(self.input_features,-1))
      M = M.view(-1, self.output_features, self.hidden_features).unsqueeze(0)
      M_t = M.permute(1, 0, 2, 3)
      # Broadcasting reduces the matrix subtraction to the form desired in the paper
      out = torch.sum(torch.exp(-(torch.abs(M - M_t).sum(3))), dim=0) - 1
      return torch.cat([x, out], 1)

"""
Discriminator Class
-------------------
This discriminator has a parameter num_cv which allows the user to specify if 
they want to have 1 or 2 Convolution Neural Network Layers.

"""

class Discriminator(nn.Module):
  def __init__(self,seq_length,batch_size,minibatch_normal_init, n_features = 1, num_cv = 1, minibatch = 0, cv1_out= 10, cv1_k = 3, cv1_s = 4, p1_k = 3, p1_s = 3, cv2_out = 10, cv2_k = 3, cv2_s = 3 ,p2_k = 3, p2_s = 3):
      super(Discriminator,self).__init__()
      self.n_features = n_features
      self.seq_length = seq_length
      self.batch_size = batch_size
      self.num_cv = num_cv
      self.minibatch = minibatch
      self.cv1_dims = int((((((seq_length - cv1_k)/cv1_s) + 1)-p1_k)/p1_s)+1)
      self.cv2_dims = int((((((self.cv1_dims - cv2_k)/cv2_s) + 1)-p2_k)/p2_s)+1)
      self.cv1_out = cv1_out
      self.cv2_out = cv2_out
      
      #input should be size (batch_size,num_features,seq_length) for the convolution layer
      self.CV1 = nn.Sequential(
                  nn.Conv1d(in_channels = self.n_features, out_channels = int(cv1_out),kernel_size = int(cv1_k), stride = int(cv1_s))
                  ,nn.ReLU()        
                  ,nn.MaxPool1d(kernel_size = int(p1_k), stride = int(p1_s))   
                 )
      
      # 2 convolutional layers
      if self.num_cv > 1:
        self.CV2 = nn.Sequential(
                      nn.Conv1d(in_channels = int(cv1_out), out_channels = int(cv2_out) ,kernel_size =int(cv2_k), stride = int(cv2_s))
                      ,nn.ReLU()
                      ,nn.MaxPool1d(kernel_size = int(p2_k), stride = int(p2_s))
                  )
        
        #Adding a minibatch discriminator layer to add a cripple affect to the discriminator so that it needs to generate sequences that are different from each other.
        
        if   self.minibatch > 0:
          self.mb1 = MinibatchDiscrimination(self.cv2_dims*cv2_out,self.minibatch, minibatch_normal_init)
          self.out = nn.Sequential(nn.Linear(int(self.cv2_dims*cv2_out)+self.minibatch,1),nn.Sigmoid()) # to make sure the output is between 0 and 1
        else:
          self.out = nn.Sequential(nn.Linear(int(self.cv2_dims*cv2_out),1),nn.Sigmoid()) # to make sure the output is between 0 and 1 
      
      # 1 convolutional layer
      else:
        
        #Adding a minibatch discriminator layer to add a cripple affect to the discriminator so that it needs to generate sequences that are different from each other.
        if self.minibatch > 0 :
          
          self.mb1 = MinibatchDiscrimination(int(self.cv1_dims*cv1_out),self.minibatch, minibatch_normal_init)
          self.out = nn.Sequential(nn.Linear(int(self.cv1_dims*cv1_out)+self.minibatch,1),nn.Dropout(0.2),nn.Sigmoid()) # to make sure the output is between 0 and 1
        else:
          self.out = nn.Sequential(nn.Linear(int(self.cv1_dims*cv1_out),1),nn.Sigmoid())  
          
 

  def forward(self,x):
     # print("Calculated Output dims after CV1: "+str(self.cv1_dims))
     # print("input: "+str(x.size()))
      x = self.CV1(x.view(self.batch_size,1,self.seq_length))
     # print("CV1 Output: "+str(x.size()))
      
      #2 Convolutional Layers
      if self.num_cv > 1:   
        
        x = self.CV2(x)
        x = x.view(self.batch_size,-1)
        
      #  print("CV2 Output: "+str(x.size()))
        if self.minibatch > 0:
             x = self.mb1(x.squeeze())
       #      print("minibatch output: "+str(x.size()))
             x = self.out(x.squeeze())
        else:
            
             x = self.out(x.squeeze())
        
      # 1 convolutional layers
      else:
        
        x = x.view(self.batch_size,-1)
       
        #1 convolutional Layer and minibatch discrimination
        if self.minibatch > 0:
             x = self.mb1(x)
             x = self.out(x)
        #1 convolutional Layer and no minibatch discrimination
        else:
             x = self.out(x)
      
      
      
      return x

"""
Generator Class
---------------
This defines the Generator for evaluation. The Generator consists of two LSTM 
layers with a final fully connected layer.

"""

class Generator(nn.Module):
  def __init__(self,seq_length,batch_size,n_features = 1, hidden_dim = 50, 
               num_layers = 2, tanh_output = False):
      super(Generator,self).__init__()
      self.n_features = n_features
      self.hidden_dim = hidden_dim
      self.num_layers = num_layers
      self.seq_length = seq_length
      self.batch_size = batch_size
      self.tanh_output = tanh_output
      

      
      self.layer1 = nn.LSTM(input_size = self.n_features, hidden_size = self.hidden_dim, 
                                  num_layers = self.num_layers,batch_first = True#,dropout = 0.2,
                                 )
      if self.tanh_output == True:
        self.out = nn.Sequential(nn.Linear(self.hidden_dim,1),nn.Tanh()) # to make sure the output is between 0 and 1 - removed ,nn.Sigmoid()
      else:
        self.out = nn.Linear(self.hidden_dim,1) 
      
  def init_hidden(self):
      weight = next(self.parameters()).data
      hidden = (weight.new(self.num_layers, self.batch_size, self.hidden_dim).zero_().to(device), weight.new(self.num_layers, self.batch_size, self.hidden_dim).zero_().to(device))
      return hidden
  
  def forward(self,x,hidden):
      
      x,hidden = self.layer1(x.view(self.batch_size,self.seq_length,1),hidden)
      
      x = self.out(x)
      
      return x #,hidden
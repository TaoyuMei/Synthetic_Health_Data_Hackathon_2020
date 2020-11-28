# -*- coding: utf-8 -*-
"""SYNHACK.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1R0tl1NvdYdNoRyCGHs5nVv_6giOG8csP
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.nn.init as init
import pandas as pd
from torch.utils.data import Dataset
from torch.utils.data import random_split
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.functional import relu, elu, relu6, sigmoid, tanh, softmax
from torch.nn.parameter import Parameter

# pd_real=pd.read_csv('real_data.csv')
# pd_real=pd_real[['time_in_hospital', 'num_lab_procedures', 'num_medications', 'number_outpatient', 'number_emergency', 'number_inpatient', 'number_diagnoses']]

# print(pd_real)
# print(pd_real.iloc[:,1:])
# print(pd_real.iloc[:,0])
!wget https://raw.githubusercontent.com/CrimsonScythe/synhack/master/real_data_dum1.csv

from sklearn import preprocessing
import torch
import numpy as np
import pandas as pd

# pd_real=pd.read_csv('real_data_dum1.csv')
# pd_real['readmitted'].replace({'<30':1, '>30':1, 'NO': 0}, inplace=True)
# pd_real = pd_real.iloc[:, 1:10]
# print(pd_real)

# pd_real.corr().to_csv('corr.csv')


# import matplotlib.pyplot as plt

# plt.matshow(pd_real.corr())
# plt.show()

# le = preprocessing.LabelEncoder()
# pd_real['ADMIT'] = le.fit_transform(np.ravel(y.values))

# x = pd_real.iloc[:, pd_real.columns != 'readmitted']
# x = pd_real.iloc[:, pd_real.columns != 'Unnamed: 0']

# print(x['readmitted'])

# import sys
# np.set_printoptions(threshold=sys.maxsize)
# pd_real=pd.read_csv('real_data_dum1.csv')
# x = pd_real.iloc[:, pd_real.columns != 'readmitted']
# x = pd_real.iloc[:, pd_real.columns != 'Unnamed: 0']

# # x = x.iloc[:, 102:]
# x = x.iloc[:, :8]
# print(x)
# # print(x)
# # print(x.columns)
# for i in x.columns:
#     print(i)
# y = pd_real[['readmitted']]
# # print(y)
# print(y.columns)

# from sklearn import preprocessing
# import torch
# import numpy as np
# le = preprocessing.LabelEncoder()
# targets = le.fit_transform(np.ravel(y.values))
# # print(targets)
# print(np.unique(targets))

# from collections import Counter
# # input =  ['a', 'a', 'b', 'b', 'b']
# c = Counter( targets )

# print( c.items() )

# zz=le.inverse_transform([0,1,2])
# print(zz)

from sklearn import preprocessing
import torch
import numpy as np
class HealthDataset(Dataset):
  def __init__(self, file_name):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        # self.landmarks_frame = pd.read_csv(csv_file)
        # pd_real=pd.read_csv(file_name)
        # pd_real=pd_real[['time_in_hospital', 'num_lab_procedures', 'num_medications', 'number_outpatient', 'number_emergency', 'number_inpatient', 'number_diagnoses']]
        # x = pd_real.iloc[:,1:].values
        # y = pd_real.iloc[:,0].values #time in hospital

        pd_real=pd.read_csv('real_data_dum1.csv')
        pd_real['readmitted'].replace({'<30':'YES', '>30':'YES'}, inplace=True)

        # x = pd_real.drop(['readmitted', 'Unnamed: 0'], axis=1).values
        '''removing extra features'''
        # x = pd_real.drop(['num_lab_procedures', 'num_procedures', 'num_medications',  'number_outpatient',  'number_emergency',  'number_inpatient', 'number_diagnoses', 'readmitted', 'Unnamed: 0'],axis=1).values
        # x = pd_real.iloc[:, 102:].values
        # x = pd_real.iloc[:, :8].values
        x = pd_real.iloc[:, 1:8].values

        # print(x)

        y = pd_real[['readmitted']].values
        print(len(y))
        print(len(x))
        self.x_train=torch.tensor(x)
        print(self.x_train.shape)
        
        le = preprocessing.LabelEncoder()
        targets = le.fit_transform(np.ravel(y))
        self.y_train=targets
        print(self.y_train.shape)

  def __len__(self):
        return len(self.y_train)

  def __getitem__(self, idx):
        return self.x_train[idx], self.y_train[idx]
        
health_dataset_real=HealthDataset(file_name='real_data_dum.csv')
# test_set, val_set, train_set = random_split(health_dataset_real, [11766, 11766, 54909], generator=torch.Generator().manual_seed(42))

# '''train loader'''
# train_loader = torch.utils.data.DataLoader(train_set, batch_size=9, shuffle=True) 
# '''test loader'''
# test_loader = torch.utils.data.DataLoader(test_set, batch_size=6, shuffle=True) 
# '''val loader'''
# val_loader = torch.utils.data.DataLoader(val_set, batch_size=6, shuffle=True) 

# train_loader_iter = iter(train_loader)
# test_loader_iter = iter(test_loader)
# val_loader_iter = iter(test_loader)

#-----------------------------------------------------------------------------------

test_set , train_set = random_split(health_dataset_real, [23532, 54909], generator=torch.Generator().manual_seed(42))

'''train loader'''
train_loader = torch.utils.data.DataLoader(train_set, batch_size=9, shuffle=True) 
'''test loader'''
test_loader = torch.utils.data.DataLoader(test_set, batch_size=12, shuffle=True) 


# health_dataset_real_loader=torch.utils.data.DataLoader(health_dataset_real, batch_size=11, shuffle=True) 
# # train_loader_iter = iter(health_dataset_real_loader)
tr=iter(train_loader)
print(tr.next())

'''Hyper params'''
num_features=7
num_hidden=5
num_output=1

'''Define network architecture'''
class Net(nn.Module):
  def __init__(self, num_feautures, num_hidden, num_output):
      super(Net, self).__init__()
      self.L1 = nn.Linear(num_features, num_hidden)
      self.L2 = nn.Linear(num_hidden, num_hidden)
      self.L3 = nn.Linear(num_hidden, num_output)

      # self.L1 = nn.Linear(num_features, num_hidden)
      # self.L2 = nn.Linear(num_hidden, num_hidden)
      # self.L3 = nn.Linear(num_hidden, num_output)

      # self.w1 = Parameter(init.kaiming_normal_(torch.Tensor(num_hidden, num_features)))
      # self.b1 = Parameter(init.constant_(torch.Tensor(num_hidden), 0))

      # self.w2 = Parameter(init.kaiming_normal_(torch.Tensor(num_hidden, num_hidden)))
      # self.b2 = Parameter(init.constant_(torch.Tensor(num_hidden), 0))

        # Batch norm
      # self.b2 = torch.nn.BatchNorm1d(self.b_2.size()[0])
        
        
      # self.w3 = Parameter(init.kaiming_normal_(torch.Tensor(num_output, num_hidden)))
      # self.b3 = Parameter(init.constant_(torch.Tensor(num_output), 0))

      self.act = torch.nn.ReLU()
      self.softmax = torch.nn.Softmax()
      self.sigmoid = torch.nn.Sigmoid()

      # self.W_1 = Parameter(torch.tensor(num_hidden, num_features))
      # self.b_1 = Parameter(torch.tensor(num_hidden))
  def forward(self, x):

      # x = F.linear(x, self.w1, self.b1)
      # x = self.act(x)
      # x = F.linear(x,self.w2, self.b2)
      # x = self.act(x)
      # x = F.linear(x,self.w3, self.b3)

      x = self.L1(x)
      x = self.act(x)
      x = self.L2(x)
      x = self.act(x)
      x = self.L3(x)
     
      return x

net = Net(num_features, num_hidden, num_output)

'''Loss function and optimizer'''
import torch.optim as optim

criterion = torch.nn.BCEWithLogitsLoss()
# criterion = torch.nn.MSELoss()
# criterion = torch.nn.BCEWithLogitsLoss()
optimizer = optim.Adam(params=net.parameters(), lr=0.001)

'''training loop'''
from sklearn.metrics import mean_squared_error
from sklearn.metrics import accuracy_score
num_epoch=1
losses=[]
losses_ev=[]
net.train()
for epoch in range(num_epoch):
    # if (epoch==1):
    #   break
    net.train()
    running_loss = 0.0
    for i, data in enumerate(train_loader, 0):
        '''get features and corresponding output'''
        x, y = data
        '''zero grads'''
        # print(x)
        optimizer.zero_grad()

        '''forward pass, loss backwaard and optimize'''
        outputs = net(x.float())
        
        outputs=outputs.view(9,)
        # print(outputs)
        # print(F.sigmoid(outputs))
        # print(outputs.shape)
        # print(y.shape)
        # print('yolo')
        loss = criterion(outputs, y.type_as(outputs))
        # print(y)
        # print('ss')
        # print(outputs)


        # print(F.sigmoid(outputs))
        # print(torch.round(F.sigmoid(outputs)))
        # print(y)

        loss.backward()
        optimizer.step()
        
        # print('[%d, %5d] loss: %.3f mse: %.3f' % (epoch + 1, i + 1, running_loss/2000, mean_squared_error(y.float(), outputs.float().detach().numpy())))
        running_loss += loss


          # print('[%d, %5d] loss: %.3f mse: %.3f' % (epoch + 1, i + 1, running_loss/2000, mean_squared_error(y.float(), outputs.float().detach().numpy())))
          # running_loss=0.0
        # print(loss)
        # print(y)
        # print(F.sigmoid(outputs))
        # print(F.sigmoid(outputs))
        if i % 500 == 499:
          print('[%d, %5d] train loss: %.3f acc: %.3f' % (epoch + 1, i + 1, running_loss/500, accuracy_score(y, torch.round(F.sigmoid(outputs)).detach().numpy())))
          

    '''validation'''
    net.eval()
    running_loss=0.0
    for i, data in enumerate(test_loader, 0):
      x, y = data
      preds=net(x.float())
      preds=preds.view(12,)
      loss=criterion(preds, y.type_as(preds))
      running_loss += loss
      if i % 100 == 99:
          print('[%d, %5d] val loss: %.3f acc: %.3f' % (epoch + 1, i + 1, running_loss/100, accuracy_score(y, torch.round(F.sigmoid(preds)).detach().numpy())))

          # print('[%d, %5d] loss: %.3f mse: %.3f' % (epoch + 1, i + 1, running_loss/2000, mean_squared_error(y.float(), outputs.float().detach().numpy())))
          # running_loss=0.0


print('finished training')

# '''test'''
# for epoch in range(num_epoch):
#     running_loss = 0.0    
#     '''validation'''
#     net.eval()
#     running_loss=0.0
#     for i, data in enumerate(val_loader_iter, 0):
#       x, y = data
#       preds=net(x.float())
#       preds=preds.view(6,)
#       loss=criterion(preds.float(), y.float())
#       running_loss += loss.item()
#       if i % 100 == 99:
#           print('[%d, %5d] val loss: %.3f ' % (epoch + 1, i + 1, running_loss/100))

#           # print('[%d, %5d] loss: %.3f mse: %.3f' % (epoch + 1, i + 1, running_loss/2000, mean_squared_error(y.float(), outputs.float().detach().numpy())))
#           running_loss=0.0


# print('finished training')
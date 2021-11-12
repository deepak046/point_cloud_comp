from time import time
import os
import h5py
import random
import pytorch3d
from torch.utils.data import Dataset
import numpy as np
import pickle
import torch.nn as nn
import math
import torch
from pytorch3d.loss import chamfer_distance
from torch.utils.data import DataLoader
from tqdm import tqdm
from torch.utils.data import Dataset
import numpy as np
import os
import matplotlib.pyplot as plt
import sys
import torch.nn.functional as F
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_distances

class PCDataLoader(Dataset):
    def __init__(self, points, gt,labels, n_points=2048):
        self.n_points = n_points
        self.points = points
        self.gt = gt
        self.labels = labels

    def __len__(self):
        return len(self.points)

    def __getitem__(self, index):

        pc = self.points[index][:,0:3]
        gt = self.gt[index][:,0:3]
        labels = self.labels[index]
        return pc, gt, labels

pts_train =  []
pts_val = []
gt_train =  []
gt_val = []
labels_val = []
labels_train = []
dic = {'03001627' : 0, '03636649':1, '04379243':2, '02958343':3, '02933112':4, '02691156':5, '04256520':6, '04530566':7}


t = time()
def get_data(folder1,folder2, li, labels = None):
  for root, dirs, files in os.walk("./shapenet/" + folder1 + "/" + folder2):
    for dirss in dirs:
      path = (os.path.join(root , dirss))
      for _, _, files in os.walk(path):
        for filess in tqdm(files, position = 0 ,leave = True): 
        # for filess in files:
          pth = (os.path.join(path, filess))
          f = h5py.File(pth, "r+")
          if folder2 == "gt":
            li.append(f['data'][:])
          else:
            li.append(f['data'][:])
          if labels is not None:
            labels.append(dic[dirss])

get_data("train", "partial", pts_train, labels_train)
get_data("val", "partial", pts_val, labels_val)
get_data("val", "gt", gt_val)
get_data("train", "gt", gt_train)

x_train, x_test = np.array(pts_train), np.array(pts_val)
y_train, y_test = np.array(gt_train), np.array(gt_val)
labels_train, labels_test = np.array(labels_train), np.array(labels_val)
print(len(pts_train), len(pts_val), len(gt_train), len(gt_val),len(labels_val), len(labels_train))
print(time() - t)


train_dataset = PCDataLoader(x_train, y_train, labels_train)
test_dataset = PCDataLoader(x_test, y_test, labels_test)



class Transform(nn.Module):
   def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv1d(128*3+128,512,1)
        self.conv2 = nn.Conv1d(512,512,1)
        self.conv3 = nn.Conv1d(512,512,1)
        self.conv4 = nn.Conv1d(512,1024,1)
       

        self.bn1 = nn.BatchNorm1d(512)
        self.bn2 = nn.BatchNorm1d(512)
        self.bn3 = nn.BatchNorm1d(512)
        # self.bn4 = nn.BatchNorm1d(512)

   def forward(self, inp_global):

        xb = F.relu(self.bn1(self.conv1(inp_global)))
        xb = F.relu(self.bn2(self.conv2(xb)))
        xb = F.relu(self.bn3(self.conv3(xb)))
        xb = self.conv4(xb)
        xb = nn.MaxPool1d(xb.size(-1))(xb)
        output = nn.Flatten(1)(xb)
        return output


class GlobalEncode(nn.Module):
   def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv1d(3,64,1)
        self.conv2 = nn.Conv1d(64,64,1)
        self.conv3 = nn.Conv1d(64,128,1)
        self.conv4 = nn.Conv1d(128,128,1)

        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(64)
        self.bn3 = nn.BatchNorm1d(128)
        # self.bn4 = nn.BatchNorm1d(512)
      
       
   def forward(self, input):

        xb = F.relu(self.bn1(self.conv1(input)))
        xb = F.relu(self.bn2(self.conv2(xb)))
        xb = F.relu(self.bn3(self.conv3(xb)))
        xb = self.conv4(xb)
        xb = nn.MaxPool1d(xb.size(-1))(xb)
        return nn.Flatten(1)(xb)

class PointCloud(nn.Module):
    def __init__(self):
        super().__init__()
        self.encode = Transform()
        self.global_encode = GlobalEncode()

        self.nb_heads = 8
        self.conv1 = nn.Conv1d(in_channels=1024*self.nb_heads, 
                               out_channels=1024*self.nb_heads, kernel_size=1, 
                               groups=self.nb_heads)
        self.conv2 = nn.Conv1d(in_channels=1024*self.nb_heads, 
                               out_channels=1024*self.nb_heads, kernel_size=1, 
                               groups=self.nb_heads)
        self.conv3 = nn.Conv1d(in_channels=1024*self.nb_heads, 
                               out_channels=256*3*self.nb_heads, kernel_size=1, 
                               groups=self.nb_heads)
        self.conv4 = nn.Conv1d(in_channels=256*3*self.nb_heads, 
                               out_channels=256*3*self.nb_heads, kernel_size=1, 
                               groups=self.nb_heads)

        

        self.bn1 = nn.BatchNorm1d(1024*self.nb_heads)
        self.bn2 = nn.BatchNorm1d(1024*self.nb_heads)
        self.bn3 = nn.BatchNorm1d(256*3*self.nb_heads)
        self.dp = nn.Dropout(p=0.2)
        

        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 128)
        self.fc4 = nn.Linear(128, 8)
        self.dropout = nn.Dropout(p=0.2)

    def forward(self, input, input_knn):
        inp_global = self.global_encode(input).unsqueeze(1)
        inp_global = inp_global.repeat(1,2048,1)
        xb = torch.cat((input_knn, inp_global), dim = 2).transpose(1,2)
        enc = self.encode(xb)

        xb = enc.repeat(1,8).unsqueeze(2)
        xb = F.relu(self.conv1(xb))
        xb = F.relu(self.conv2(xb))
        xb = F.relu(self.conv3(xb))
        xb = self.conv4(xb)
        output = xb.reshape(-1, 2048,3)

        xb = self.dropout(F.relu(self.fc1(enc)))
        xb = self.dropout(F.relu(self.fc2(xb)))
        xb = F.relu(self.fc3(xb))
        labels = self.fc4(xb)

        return output, labels


def knn_matrix(knnm, k):
  with torch.no_grad():

      bs = knnm.shape[0]
      a = pytorch3d.ops.knn_points(knnm, knnm, K = k)[1].reshape(bs, -1).unsqueeze(2)
      index = torch.cat((a,a,a), dim=2)
      inp = knnm.gather(dim=1, index=index).reshape(bs,-1,k*3)
  return inp


bs = 128
train_dataloader = DataLoader(train_dataset, batch_size=bs, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=bs, shuffle=True)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)
pointcloud = PointCloud().to(device)
print(sum(p.numel() for p in pointcloud.parameters() if p.requires_grad)/1e6)
classify_loss = nn.CrossEntropyLoss()
epoch = 0

optimizer = torch.optim.Adam(pointcloud.parameters(), lr=1e-3)
# scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,
#       mode='min', factor=0.32, patience=3, threshold=0.029,
#       threshold_mode='rel', cooldown=0, min_lr=0, eps=1e-08, verbose=1)
scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, 0.78, last_epoch=-1)




num_loss = 100
val_check = 200
num_test = len(test_dataloader)
best_val = 21
val_ = 1
total = 0
correct = 0
lr = 0.001


for epoch in range(epoch,100): 

    # f = open("/content/lr.txt", 'r')
    # lr = float(f.read())
    # optimizer.param_groups[0]['lr'] = lr

    
    running_loss1 = []
    running_loss2 = []
    pbar = tqdm(total=len(train_dataloader) ,position=0, leave=True)
    for i, (inputs, gt, labels) in enumerate(train_dataloader):
        pointcloud.train()
        pbar.update()

        inputs, gt, labels = inputs.to(device).float(), gt.to(device).float(),labels.to(device)
        inputs_knn = knn_matrix(inputs, 128)

        optimizer.zero_grad()
        
        outputs, output_labels = pointcloud(inputs.transpose(1,2), inputs_knn)
        loss1 = chamfer_distance(gt, outputs)[0]

        loss2 = 0.001*classify_loss(output_labels, labels)

        loss = loss1 + loss2
        loss.backward()
        # torch.nn.utils.clip_grad_value_(pointcloud.parameters(), clip_value=5)
        optimizer.step()

        ## print statistics
        running_loss1.append(loss1.item())
        running_loss2.append(loss2.item())
        _, predicted = torch.max(output_labels, 1)
        total += labels.size(0)
        correct += (predicted == labels.data).sum().item()
    
        # if (i+1)%num_loss == 0:    # print every 10 mini-batches
    if 1:
                accu = (correct*100)/total
                print('epoch: %.1f,  chamfer: %.3f, classify: %.3f,  accu: %.3f' % (epoch, (np.sum(running_loss1)*1e4)/(len(running_loss1)), 
                                                                        (np.sum(running_loss2)*1e4)/(len(running_loss2)), accu))
                running_loss1 = []
                running_loss2 = []
                total = 0
                correct = 0


        ## validation

        # if (i)%val_check == 0 and val_: 

                pointcloud.eval()
                val_loss1 = []
                val_loss2 = []
                total = 0
                correct = 0

                with torch.no_grad():
                  for i, (inputs, gt, labels) in enumerate(test_dataloader):
                    inputs, gt, labels = inputs.to(device).float(), gt.to(device).float(), labels.to(device)
                    inputs_knn = knn_matrix(inputs, 128)
                    outputs, output_labels = pointcloud(inputs.transpose(1,2), inputs_knn)

                    loss1 = chamfer_distance(gt, outputs)[0]
                    loss2 = 0.001*classify_loss(output_labels, labels)

                    val_loss1.append(loss1.item())
                    val_loss2.append(loss2.item())

                    _, predicted = torch.max(output_labels, 1)
                    total += labels.size(0)
                    correct += (predicted == labels.data).sum().item()

                  val_loss1 = np.round((np.sum(val_loss1)*1e4)/(len(val_loss1)), 3)
                  val_loss2 = np.round((np.sum(val_loss2)*1e4)/(len(val_loss2)), 3)
                  print('')
                  accu = (correct*100)/total
                  print('chamfer = {:.2f}, classify = {:.2f}, accu = {:.2f}, lr = {:.6f}'.format(val_loss1, val_loss2, accu, lr))
                  total = 0
                  correct = 0
                  

                  if val_loss1 < best_val:
                    print('saving')
                    torch.save(pointcloud, "ckpnts/save_KNN128_bs32_labels_"  + str(val_loss1) + '_' + ".pth")
                    best_val = val_loss1 - 0.3
    if epoch > 5:
      scheduler.step()
import numpy as np
import time
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import normalized_mutual_info_score as nmi, adjusted_rand_score as ari
from sklearn.cluster import KMeans
import random
import warnings
warnings.filterwarnings('ignore')

import tqdm
from sklearn.metrics import f1_score

import torch
import torch.nn as nn
import torch.nn.functional as F

# from torch_geometric.nn import GCNConv
# from torch_geometric.datasets import Planetoid, Reddit, Yelp
# from torch_geometric.data import Data

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)
def zscore(vec):
    m = torch.mean(vec)
    s = torch.std(vec)
    if s!=0:
        zs = (vec-m)/s
    else:
        zs = vec-m
    return zs
def skfr1(X,Class,classes,sparsity):
    features = X.shape[0]
    cases = X.shape[1]
    criteria = torch.zeros(features).to(device)

    for i in range(features):
        X[i,:] = zscore(X[i,:])

    switched = True
    iternum = 1
    loss_list = []
    while switched and iternum < 20:
        center = torch.zeros((features,classes)).to(device)
        members = torch.zeros(classes).to(device)
        for j in range(cases):
            i = Class[j]
            center[:,i] = center[:,i] + X[:,j]
            members[i] = members[i] + 1
        for j in range(classes):
            if members[j]>0:
                center[:,j] = center[:,j]/members[j]
        criteria = torch.matmul(torch.mul(center, center), members.T)
        index = torch.LongTensor([i for i in range(len(criteria))]).to(device)
        sorted_criteria = sorted(zip(criteria,index))
        J = [x[1] for x in sorted_criteria]
        J = torch.LongTensor(J).to(device)
        J = J[:features-sparsity]
        # importantvec = list(set(wholevec)-set(J))
        for i in range(len(J)):
            center[J[i]] = torch.zeros(classes).to(device)
        del members, criteria, index, sorted_criteria, J
        distance = torch.sqrt(((X.T - center.T[:, np.newaxis])**2).sum(axis=2))
        switched = False
        for i in range(cases):
            j = torch.argmin(distance[:,i])
            if j!=Class[i]:
                switched =True
                Class[i] = j
        del distance
        # WSStemp = torch.zeros(classes).to(device)
        # for k in range(classes):
        #     tempIndex = torch.LongTensor(np.where(Class.cpu().numpy()==k)[0]).to(device)
        #     tempX = torch.zeros((features,len(tempIndex))).to(device)
        #     for j in range(len(tempIndex)):
        #         tempX[:,j] = X[:,tempIndex[j]]
        #     WSStemp[k] = torch.mean(((tempX.T-center[:,k]).T)**2)
        # loss = torch.sum(WSStemp)
        # loss_list.append(loss)
        # print('Iteration : {}, Loss : {}'.format(iternum, loss))
        # del tempX, tempIndex, WSStemp, loss
        # iternum += 1
    return center, Class
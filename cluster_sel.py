import statistics
from collections import Counter
from collections import Counter
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
import torch
# 读取数据
from KCmean import FC_K_Means
from SKFR import skfr1
from transfor import transfor_data
import numpy as np
import pandas as pd
from transfor import transfor_data
from sklearn.cluster import KMeans
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import Birch
from sklearn.cluster import MiniBatchKMeans
from sklearn.cluster import BisectingKMeans
from sklearn.cluster import DBSCAN
from sklearn.cluster import SpectralClustering
from sklearn.cluster import SpectralCoclustering
from sklearn.cluster import kmeans_plusplus
from sklearn.cluster import linkage_tree
from sklearn.cluster import mean_shift
from KCmean import FC_K_Means
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
def cluster_sel(data,clumodel,num_class):
    data, n = transfor_data(data)
    X=data.iloc[:,0:-1]
    y=data.iloc[:,-1]
    if ( clumodel== 0):
        me = KMeans(n_clusters=num_class, n_init=10).fit(X)
    elif (clumodel==1):
        me = AgglomerativeClustering(n_clusters=num_class).fit(X)
    elif (clumodel== 2):
        X = np.array(data.iloc[:, 0:-1])
        fx = [X[1], X[2]]
        k_means = FC_K_Means(k=num_class, f=2, max_iter=1000, function_type='FC')
        k_means.fit(data=X, fix_centroid=fx)
        mc = k_means.label
    elif (clumodel == 3):
        me = MiniBatchKMeans(n_clusters=num_class,n_init=10).fit(X)
    elif (clumodel == 4):
        me = BisectingKMeans(n_clusters=num_class,n_init=10).fit(X)
    elif (clumodel == 5):
        me = KMeans(n_clusters=num_class, n_init=30).fit(X)
    elif (clumodel == 6):
        me = SpectralCoclustering(n_clusters=num_class,n_init=20).fit(X)
        mc = me.row_labels_
    elif (clumodel == 7):
        me = SpectralCoclustering(n_clusters=num_class).fit(X)
        mc=me.row_labels_
    elif (clumodel == 8):
        me =SpectralClustering(n_clusters=num_class).fit(X)
    elif (clumodel == 9):
        X = np.array(X)
        y1 = np.array(y)
        X = torch.FloatTensor(X)
        y1 = torch.LongTensor(y1.tolist()).to(device)
        X = X.to(device)
        y1 = y1.to(device)
        k = 4
        sparsity = 2
        centroids, cluster = skfr1(X, y1, k, sparsity)
        mc=cluster.cpu().tolist()
    if clumodel==2 or clumodel==6 or clumodel==7 or clumodel==9:
        list1=mc
    else:
        labels = me.labels_
        list1=labels
    list2=[]
    for i in y:
        list2.append(i)
    arr = np.stack([list1, list2], axis=1)
    counter=Counter(list1)
    # print(counter)
    l4 = []
    for j in range(num_class):
        j = num_class - 1-j
        # print(j)
        l = []
        ld = []
        l3 = []
        idx = np.where(arr[:, 1] == j)[0]
        for id in idx:
            ld.append(id)
        idxx = list(set(ld) - set(l4))
        idr = arr[idxx, 0]
        for ir in idr:
            l.append(ir)
        mode = statistics.mode(l)
        for k in ld:
            if arr[k, 0] == mode:
                l3.append(k)
                l4.append(k)
        arr[l3, 0] = j
    ls = list(set(l4))
    ly = []
    len1 = len(list1)
    for i in range(len1):
        ly.append(i)
    lyt = list(set(ly) - set(ls))
    return arr[:,0]
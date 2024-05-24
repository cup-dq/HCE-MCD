import numpy as np

from cluster_sel import cluster_sel
import pandas as pd
from transfor import transfor_data
def Cusdata(data):
   data, n = transfor_data(data)
   Data=pd.DataFrame([])
   y=data.iloc[:,-1]
   y = y.reset_index(drop=True)
   y = np.array(y)
   l1 = []
   for i in y:
      i = float(i)
      l1.append(i)
   n_classes = len(set(l1))
   for clu in range(10):
      ar=cluster_sel(data,clumodel=clu,num_class=n_classes)
      # print(ar)
      Data.loc[:,clu]=ar
   Data.loc[:,-1]=y
   # print(Data)
   return Data



import numpy as np
import pandas as pd
from Cusdata import Cusdata
from gadata2 import gadata2
from  transfor import transfor_data
class HCEMCD:
    def fit(self, X, y,fitness_,P,N,M,C):
        data = pd.DataFrame(X)
        data['class'] = y
        datac,n=transfor_data(data)
        data_c=Cusdata(data)
        data_ga,y_t=gadata2(data_c,Fitness=fitness_,Population_size=P,Num_generations=N,Mutation_rate=M,Crossover_rate=C)
        data_ga=np.array(data_ga)
        data_ga= data_ga.astype(np.int)
        y_g = np.apply_along_axis(lambda x: np.bincount(x).argmax() if np.any(x) else 0, axis=1, arr=data_ga)
        result = []
        for i in range(len(y_g)):
            if y_g[i] != y_t[i]:
                result.append(i)
        num_classes = len(np.unique(y_t))
        wrong_counts = np.zeros(num_classes, dtype=int)
        total_counts = np.zeros(num_classes, dtype=int)
        for i in range(num_classes):
            indices = np.where(y_t == i)[0]
            total_counts[i] = len(indices)
            wrong_counts[i] = np.sum(y_g[indices] != i)
        X=data.iloc[:,:-1]
        y=data.iloc[:,-1]
        X = pd.DataFrame(X)
        y = pd.DataFrame(y)
        lr=[]
        for i in range(len(y_g)):
            if y_t[i]==num_classes-1:
                lr.append(i)
        resultr=[]
        resultr=list(set(result)-set(lr))

        for i in resultr:
            X.drop(index=i, inplace=True)
            y.drop(index=i, inplace=True)

        return X,y
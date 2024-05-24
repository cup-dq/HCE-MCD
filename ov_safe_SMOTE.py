import random
import numpy as np
from scipy.spatial import distance_matrix


class ovSafe_SMOTE():

    def __init__(self,
                 proportion=1.0,
                 n_neighbors=5,
                 n_jobs=1,
                 random_state=None):
       
        self.proportion = proportion
        self.n_neighbors = n_neighbors
        self.n_jobs = n_jobs
        self.set_random_state(random_state)


    def set_random_state(self, random_state):

        self._random_state_init = random_state

        if random_state is None:
            self.random_state = np.random
        elif isinstance(random_state, int):
            self.random_state = np.random.RandomState(random_state)
        elif isinstance(random_state, np.random.RandomState):
            self.random_state = random_state
        elif random_state is np.random:
            self.random_state = random_state
        else:
            raise ValueError(
                "random state cannot be initialized by " + str(random_state))



    def sampling(self, ov_safeX, safeX, overSample_n, p_norm):
        appened=[]

        if overSample_n==0:
            return appened
        if len(safeX)==0:
            return appened

        n_neigh = min([len(safeX), self.n_neighbors])
        distances = distance_matrix(ov_safeX, safeX, p_norm)

        for i in range(len(ov_safeX)):
            X_base = ov_safeX[i]
            overSample_n=max(1,int(overSample_n/len(ov_safeX)))
            n_synthetic_samples=random.randint(0,overSample_n)
            ind = np.argsort(distances[i])[:n_neigh]

            for _ in range(n_synthetic_samples):
                X_neighbor = safeX[self.random_state.choice(ind)]
                sample=X_base + np.multiply(self.random_state.rand(1),X_neighbor - X_base)
                appened.append(sample)

        return appened





import numpy as np
from matplotlib import pyplot
from sklearn.cluster import KMeans


class K_Means(object):
    def __init__(self, k=2, max_iter=300, init='random', tol=0.0001, ):
        self.k = k
        self.max_iter = max_iter
        self.clf = {}
        self.centroid = []
        self.init = init
        self.label = []
        self.tol = tol

    def data_init(self, data):
        if self.init == 'random':
            self.data_init_random(data)
        else:
            self.data_init_kmeanpp(data)

    def data_init_random(self, data):
        lista = np.random.choice(np.arange(data.shape[0]), self.k, replace=False)
        self.centroid = data[lista]

    def data_init_kmeanpp(self, data):
        # select the first centroid
        lista = np.random.choice(np.arange(data.shape[0]), 1)
        # calculate distance from each point to the nearest centroid
        distance = np.zeros(data.shape[0])
        for i in range(1, self.k):
            for j, x in enumerate(data):
                # for each data point compute the distance with the nearest centroid
                distance[j] = np.min([np.linalg.norm(x - data[c]) for c in lista])
            # compute the probability according to the distance between each point and its corresponding nearest centroid
            # the point with largest distance weighted the most
            prob = distance ** 2 / np.sum(distance ** 2)
            # random choose a new centroid index according to the probability
            lista = np.append(lista, np.random.choice(data.shape[0], 1, p=prob))

        self.centroid = data[lista]

    def fit(self, data):
        self.label = np.zeros(data.shape[0], dtype=int)
        self.data_init(data)
        for i in range(self.max_iter):
            self.clf = {}
            for j in range(self.k):
                self.clf[j] = []

            for p in range(data.shape[0]):

                distance = []
                for centroid in self.centroid:
                    distance.append(np.linalg.norm(data[p] - centroid))

                clas = np.argmin(distance)
                self.clf[clas].append(data[p])
                self.label[p] = clas

            pre_centroid = np.array(self.centroid)
            for clas in list(self.clf.keys()):
                self.centroid[clas] = np.average(self.clf[clas], axis=0)
            if np.sum((pre_centroid - self.centroid) ** 2) <= self.tol:
                break


class FC_K_Means(object):
    def __init__(self, k=2, f=0, max_iter=300, function_type='FC', tol=0.0001):
        self.k = k
        self.max_iter = max_iter
        self.clf = {}
        self.centroid = []
        self.fix_centroid = []
        self.function_type = function_type
        self.f = f
        self.label = []
        self.tol = tol

    def fit(self, data, fix_centroid=None):
        if self.function_type == 'FC':
            self.FC_kmeans(data, fix_centroid)
        else:
            self.FC_kmeans2(data, fix_centroid)

    def data_init_random(self, data):
        lista = np.random.choice(np.arange(data.shape[0]), self.k, replace=False)
        self.centroid = data[lista]

    def data_init_kmeanpp(self, data):
        # select the first centroid
        lista = np.random.choice(np.arange(data.shape[0]), 1)
        # calculate distance from each point to the nearest centroid
        distance = np.zeros(data.shape[0])
        for i in range(1, self.k):
            for j, x in enumerate(data):
                # for each data point compute the distance with the nearest centroid
                distance[j] = np.min([np.linalg.norm(x - data[c]) for c in lista])
            # compute the probability according to the distance between each point and its corresponding nearest centroid
            # the point with largest distance weighted the most
            prob = distance ** 2 / np.sum(distance ** 2)
            # random choose a new centroid index according to the probability
            lista = np.append(lista, np.random.choice(data.shape[0], 1, p=prob))

        self.centroid = data[lista]

    def FC_kmeans(self, data, fix_centroid):
        # initialize k center by kmeans++
        self.label = np.zeros(data.shape[0], dtype=int)
        self.data_init_kmeanpp(data)
        self.fix_centroid = fix_centroid
        # perform a normal kmeans update in phase I
        for i in range(self.max_iter):
            self.clf = {}
            for j in range(self.k):
                self.clf[j] = []
            for p in data:

                distance = []
                for centroid in self.centroid:
                    distance.append(np.linalg.norm(p - centroid))

                clas = np.argmin(distance)
                self.clf[clas].append(p)

            pre_centroid = np.array(self.centroid)
            for clas in self.clf:
                self.centroid[clas] = np.average(self.clf[clas], axis=0)
            if np.sum((pre_centroid - self.centroid) ** 2) <= self.tol:
                break

        # phase II
        # calculate the mean distance from a non-fix center to all fix center
        distance = np.zeros(self.k)
        for i in range(self.k):
            distance[i] = np.average(
                [np.linalg.norm(self.centroid[i] - fix_centroid) for fix_centroid in self.fix_centroid])
        # delete f number of non_fix center which is closest to fix_center
        for i in range(self.f):
            self.centroid = np.delete(self.centroid, np.argmin(distance), 0)
            distance = np.delete(distance, np.argmin(distance))
        # merge
        self.centroid = np.concatenate((self.centroid, self.fix_centroid), axis=0)

        # compute the normal kmeans for nonfix center
        for i in range(self.max_iter):
            self.clf = {}
            for j in range(self.k):
                self.clf[j] = []
            for p in range(data.shape[0]):

                distance = []
                for centroid in self.centroid:
                    distance.append(np.linalg.norm(data[p] - centroid))

                clas = np.argmin(distance)
                self.clf[clas].append(data[p])
                self.label[p] = clas
            pre_centroid = np.array(self.centroid)
            for clas in list(self.clf.keys())[:(self.k - self.f)]:
                self.centroid[clas] = np.average(self.clf[clas], axis=0)
            if np.sum(np.abs(pre_centroid - self.centroid)) == 0:
                break

    def FC_kmeans2(self, data, fix_centroid):

        # phase I
        # select f number of fix centroid
        self.label = np.zeros(data.shape[0], dtype=int)
        self.fix_centroid = fix_centroid
        lista = np.array([], dtype=int)
        for i in self.fix_centroid:
            lista = np.append(lista, (np.where((data == i).all(1))[0][0]))
        # calculate distance from each point to the nearest centroid
        distance = np.zeros(data.shape[0])
        for i in range(self.f, self.k):
            for j, x in enumerate(data):
                # for each data point compute the distance with the nearest centroid
                distance[j] = np.min([np.linalg.norm(x - data[c]) for c in lista])
            # compute the probability according to the distance between each point and its corresponding nearest centroid
            # the point with largest distance weighted the most
            prob = distance ** 2 / np.sum(distance ** 2)
            # random choose a new centroid index according to the probability
            lista = np.append(lista, np.random.choice(data.shape[0], 1, p=prob))
        # merge
        lista = np.concatenate((lista[self.f:self.k], lista[:self.f]), axis=0)
        self.centroid = data[lista]

        # phase II
        # calculate the mean distance from a non-fix center to all fix cente

        # compute the normal kmeans for nonfix center
        for i in range(self.max_iter):
            self.clf = {}
            for j in range(self.k):
                self.clf[j] = []
            for p in range(data.shape[0]):

                distance = []
                for centroid in self.centroid:
                    distance.append(np.linalg.norm(data[p] - centroid))

                clas = np.argmin(distance)
                self.clf[clas].append(data[p])
                self.label[p] = clas
            pre_centroid = np.array(self.centroid)
            for clas in list(self.clf.keys())[:(self.k - self.f)]:
                self.centroid[clas] = np.average(self.clf[clas], axis=0)
            ##stop criteria check
            if np.sum((pre_centroid - self.centroid) ** 2) <= self.tol:
                break


if __name__ == '__main__':
    x = np.array([[1, 2], [1.5, 1.8], [5, 8], [8, 8], [1, 0.6], [9, 11],[10,10],[3,8]])
    fx=[x[3],x[5]]

    k_means = FC_K_Means(k=5,f=2,max_iter=1000,function_type='FC')
    k_means.fit(data=x,fix_centroid=fx)
    center=np.arange(k_means.centroid.shape[0])
    pyplot.scatter(x[:,0], x[:,1], c=k_means.label)
    pyplot.scatter(k_means.centroid[:(k_means.k-k_means.f),0], k_means.centroid[:(k_means.k-k_means.f),1], marker='*',c='r', s=150)
    pyplot.scatter(k_means.centroid[(k_means.k-k_means.f):,0], k_means.centroid[(k_means.k-k_means.f):,1], marker='*',c='b', s=150)
    pyplot.show()
    k_means1 = FC_K_Means(k=5,f=2,max_iter=1000,function_type='FC2')
    k_means1.fit(data=x,fix_centroid=fx)
    center=np.arange(k_means1.centroid.shape[0])

    pyplot.scatter(x[:,0], x[:,1], c=k_means1.label)
    pyplot.scatter(k_means1.centroid[:(k_means1.k-k_means1.f),0], k_means1.centroid[:(k_means1.k-k_means1.f),1], marker='*',c='r', s=150)
    pyplot.scatter(k_means1.centroid[(k_means1.k-k_means1.f):,0], k_means1.centroid[(k_means1.k-k_means1.f):,1], marker='*',c='b', s=150)
    pyplot.show()
    #print(cat)
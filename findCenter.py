import numpy as np

def distance(x, y, p_norm=2):
    return np.sum(np.abs(x - y) ** p_norm) ** (1 / p_norm)

class findCenter:
    def __init__(self,X,y,p_norm):
        self.X=X
        self.y=y
        self.p_norm=p_norm

    # 多类
    # 直接计算类均值为类中心
    def meanCenter(self):
        X=self.X
        y=self.y
        classes = np.unique(y)
        sizes = np.array([sum(y == c) for c in classes])
        indices = np.argsort(sizes)[::-1]
        classes = classes[indices]
        observations = {c: X[y == c] for c in classes}

        X,y= self._unpack_observations(observations)
        center=np.zeros((len(classes),X.shape[1]))
        u = np.zeros((len(classes), X.shape[0]))
        d = np.zeros((len(classes), X.shape[0]))
        p_norm = self.p_norm

        for j in range(X.shape[0]):
            for i in range(len(classes)):
                ''' the center of class i '''
                current_class = classes[i]
                center[i] = observations[current_class].mean(axis=0)
                d[i, j] = distance(center[i], X[j], p_norm=p_norm)

            for k in range(len(classes)):
                u[k,j]=1/((d[k,j]/d[:,j]).sum())


        center_dict=dict(zip(classes,center))

        return center_dict, u, d, X,y,classes



    @staticmethod
    def _unpack_observations(observations):
        unpacked_points = []
        unpacked_labels = []

        for cls in observations.keys():
            if len(observations[cls]) > 0:
                unpacked_points.append(observations[cls])
                unpacked_labels.append(np.tile([cls], len(observations[cls])))

        unpacked_points = np.concatenate(unpacked_points)
        unpacked_labels = np.concatenate(unpacked_labels)

        return unpacked_points, unpacked_labels



from sklearn import preprocessing
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn import tree


def classifier(train_data, test_data, algor):

    zscore = preprocessing.StandardScaler()
    x_train = zscore.fit_transform(train_data[:, 0:-1])
    x_test = zscore.fit_transform(test_data[:, 0:-1])
    y_train = train_data[:, -1]


    if algor == 'CART':

        clf = tree.DecisionTreeClassifier()
        clf.fit(x_train, y_train)
        y_pre = clf.predict(x_test)

    elif algor == 'KNN':
        knn = KNeighborsClassifier()
        knn.fit(x_train, y_train)
        y_pre = knn.predict(x_test)

    elif algor == 'RF':
        clf = RandomForestClassifier(n_estimators=10, random_state=0)
        clf.fit(x_train, y_train)
        y_pre = clf.predict(x_test)

    elif algor == 'MLP':
        mlp = MLPClassifier(hidden_layer_sizes=(100,))
        mlp.fit(x_train, y_train)
        y_pre = mlp.predict(x_test)

    else:
        raise Exception("没有"+str(algor)+"算法")


    return y_pre



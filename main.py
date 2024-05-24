import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from HCEMCD import HCEMCD
np.random.seed(42)
num_samples = 100
num_features = 10
X = np.random.rand(num_samples, num_features)
y = np.random.randint(0, 11, num_samples)
population_size = 100
num_generations = 20
mutation_rate = 0.5
crossover_rate = 0.1
kf = StratifiedKFold(n_splits=5, shuffle=True)
for train_index, test_index in kf.split(X, y):
    X_train, y_train = X[train_index], y[train_index]
    X_test, y_test = X[test_index], y[test_index]
    lclf = HCEMCD()
    X_resampled, y_resampled = lclf.fit(X, y,fitness_="f1",P=population_size,N=num_generations,M=mutation_rate,C=crossover_rate)
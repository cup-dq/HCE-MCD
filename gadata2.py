import pandas as pd
import numpy as np
from imbalanced_ensemble.metrics import geometric_mean_score

from sklearn.metrics import f1_score, roc_curve, auc, recall_score, precision_score
import random

from transfor import transfor_data
def gadata2(data,Fitness,Population_size,Num_generations,Mutation_rate,Crossover_rate):
    data, n = transfor_data(data)
    X = data.iloc[:, :-1]
    y_true = data.iloc[:, -1]
    X = X.to_numpy()
    y_true = y_true.to_numpy()
    num_classes = len(np.unique(y_true))
    def fitness_function(individual,fitness=Fitness):
        columns = [i for i in range(len(individual)) if individual[i]]
        X_subset = X[:, columns]
        X_subset=X_subset.astype(np.int)
        y_pred = np.apply_along_axis(lambda x: np.bincount(x).argmax() if np.any(x) else 0, axis=1, arr=X_subset)
        gm = geometric_mean_score(y_true, y_pred,average='macro')
        if gm==0:
            gm=0.0001
        f1=f1_score(y_true, y_pred, average='macro')
        fpr = dict()
        tpr = dict()
        roc_auc = dict()
        for i in range(num_classes):
            fpr[i], tpr[i], _ = roc_curve(y_true == i, y_pred)
            roc_auc[i] = auc(fpr[i], tpr[i])
        macro_auc = np.mean(list(roc_auc.values()))
        recall = recall_score(y_true, y_pred, average='macro')
        pre = precision_score(y_true, y_pred, average='macro')
        if fitness == "f1":
            fitness_value = f1
        elif fitness == "gm":
            fitness_value = gm
        elif fitness == "auc":
            fitness_value = macro_auc
        elif fitness == "recall":
            fitness_value = recall
        elif fitness == "pre":
            fitness_value = pre
        return fitness_value
    population_size = Population_size
    num_generations =Num_generations
    mutation_rate =Mutation_rate
    crossover_rate=Crossover_rate
    num_features = X.shape[1]
    def genetic_algorithm(population_size, num_generations,mutation_rate,crossover_rate):

        population = []
        def crossover(parent1, parent2,crossover_rate):
            #交叉
            offspring = []
            for gene1, gene2 in zip(parent1, parent2):
                if np.random.random() < crossover_rate:
                    offspring.append(gene1)
                else:
                    offspring.append(gene2)
            return offspring
        def mutation(individual, mutation_rate):
            mutated_individual = []
            for gene in individual:
                if np.random.random() < mutation_rate:
                    mutated_gene = 1 - gene
                else:
                    mutated_gene = gene
                mutated_individual.append(mutated_gene)
            return mutated_individual
        for i in range(population_size):
            individual = [np.random.randint(2) for _ in range(num_features)]
            population.append(individual)
        for i in range(num_generations):
            fitness_values = [fitness_function(individual) for individual in population]
            offspring_population = []
            for j in range(population_size):
                parent1_index = np.random.choice(range(population_size), size=1, p=fitness_values / np.sum(fitness_values))[0]
                parent1 = population[parent1_index]
                parent2_index = np.random.choice(range(population_size), size=1)[0]
                parent2 = population[parent2_index]
                offspring = []
                offspring = crossover(parent1,parent2,crossover_rate)  # 交叉
                offspring = mutation(offspring, mutation_rate)  # 变异
                offspring_population.append(offspring)#保存种群
            population = offspring_population#更新种群
        fitness_values = [fitness_function(individual) for individual in population]
        # Return best individual and fitness value
        best_fitness_index = np.argmax(fitness_values)
        best_individual = population[best_fitness_index]
        best_fitness_value = fitness_values[best_fitness_index]
        return best_individual, best_fitness_value
    best_individual, best_fitness_value = genetic_algorithm(population_size, num_generations, mutation_rate,crossover_rate)
    selected_columns = [i for i in range(len(best_individual)) if best_individual[i]]
    Cus = data.iloc[:, selected_columns]
    return Cus, y_true


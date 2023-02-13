import random
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold, cross_val_score
from sklearn.base import TransformerMixin, BaseEstimator


class GA_FeatureSelection (TransformerMixin, BaseEstimator):

    def __init__(self, fc=10, md=LogisticRegression(max_iter=20000), ps=50, ep=100, mp=10, cp=10):
        self.feature_list = []
        self.feature_count = fc
        self.total_feature_count = 0
        self.model = md
        self.pop_size = ps
        self.epochs = ep
        self.mutation_probability = mp
        self.crossover_probability = cp

    def fit(self, X, y=None):
        solutions = []
        self.feature_list = list(X.columns.values)
        if y is None:
            print("Error: y is NULL")
            return self.feature_list
        for x in range(self.pop_size):
            # generates a random pool of solutions
            arr = random.sample(self.feature_list, self.feature_count)  # random.choice returns duplicates
            solutions.append(arr)
        # evaluates the generated solutions
        evaluated_solutions = self.fitness_solutions(solutions, X, y)
        # sorts the solutions in decreasing order of fitness value
        evaluated_solutions = sorted(evaluated_solutions, key=lambda t: t[1], reverse=True)
        for x in range(self.epochs):
            # removes the scores from the solutions
            solutions = [x[0] for x in evaluated_solutions]
            # generates crossover solutions
            crossover_solutions = self.crossover(solutions)
            # evaluates them
            new_crossover_solutions = self.fitness_solutions(crossover_solutions, X, y)
            # adds them to the current population
            evaluated_solutions = evaluated_solutions + new_crossover_solutions
            # removes the scores from the solutions
            solutions = [x[0] for x in evaluated_solutions]
            # generates mutated solutions
            mutated_solutions = self.mutation(solutions)
            # evaluates them
            mutated_evaluated_solutions = self.fitness_solutions(mutated_solutions, X, y)
            # adds them to the current population
            evaluated_solutions = evaluated_solutions + mutated_evaluated_solutions
            # sorts the solutions in decreasing order of fitness value
            evaluated_solutions = sorted(evaluated_solutions, key=lambda t: t[1], reverse=True)
            # keeps only the pop_size most fit solutions
            evaluated_solutions = evaluated_solutions[: self.pop_size]
        # returns the most fit solution
        self.feature_list = evaluated_solutions[0][0]

    def transform(self, X):
        # cuts the dataset to the relevant features
        X = X.filter(self.feature_list)
        return X

    def fit_transform(self, X, y=None, **fit_params):
        self.fit(X, y)
        X = self.transform(X)
        return X

    def evaluation(self, X, Y, model, solution):
        df = X.copy()
        # discards every feature from the dataset excluding the ones present in the current solution
        df = df.filter(solution)
        # adds the target vector as a column in the dataset
        df['Target'] = Y
        # removes any instance with NaN values
        df.dropna(inplace=True)
        Y = df.Target.to_list()
        # removes the target column from the dataset
        df.drop(columns=['Target'], inplace=True)
        # calculates the fitness scores of the solution using a 5-Fold Cross Validation
        cv = KFold(n_splits=5, random_state=1, shuffle=True)
        fitness_scores = cross_val_score(model, df, Y, scoring='f1_weighted', cv=cv)
        np_scores = np.array(fitness_scores)
        # we calculate the mean of the fitness scores of the folds as our final result
        fitness_value = np.mean(np_scores)
        return fitness_value

    def mutation(self, solutions):
        np_solutions = np.array(solutions)
        random_solutions = np_solutions[np.random.choice(np_solutions.shape[0], round((self.mutation_probability*self.pop_size)/100), replace=False), :]
        new_solutions = []
        for x in range(random_solutions.shape[0]):
            # selects a random feature to include
            new_feature = random.choice(self.feature_list)
            # if already present in the solution, I replace it
            while new_feature in random_solutions[x]:
                new_feature = random.choice(self.feature_list)
            # chooses a random feature to replace
            new_int = random.randint(0, (self.feature_count - 1))
            mutated_solution = random_solutions[x]
            mutated_solution[new_int] = new_feature
            # adds to the list of new mutated solutions
            new_solutions.append(mutated_solution.tolist())
        return new_solutions

    def crossover(self, solutions):
        np_solutions = np.array(solutions)
        n_solutions = round((self.crossover_probability * self.pop_size) / 100)
        # I remove the last solution if the number of solutions is odd
        if (n_solutions % 2) == 1:
            n_solutions = n_solutions - 1
        random_solutions = np_solutions[np.random.choice(np_solutions.shape[0], n_solutions, replace=False), :]
        new_solutions = []

        for x in range(0, random_solutions.shape[0], 2):
            sol1_flag = True
            sol2_flag = True
            new_int = random.randint(1, (self.feature_count - 1))
            part_size = self.feature_count - new_int

            cross1_part1 = random_solutions[x][:new_int]
            cross1_part2 = random_solutions[x+1][-part_size:]
            # I check that there are no duplicate features in the 2 parts
            for i in range(part_size):
                if cross1_part2[i] in cross1_part1:
                    # If I found one, I try to replace it with a feature from the first solution in the same index
                    if not random_solutions[x][i+new_int] in cross1_part2:
                        cross1_part2[i] = random_solutions[x][i + new_int]
                    else:
                        # If that doesn't work, I try to replace it with a feature from the first solution
                        # in another index, starting from the index of the crossover point
                        found = False
                        for j in range(new_int, self.feature_count):
                            if not random_solutions[x][j] in cross1_part2:
                                cross1_part2[i] = random_solutions[x][j]
                                found = True
                                break
                        if not found:
                            # If I can't find any feature to replace, I discard the whole crossover solution
                            sol1_flag = False
            cross_solution1 = np.concatenate((cross1_part1, cross1_part2))

            cross2_part1 = random_solutions[x+1][:new_int]
            cross2_part2 = random_solutions[x][-part_size:]
            # I check that there are no duplicate features in the 2 parts
            for i in range(part_size):
                if cross2_part2[i] in cross2_part1:
                    # If I found one, I try to replace it with a feature from the second solution in the same index
                    if not random_solutions[x+1][i + new_int] in cross2_part2:
                        cross2_part2[i] = random_solutions[x+1][i + new_int]
                    else:
                        # If that doesn't work, I try to replace it with a feature from the second solution
                        # in another index, starting from the index of the crossover point
                        found = False
                        for j in range(new_int, self.feature_count):
                            if not random_solutions[x+1][j] in cross2_part2:
                                cross2_part2[i] = random_solutions[x+1][j]
                                found = True
                                break
                        if not found:
                            # If I can't find any feature to replace, I discard the whole crossover solution
                            sol2_flag = False
            cross_solution2 = np.concatenate((cross2_part1, cross2_part2))

            if sol1_flag:
                new_solutions.append(cross_solution1.tolist())

            if sol2_flag:
                new_solutions.append(cross_solution2.tolist())

        return new_solutions

    def fitness_solutions(self, solutions, X, Y):
        fitness_scores = []
        # evaluates each solution
        for x in range(len(solutions)):
            fitness_value = self.evaluation(X, Y, self.model, solutions[x])
            fitness_scores.append(fitness_value)
        # associates every solution with its fitness score
        evaluated_solutions = list(zip(solutions, fitness_scores))
        return evaluated_solutions

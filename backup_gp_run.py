from gplearn_1230.genetic import SymbolicRegressor
from gplearn_1230.functions import _Function

import time
import os
import shutil
import random
import time
import datetime
import multiprocessing as mp
import numpy as np
import matplotlib.pyplot as plt

def main():

    # Nguyen-1
    X_train = np.arange(-1, 1, 0.01).reshape(200, 1)
    y_train = X_train**3 + X_train**2 + X_train  
    y_train = y_train.ravel()

    est_gp = SymbolicRegressor(population_size=100,
                            generations=5, stopping_criteria=1e-5,
                            p_crossover=0.7, p_subtree_mutation=0.1,
                            p_hoist_mutation=0.05, p_point_mutation=0.1,
                            max_samples=0.9, verbose=1,
                            parsimony_coefficient=0.01, random_state=0
                            # random_state=random.seed(datetime.datetime.now().timestamp())
                            )
                            
    est_gp.fit(X_train, y_train)
    result = est_gp
    # print(f'result: {result}')
    best_fitness = min(result.run_details_['best_fitness'])
    print(f'best fitness: {best_fitness}')
    # best_fitness = min(result.run_details_['best_fitness'])

    return result

if __name__ == '__main__':
    main()
    
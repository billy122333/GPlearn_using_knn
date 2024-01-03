from gplearn_1230.genetic import SymbolicRegressor
from gplearn_1230.functions import _Function

import random
import time
import datetime
import multiprocessing as mp
import numpy as np
import sys

def main():

    # Nguyen-1
    X_train = np.arange(-1, 1, 0.01).reshape(200, 1)
    if len(sys.argv) < 2:
        print('Usage: python backup_gp_run.py <X_train> <index>')
        y_train = X_train**3 + X_train**2 + X_train  
        index = 1
    # y_train = 6.87 + 11*math.cos(7.23*X_train)
    # y_train = math.sqrt(X_train)
    else:
        y_train = eval(sys.argv[1])
        index = int(sys.argv[2])
    y_train = y_train.ravel()

    est_gp = SymbolicRegressor(population_size=100,
                            generations=20, stopping_criteria=1e-5,
                            p_crossover=0.8, p_subtree_mutation=0.0,
                            p_hoist_mutation=0.0, p_point_mutation=0.0,
                            max_samples=0.9, verbose=1,
                            parsimony_coefficient=0.01,
                            random_state=random.seed(datetime.datetime.now().timestamp())
                            )
                            
    est_gp.fit(X_train, y_train, index)
    result = est_gp
    # print(f'result: {result}')
    best_fitness = min(result.run_details_['best_fitness'])
    print(f'best fitness: {best_fitness}')
    # best_fitness = min(result.run_details_['best_fitness'])

    return result

if __name__ == '__main__':
    main()
    
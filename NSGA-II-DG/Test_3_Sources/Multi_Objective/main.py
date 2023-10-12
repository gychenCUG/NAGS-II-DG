import warnings
import geatpy as ea  # import geatpy
from MyProblem import MyProblem
from MyProblem import normalize
import pandas as pd
from moea_NSGA2_log import moea_NSGA2_templet
import numpy as np

if __name__ == '__main__':
    warnings.filterwarnings("ignore")

    Encoding = "RI"
    # Instantiate the problem object
    problem = MyProblem()
    Field = ea.crtfld(Encoding,problem.varTypes, problem.ranges, problem.borders)  # Create a region descriptor
    # Building algorithms
    algorithm = moea_NSGA2_templet(
        problem,
        ea.Population(Encoding='RI',Field=Field,NIND=100),
        MAXGEN=5, # Maximum generations
        logTras=1, # Set how many generations to record logs. If set to 0, it means no logs will be recorded.
        verbose=True, # Set whether to print out log information
        drawing=1 # Set the drawing mode (0: no drawing; 1: draw the result graph; 2: draw the target space process animation; 3: draw the decision space process animation)
    )

    [NDSet, pop] = algorithm.run() # Execute the algorithm template to obtain the Pareto optimal individual and the last generation population
    NDSet.save(dirName=f'NDSet')  # Save the best individual information to a file
    pop.save(dirName=f'pop')
    """==================================Output results=============================="""
    print('number of evaluations：%s' % algorithm.evalsNum)
    print('time has passed %s s' % algorithm.passTime)

    # NDSet matrix
    NDSet_X = NDSet.Phen

    NDSet_X = normalize(NDSet_X)

    # NDSet objective function matrix
    NDSet_Y = NDSet.ObjV

    # save (NDSet_X,NDSet_Y)
    NDSet_X = pd.DataFrame(NDSet_X)
    NDSet_Y = pd.DataFrame(NDSet_Y)
    NDSet_Y.columns = ['r2', 'v', 'd']
    NDSet_X.columns = ['source1', 'source2', 'source3']
    NDSet_X = pd.concat([NDSet_X, NDSet_Y], axis=1)

    NDSet_X.to_csv(f'./results_NDset.csv', index=False)

    # population matrix
    X = pop.Phen

    # Normalized
    X = normalize(X)

    # population objective function matrix
    Y = pop.ObjV

    # save (X,Y)
    X = pd.DataFrame(X)
    Y = pd.DataFrame(Y)
    Y.columns = ['r2','v','d']
    X.columns = ['source1', 'source2', 'source3']
    X = pd.concat([X, Y], axis=1)

    X.to_csv(f'./results_pop.csv', index=False)
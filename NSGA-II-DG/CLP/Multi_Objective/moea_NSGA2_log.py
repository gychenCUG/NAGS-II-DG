# -*- coding: utf-8 -*-
import numpy as np
import geatpy as ea
import time
import csv

class moea_NSGA2_templet(ea.MoeaAlgorithm):
    """
moea_NSGA2_templet : class - Multi-objective evolutionary NSGA-II algorithm class

Algorithm Description:
     NSGA-II is used for multi-objective optimization. For details of the algorithm, see reference [1].

Reference:
    [1] Deb K , Pratap A , Agarwal S , et al. A fast and elitist multiobjective 
    genetic algorithm: NSGA-II_trial_1[J]. IEEE Transactions on Evolutionary
    Computation, 2002, 6(2):0-197.

    """

    def __init__(self,
                 problem,
                 population,
                 MAXGEN=None,
                 MAXTIME=None,
                 MAXEVALS=None,
                 MAXSIZE=None,
                 logTras=None,
                 verbose=None,
                 outFunc=None,
                 drawing=None,
                 dirName=None,
                 **kwargs):
        # First call the parent class constructor
        super().__init__(problem, population, MAXGEN, MAXTIME, MAXEVALS, MAXSIZE, logTras, verbose, outFunc, drawing,
                         dirName)
        if population.ChromNum != 1:
            raise RuntimeError('传入的种群对象必须是单染色体的种群类型。')
        self.name = 'NSGA2'
        if self.problem.M < 10:
            self.ndSort = ea.ndsortESS  # Using ENS_SS for non-dominated sorting
        else:
            self.ndSort = ea.ndsortTNS  # High-dimensional targets use T_ENS for non-dominated sorting, which is generally faster than ENS_SS.
        self.selFunc = 'etour'  # Selection operator, using elite tournament selection
        if population.Encoding == 'P':
            self.recOper = ea.Xovpmx(XOVR=1)  # Generate partial matching crossover operator objects
            self.mutOper = ea.Mutinv(Pm=1)  # Generate reverse mutation operator object
        elif population.Encoding == 'BG':
            self.recOper = ea.Xovud(XOVR=1)  # Generate uniform crossover operator objects
            self.mutOper = ea.Mutbin(Pm=None)  # Generate a binary mutation operator object. When Pm is set to None, the specific value takes the default value of Pm in the mutation operator.
        elif population.Encoding == 'RI':
            self.recOper = ea.Recsbx(XOVR=1, n=20)  # Generate simulated binary crossover operator objects
            self.mutOper = ea.Mutpolyn(Pm=1 / self.problem.Dim, DisI=20)  # Generate polynomial mutation operator objects
        else:
            raise RuntimeError('编码方式必须为''BG''、''RI''或''P''.')

    def logging(self, pop):
        """
        describe:
             Used to record logs during the evolution process. This function is called inside the stat() function.
                 If you need to record other data in the log, you need to rewrite this function in the custom algorithm class.

             Input parameters:
                 pop : class <Population> - Population object.

             Output parameters:
                 No output parameters.
        """

        self.passTime += time.time() - self.timeSlot  # Update time record, do not calculate logging time
        if len(self.log['gen']) == 0:  # Initialize each key value of the log
            self.log['gd'] = []
            self.log['igd'] = []
            self.log['hv'] = []
            self.log['spacing'] = []

            self.log['f1_mean'] = []
            self.log['f2_mean'] = []
            self.log['f3_mean'] = []

        self.log['gen'].append(self.currentGen)
        self.log['eval'].append(self.evalsNum)  # Record the number of reviews
        [levels, _] = ea.ndsortDED(pop.ObjV, needLevel=1, CV=pop.CV, maxormins=self.problem.maxormins)  # nondominated stratification
        NDSet = pop[np.where(levels == 1)[0]]  # Only retain non-dominated individuals in the population to form a non-dominated population

        if self.problem.ReferObjV is not None:
            self.log['gd'].append(ea.indicator.GD(NDSet.ObjV, self.problem.ReferObjV))  # Calculate GD metric
            self.log['igd'].append(ea.indicator.IGD(NDSet.ObjV, self.problem.ReferObjV))  # Calculate IGD metric
            self.log['hv'].append(ea.indicator.HV(NDSet.ObjV, self.problem.ReferObjV))  # Calculate HV metric
        else:
            self.log['gd'].append(None)
            self.log['igd'].append(None)
            self.log['hv'].append(ea.indicator.HV(NDSet.ObjV))  # Calculate HV metric
        self.log['spacing'].append(ea.indicator.Spacing(NDSet.ObjV))  # Calculate Spacing metric
        self.log['f1_mean'].append(np.mean(pop.ObjV[:,0]))
        self.log['f2_mean'].append(np.mean(pop.ObjV[:,1]))
        self.log['f3_mean'].append(np.mean(pop.ObjV[:,2]))

        self.timeSlot = time.time()  # Update timestamp

        if (len(self.log['gen']) == 50):
            log_file = 'log.csv'
            # Save logs as CSV file
            with open(log_file, 'w', newline='') as file:
                writer = csv.writer(file)

                # Write header
                writer.writerow(['Generation', 'Evaluation', 'GD', 'IGD', 'HV', 'Spacing','F1_mean','F2_mean','F3_mean'])

                # Write log data for each generation
                for i in range(len(self.log['gen'])):
                    generation = self.log['gen'][i]
                    evaluation = self.log['eval'][i]
                    gd = self.log['gd'][i]
                    igd = self.log['igd'][i]
                    hv = self.log['hv'][i]
                    spacing = self.log['spacing'][i]
                    f1_mean = self.log['f1_mean'][i]
                    f2_mean = self.log['f2_mean'][i]
                    f3_mean = self.log['f3_mean'][i]

                    writer.writerow([generation, evaluation, gd, igd, hv, spacing,f1_mean,f2_mean,f3_mean])

    def draw(self, pop, EndFlag=False):

        """
        describe:
             This function is used to plot during evolution. This function is called in stat() and finishing functions.

        Input parameters:
             pop : class <Population> - Population object.

             EndFlag: bool - Indicates whether this function is called for the last time.

        Output parameters:
             No output parameters.

        """

        if not EndFlag:
            self.passTime += time.time() - self.timeSlot  # Update the time record and do not calculate the time spent on drawing.
            # Draw animation
            if self.drawing == 2:
                # Draw a dynamic map of the target space
                if pop.ObjV.shape[1] == 2 or pop.ObjV.shape[1] == 3:
                    if self.plotter is None:
                        self.plotter = ea.PointScatter(self.problem.M, grid=True, legend=True,
                                                       title='Pareto Front Plot')
                    self.plotter.refresh()
                    self.plotter.add(pop.ObjV, color='red', label='MOEA PF at ' + str(self.currentGen) + ' Generation')
                else:
                    if self.plotter is None:
                        self.plotter = ea.ParCoordPlotter(self.problem.M, grid=True, legend=True,
                                                          title='Parallel Coordinate Plot')
                    self.plotter.refresh()
                    self.plotter.add(pop.ObjV, color='red',
                                     label='MOEA Objective Value at ' + str(self.currentGen) + ' Generation')
                self.plotter.draw()
            elif self.drawing == 3:
                # Draw a dynamic diagram of the decision space
                if self.plotter is None:
                    self.plotter = ea.ParCoordPlotter(self.problem.Dim, grid=True, legend=True,
                                                      title='Variables Value Plot')
                self.plotter.refresh()
                self.plotter.add(pop.Phen, marker='o', color='blue',
                                 label='Variables Value at ' + str(self.currentGen) + ' Generation')
                self.plotter.draw()
            self.timeSlot = time.time()  # Update timestamp
        else:
            # Plot the final result
            if self.drawing != 0:
                if self.plotter is not None:  # If animation is drawn, save and close the animation
                    self.plotter.createAnimation()
                    self.plotter.close()
                if pop.ObjV.shape[1] == 2 or pop.ObjV.shape[1] == 3:
                    figureName = 'Pareto Front Plot'
                    self.plotter = ea.PointScatter(self.problem.M, grid=True, legend=True, title=figureName,
                                                   saveName=self.dirName + figureName)
                    self.plotter.add(self.problem.ReferObjV, color='gray', alpha=0.1, label='Monte Optimal Set')
                    self.plotter.add(pop.ObjV, color='red', label='MOEA PF')
                    self.plotter.draw()
                else:
                    figureName = 'Parallel Coordinate Plot'
                    self.plotter = ea.ParCoordPlotter(self.problem.M, grid=True, legend=True, title=figureName,
                                                      saveName=self.dirName + figureName)
                    self.plotter.add(self.problem.TinyReferObjV, color='gray', alpha=0.5, label='True Objective Value')
                    self.plotter.add(pop.ObjV, color='red', label='MOEA Objective Value')
                    self.plotter.draw()

    def reinsertion(self, population, offspring, NUM):

        """
        describe:
             Re-insert individuals to generate a new generation of populations (using the strategy of parent-child combined selection).
             NUM is the number of individuals that need to be retained to the next generation.
             Note: Here is an equivalent modification to the original NSGA-II: first calculate the fitness of individual populations based on Pareto classification and crowding distance.
             Then call the dup selection operator (see help(ea.dup) for details) to select individuals in descending order of fitness and retain them to the next generation.
             This is exactly the same result as the original NSGA-II selection method.

        """

        # Merger of father and offspring generations
        population = population + offspring
        # Select individuals to retain to the next generation
        [levels, _] = self.ndSort(population.ObjV, NUM, None, population.CV, self.problem.maxormins)  # Perform non-dominated stratification on NUM individuals
        dis = ea.crowdis(population.ObjV, levels)  # Calculate crowding distance
        population.FitnV[:, 0] = np.argsort(np.lexsort(np.array([dis, -levels])), kind='mergesort')  # Calculate fitness
        chooseFlag = ea.selecting('dup', population.FitnV, NUM)  # Call the low-level selection operator dup to perform selection based on fitness sorting, retaining NUM individuals
        return population[chooseFlag]

    def run(self, prophetPop=None):  # prophetPop is the prophet population (that is, the population containing prior knowledge)
        # ==========================Initial configuration===========================
        population = self.population
        NIND = population.sizes
        self.initialization()  # Initialize some dynamic parameters of the algorithm class
        # ===========================Ready to evolve============================
        population.initChrom()  # Initialize the population chromosome matrix
        # Insert prior knowledge (note: the legitimacy of the prophet population prophetPop will not be checked here)
        if prophetPop is not None:
            population = (prophetPop + population)[:NIND]  # Insert prophet population
        self.call_aimFunc(population)  # Calculate the objective function value of the population
        [levels, _] = self.ndSort(population.ObjV, NIND, None, population.CV, self.problem.maxormins)  # Non-dominated stratification of NIND individuals
        population.FitnV = (1 / levels).reshape(-1, 1)  # Calculate the fitness of the first-generation individuals directly based on levels
        # ===========================Start evolving============================
        while not self.terminated(population):
            # 选择个体参与进化
            offspring = population[ea.selecting(self.selFunc, population.FitnV, NIND)]
            # Select individuals to participate in evolution
            offspring.Chrom = self.recOper.do(offspring.Chrom)  # Reorganization
            # if self.currentGen > self.MAXGEN * 0.5:
            #     offspring.Chrom = ea.mutmani(offspring.Encoding, offspring.Chrom, offspring.Field, self.problem.M-1)
            offspring.Chrom = self.mutOper.do(offspring.Encoding, offspring.Chrom, offspring.Field)  # Mutation
            self.call_aimFunc(offspring)  # Find the objective function value of the evolved individual
            population = self.reinsertion(population, offspring, NIND)  # Re-insertion generates a new generation population
        return self.finishing(population)  # Call finishing to complete subsequent work and return the results

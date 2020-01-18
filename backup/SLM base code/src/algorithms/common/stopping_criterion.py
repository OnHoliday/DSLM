from numpy import std, abs

from algorithms.common.metric import is_better


class StoppingCriterion():

    def __init__(self, maximum_iterations=500):
        self.maximum_iterations = maximum_iterations
    
    def evaluate(self, algorithm):
        return self.evaluate_final(algorithm)
    
    def evaluate_final(self, algorithm):
        """Makes sure that all algorithms terminate."""
        return algorithm.current_generation >= self.maximum_iterations
    
    def __repr__(self):
        return 'GenericStoppingCriterion' + '_' + str(self.maximum_iterations)


class MaxGenerationsCriterion(StoppingCriterion):
    """Stops evolutionary process, if current generation of algorithms is greater than max generation."""

    def __init__(self, max_generation):
        self.max_generation = 11 #max_generation

    def evaluate(self, algorithm):
        return algorithm.current_generation >= self.max_generation
    
    def __repr__(self):
        return 'MaxGen' + '_' + str(self.max_generation)


class ErrorDeviationVariationCriterion(StoppingCriterion):
    """Stops evolutionary process, if the share of solutions with lower error deviation variation amongst the
    superior offspring is less than defined threshold."""

    def __init__(self, threshold=0.25, maximum_iterations=500):
        super().__init__(maximum_iterations)
        self.threshold = threshold

    def evaluate(self, algorithm):

        # If current generation is 0, return False (since there exists no champion).
        if algorithm.current_generation == 0:
            return False
        champion = algorithm.champion
        # Subsets offspring that are better than ancestor.
        superior_solutions = [solution for solution in algorithm.population if solution.better_than_ancestor]
        # If not superior offspring exist, determine parent stopping criterion.
        if not superior_solutions:
            return super().evaluate(algorithm)
        # Calculate error SD for champion.
        var_champion = std(abs(champion.predictions - algorithm.target_vector))
        # Calculate error SD for offspring.
        var_superior_solutions = [std(abs(superior_solution.predictions - algorithm.target_vector)) for superior_solution in superior_solutions]
        # Subsets offspring that have a lower error deviation variation.
        lower_var = [var for var in var_superior_solutions if var < var_champion]
        # Calculates percentage.
        percentage_lower = len(lower_var) / len(var_superior_solutions)
        # If percentage lower than threshold, return True, else evaluate parent.
        if percentage_lower < self.threshold:
            return True
        else:
            return super().evaluate(algorithm)
    
    def __repr__(self):
        return 'EDV' + '_threshold_' + str(self.threshold)


class TrainingImprovementEffectivenessCriterion(StoppingCriterion): 
    """ stops evolutionary process if the mutation effectiveness (ie percentage of solutions better than the champion) drops 
    to a value lower than a certain threshold"""

    def __init__(self, threshold=0.25, maximum_iterations=500):
        super().__init__(maximum_iterations)
        self.threshold = threshold
    
    def evaluate(self, algorithm):
    
        # if current generation is 0, return False (since there exists no champion)
        if algorithm.current_generation == 0:
            return False 
        champion = algorithm.champion
        # subsets offspring that are better than the current champion 
        superior_solutions = [solution for solution in algorithm.population if is_better(solution.value, champion.value, algorithm.metric)]
        # if not superior offspring exist, determine parent stopping criterion
        if not superior_solutions: 
            return super().evaluate(algorithm)
        # calculates nr of superior solutions 
        # nr_superior_solutions = len(superior_solutions) 
        # calculate percentage of superior solutions 
        percentage_superior_solutions = len(superior_solutions) / len(algorithm.population)
        if percentage_superior_solutions < self.threshold: 
            return True
        else: 
            return super().evaluate(algorithm)

    def __repr__(self):
        return 'TIE' + '_threshold_' + str(self.threshold)

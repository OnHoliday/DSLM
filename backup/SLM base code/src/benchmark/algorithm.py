from timeit import default_timer

from numpy.random import uniform

from algorithms.common.ensemble import Ensemble
from algorithms.common.metric import RootMeanSquaredError, WeightedRootMeanSquaredError, Accuracy
from algorithms.neat_python.algorithm import Neat
from algorithms.semantic_learning_machine.algorithm import SemanticLearningMachine
from algorithms.semantic_learning_machine.algorithm_cnn import DeepSemanticLearningMachine
from algorithms.simple_genetic_algorithm.algorithm import SimpleGeneticAlgorithm
from data.extract import generate_sub_training_set, generate_training_target, is_classification_target
import numpy as np 
from utils.useful_methods import generate_random_weight_vector

_time_seconds = lambda: default_timer()


def _get_parent(algorithm):
    return algorithm.__class__.__bases__[0]


def _benchmark_fit(algorithm, input_matrix, target_vector, metric, verbose):
    parent = _get_parent(algorithm) 
    parent.fit(algorithm, input_matrix, target_vector, metric, verbose)
    return algorithm.log


def _benchmark_run(algorithm, verbose=False):
    parent = _get_parent(algorithm)
    time_log = list()
    solution_log = list()
    stopping_criterion = False
    
    while (not stopping_criterion):
        start_time = _time_seconds()
        stopping_criterion = parent._epoch(algorithm)
        end_time = _time_seconds()
        # print("generation time: ", end_time-start_time)
        time_log.append(end_time - start_time)
        solution_log.append(algorithm.champion)
        
        #auroc = Accuracy.evaluate(algorithm.champion.predictions, algorithm.target_vector)
        #print('\t\t\t\tChampion at iteration', algorithm.current_generation, ', AUROC', auroc, ', RMSE', algorithm.champion.value)
        
        if algorithm.current_generation % 10 == 0:
            print('\t\t\t\tChampion at iteration', algorithm.current_generation, ', RMSE', algorithm.champion.value)
        
        if verbose:
            parent._print_generation(algorithm)
        algorithm.current_generation += 1
    
    algorithm.log = {
        'time_log': time_log,
        'solution_log': solution_log
    }
    
    #===========================================================================
    # if is_classification_target(algorithm.target_vector): 
    #     algorithm.champion.accuracy = Accuracy.evaluate(algorithm.champion.predictions, algorithm.target_vector)  
    #===========================================================================


def _benchmark_run_rst(algorithm, verbose):
    """if random_sampling_technique is set to true then the training instances change at each iteration"""
    parent = _get_parent(algorithm)
    time_log = list()
    solution_log = list()
    stopping_criterion = False
    original_input_matrix = algorithm.input_matrix
    original_target_vector = algorithm.target_vector
    original_metric = algorithm.metric
    algo = algorithm.metric.__class__.__name__
    size = int(original_input_matrix.shape[0] * algorithm.subset_ratio)
    
    if algorithm.metric.__class__.__name__ == 'WeightedRootMeanSquaredError': 
        original_weight_vector = algorithm.metric.weight_vector
        idx = np.random.choice(np.arange(original_input_matrix.shape[0]), size, replace=False)
        algorithm.metric.weight_vector = original_weight_vector[idx]
    
    while (not stopping_criterion):
        start_time = _time_seconds() 
        idx = np.random.choice(np.arange(original_input_matrix.shape[0]), size, replace=False)
        algorithm.input_matrix = original_input_matrix[idx]
        algorithm.target_vector = original_target_vector[idx]
        stopping_criterion = parent._epoch(algorithm)
        end_time = _time_seconds()
        # print("generation time: ", end_time-start_time)
        time_log.append(end_time - start_time)
        solution_log.append(algorithm.champion)
        if verbose:
            parent._print_generation(algorithm)
        algorithm.current_generation += 1
    
    algorithm.log = {
        'time_log': time_log,
        'solution_log': solution_log
    }
    
    if algorithm.metric.__class__.__name__ == 'WeightedRootMeanSquaredError':
        algorithm.metric.weight_vector = original_weight_vector
    
    algorithm.champion.predictions = algorithm.champion.neural_network.predict(original_input_matrix)
    algorithm.champion.value = original_metric.evaluate(algorithm.champion.predictions, original_target_vector)


def _benchmark_run_rwt(algorithm, verbose):
    """if random_weighting_technique is set to true then the weights change at each iteration """
    parent = _get_parent(algorithm)
    time_log = list()
    solution_log = list()
    stopping_criterion = False
    original_metric = algorithm.metric 
    while (not stopping_criterion):
        start_time = _time_seconds() 
        algorithm.metric = WeightedRootMeanSquaredError(uniform(0, algorithm.weight_range, algorithm.input_matrix.shape[0]))
        stopping_criterion = parent._epoch(algorithm)
        end_time = _time_seconds()
        # print("generation time: ", end_time-start_time)
        time_log.append(end_time - start_time)
        solution_log.append(algorithm.champion)
        if verbose:
            parent._print_generation(algorithm)
        algorithm.current_generation += 1
    
    algorithm.log = {
        'time_log': time_log,
        'solution_log': solution_log
    }
    
    algorithm.champion.predictions = algorithm.champion.neural_network.predict(algorithm.input_matrix)
    algorithm.champion.value = original_metric.evaluate(algorithm.champion.predictions, algorithm.target_vector)

class BenchmarkDSLM(DeepSemanticLearningMachine):

    fit = _benchmark_fit
    _run = _benchmark_run

class BenchmarkSLM(SemanticLearningMachine):

    fit = _benchmark_fit
    _run = _benchmark_run


class BenchmarkNEAT(Neat):

    fit = _benchmark_fit
    _run = _benchmark_run


class BenchmarkSGA(SimpleGeneticAlgorithm):

    fit = _benchmark_fit
    _run = _benchmark_run


class BenchmarkSLM_RST(SemanticLearningMachine):
    
    fit = _benchmark_fit
    _run = _benchmark_run_rst 

    def __repr__(self):
        return 'BenchmarkSLM_RST'


class BenchmarkSLM_RWT(SemanticLearningMachine):
    
    fit = _benchmark_fit
    _run = _benchmark_run_rwt 

    def __repr__(self):
        return 'BenchmarkSLM_RWT'

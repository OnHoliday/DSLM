from random import shuffle
from timeit import default_timer
import warnings

from neat.nn.feed_forward import FeedForwardNetwork
from numpy import append, array
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.svm import SVC, SVR
from tqdm import tqdm

from algorithms.common.ensemble import Ensemble, EnsembleBagging, EnsembleRandomIndependentWeighting, EnsembleBoosting
from algorithms.common.metric import is_better, Accuracy
from algorithms.semantic_learning_machine.algorithm import SemanticLearningMachine
from algorithms.semantic_learning_machine.algorithm_cnn import DeepSemanticLearningMachine
from benchmark.algorithm import BenchmarkSLM, BenchmarkNEAT, BenchmarkSGA, BenchmarkSLM_RST, BenchmarkSLM_RWT, BenchmarkDSLM
from data.extract import get_input_variables, get_target_variable, is_classification_target

# from neat.nn import FeedForwardNetwork
# from multiprocessing import Process 
# from threading import Thread
# Disable the monitor thread. (https://github.com/tqdm/tqdm/issues/481)
tqdm.monitor_interval = 0

TIME_LIMIT_SECONDS = 50  # changed from 300
TIME_BUFFER = 0.1

MAX_COMBINATIONS = 3


class Evaluator(object):

    def __init__(self, model, configurations, training_set, validation_set, testing_set, metric):
        self.model = model
        self.configurations = configurations
        self.training_set = training_set
        self.validation_set = validation_set
        self.testing_set = testing_set
        self.metric = metric

    def _get_learner_meta(self, learner):
        
        learner_meta = {}
        
        if is_classification_target(get_target_variable(self.testing_set).values):
            learner_meta['training_accuracy'] = self._calculate_accuracy(learner, self.training_set)
            learner_meta['testing_accuracy'] = self._calculate_accuracy(learner, self.testing_set)
            # print('\t\t\t\t\ttesting AUROC vs. training AUROC:', learner_meta['testing_accuracy'], 'vs.', learner_meta['training_accuracy'] )
        else:
            learner_meta['testing_value'] = self._calculate_value(learner, self.testing_set)
        
        return learner_meta
    
    def _calculate_accuracy(self, learner, dataset):
        prediction = learner.predict(get_input_variables(dataset).values)
        target = get_target_variable(dataset).values
        return Accuracy.evaluate(prediction, target.astype(int))
    
    #################################################################################################################
    """ beginning of code for nested cv"""

    def _fit_learner(self, verbose=False):

        def time_seconds(): return default_timer()

        # Create learner from configuration
        learner = self.model(**self.configurations)
        # Train learner 
        if self.__class__.__bases__[0] == EvaluatorSklearn:
            start_time = time_seconds()
            learner.fit(get_input_variables(self.training_set).values, get_target_variable(self.training_set).values)
            training_time = time_seconds() - start_time
        else: 
            start_time = time_seconds()
            learner.fit(get_input_variables(self.training_set).values, get_target_variable(self.training_set).values,
                        self.metric, verbose)
            training_time = time_seconds() - start_time
        # testing_value = self._calculate_value(learner, self.testing_set)
        return {
            'learner': learner,
            # 'testing_value': testing_value,
            'training_time': training_time
        }
    
    def run_nested_cv(self, verbose=False):
        log = self._fit_learner(verbose)
        learner_meta = self._get_learner_meta(log['learner'])
        learner_meta['training_time'] = log['training_time']
        return learner_meta

    """ end of code for nested cv"""
    #################################################################################################################     
    
    def _select_best_learner(self, time_limit=TIME_LIMIT_SECONDS, time_buffer=TIME_BUFFER, verbose=False):
        # Best learner found (lowest validation error).
        best_learner = None
        # Lowest validation error found.
        best_validation_value = float('-Inf') if self.metric.greater_is_better else float('Inf')
        # Validation error list.
        validation_value_list = list()

        # Current time in seconds.
        def time_seconds(): return default_timer()

        # Random order of configurations.
        shuffle(self.configurations)
        # Number of configurations run.
        number_of_runs = 0
        # # Start of run.
        # run_start = time_seconds()
        # Time left.

        # def time_left(): return time_limit - (time_seconds() - run_start)
        # Iterate though all configurations.
        for configuration in tqdm(self.configurations):
            # p = Thread(target=self._fit_learner, args=(configuration, verbose))
            
            # Create learner from configuration.
            learner = self.model(**configuration)
            # Train learner.
            if self.__class__.__bases__[0] == EvaluatorSklearn:
                learner.fit(get_input_variables(self.training_set).values,
                            get_target_variable(self.training_set).values)
            else:
                start_time = time_seconds()
                learner.fit(get_input_variables(self.training_set).values, get_target_variable(self.training_set).values,
                            self.metric, verbose)
                print("\ntime to fit algorithm: ", configuration, time_seconds() - start_time)
            # Calculate validation value.
            validation_value = self._calculate_value(learner, self.validation_set)
            # If validation error lower than best validation error, set learner as best learner and validation error as best validation error.
            if is_better(validation_value, best_validation_value, self.metric):
                best_learner = learner
                best_validation_value = validation_value
            # Add configuration and validation error to validation error list.
            validation_value_list.append((configuration, validation_value))
            # Increase number of runs.
            number_of_runs += 1
            # Calculate time left.
            # run_end = time_left()
            # Calculate time expected for next run.
            # run_expected = (time_limit - run_end) / number_of_runs
            # If no time left or time expected for next run is greater than time left, break.
            # if run_end < 0 or run_end * (1+time_buffer) < run_expected:
            #     print("break!!!!!")
            #     break

            if number_of_runs == MAX_COMBINATIONS: 
                break 

        # When all configurations tested, return best learner.
        return {
            'best_learner': best_learner,
            'validation_value_list': validation_value_list
        }

    def _calculate_value(self, learner, dataset):
        prediction = learner.predict(get_input_variables(dataset).values)
        target = get_target_variable(dataset).values
        return self.metric.evaluate(prediction, target)
        # return self.metric.evaluate(prediction, target.astype(int))

    def run(self, time_limit=TIME_LIMIT_SECONDS, time_buffer=TIME_BUFFER, verbose=False):
        log = self._select_best_learner(time_limit, time_buffer, verbose)
        learner_meta = self._get_learner_meta(log['best_learner'])
        learner_meta['validation_value_list'] = log['validation_value_list']
        return learner_meta


class EvaluatorSLM(Evaluator):

    def __init__(self, configurations, training_set, validation_set, testing_set, metric):
        super().__init__(BenchmarkSLM, configurations, training_set, validation_set, testing_set, metric)

    def _get_learner_meta(self, learner):
        learner_meta = super()._get_learner_meta(learner)
        learner_meta['nr_generations'] = learner.current_generation - 1
        learner_meta['training_value'] = learner.champion.value
        learner_meta['neural_network'] = learner.champion.neural_network
        learner_meta['training_value_evolution'] = self._get_training_value_evolution(learner)
        # learner_meta['testing_value_evolution'] = self._get_testing_value_evolution(learner)
        learner_meta['processing_time'] = self._get_processing_time(learner)
        learner_meta['topology'] = self._get_topology(learner)
        return learner_meta

    def _get_solutions(self, learner):
        return learner.log['solution_log']

    def _get_training_value_evolution(self, learner):
        solutions = self._get_solutions(learner)
        return [solution.value for solution in solutions]

    def _get_testing_value_evolution(self, learner):
        solutions = self._get_solutions(learner)
        return [self._calculate_network_value(solution.neural_network, self.testing_set) for solution in solutions]

    def _calculate_network_value(self, network, dataset):
        predictions = network.predict(get_input_variables(dataset).values)
        target = get_target_variable(dataset).values
        return self.metric.evaluate(predictions, target)

    def _get_processing_time(self, learner):
        return learner.log['time_log']

    def _get_topology(self, learner):
        solutions = self._get_solutions(learner)
        return [solution.neural_network.get_topology() for solution in solutions]
    
    def get_corresponding_algo():
        return SemanticLearningMachine


class EvaluatorDSLM(Evaluator):

    def __init__(self, configurations, training_set, validation_set, testing_set, metric):
        super().__init__(BenchmarkDSLM, configurations, training_set, validation_set, testing_set, metric)

    def _get_learner_meta(self, learner):
        learner_meta = super()._get_learner_meta(learner)
        learner_meta['nr_generations'] = learner.current_generation - 1
        learner_meta['training_value'] = learner.champion.value
        learner_meta['neural_network'] = learner.champion.neural_network
        learner_meta['training_value_evolution'] = self._get_training_value_evolution(learner)
        # learner_meta['testing_value_evolution'] = self._get_testing_value_evolution(learner)
        learner_meta['processing_time'] = self._get_processing_time(learner)
        learner_meta['topology'] = self._get_topology(learner)
        return learner_meta

    def _get_solutions(self, learner):
        return learner.log['solution_log']

    def _get_training_value_evolution(self, learner):
        solutions = self._get_solutions(learner)
        return [solution.value for solution in solutions]

    def _get_testing_value_evolution(self, learner):
        solutions = self._get_solutions(learner)
        return [self._calculate_network_value(solution.neural_network, self.testing_set) for solution in solutions]

    def _calculate_network_value(self, network, dataset):
        predictions = network.predict(get_input_variables(dataset).values)
        target = get_target_variable(dataset).values
        return self.metric.evaluate(predictions, target)

    def _get_processing_time(self, learner):
        return learner.log['time_log']

    def _get_topology(self, learner):
        solutions = self._get_solutions(learner)
        return [solution.neural_network.get_topology() for solution in solutions]

    def get_corresponding_algo():
        return DeepSemanticLearningMachine


class EvaluatorSLM_RST(Evaluator):

    def __init__(self, configurations, training_set, validation_set, testing_set, metric):
        super().__init__(BenchmarkSLM_RST, configurations, training_set, validation_set, testing_set, metric)

    def _get_learner_meta(self, learner):
        learner_meta = super()._get_learner_meta(learner)
        learner_meta['nr_generations'] = learner.current_generation
        learner_meta['training_value'] = learner.champion.value
        learner_meta['neural_network'] = learner.champion.neural_network
        learner_meta['training_value_evolution'] = self._get_training_value_evolution(learner)
        # learner_meta['testing_value_evolution'] = self._get_testing_value_evolution(learner)
        learner_meta['processing_time'] = self._get_processing_time(learner)
        learner_meta['topology'] = self._get_topology(learner)
        return learner_meta

    def _get_solutions(self, learner):
        return learner.log['solution_log']

    """calculates accuracy for testing set"""

    def _calculate_accuracy(self, learner, dataset):
        prediction = learner.predict(get_input_variables(dataset).values)
        target = get_target_variable(dataset).values
        return Accuracy.evaluate(prediction, target.astype(int))

    def _get_training_value_evolution(self, learner):
        solutions = self._get_solutions(learner)
        return [solution.value for solution in solutions]

    def _get_testing_value_evolution(self, learner):
        solutions = self._get_solutions(learner)
        return [self._calculate_network_value(solution.neural_network, self.testing_set) for solution in solutions]

    def _calculate_network_value(self, network, dataset):
        predictions = network.predict(get_input_variables(dataset).values)
        target = get_target_variable(dataset).values
        return self.metric.evaluate(predictions, target)

    def _get_processing_time(self, learner):
        return learner.log['time_log']

    def _get_topology(self, learner):
        solutions = self._get_solutions(learner)
        return [solution.neural_network.get_topology() for solution in solutions]

    def get_corresponding_algo():
        return BenchmarkSLM_RST


class EvaluatorSLM_RWT(Evaluator):

    def __init__(self, configurations, training_set, validation_set, testing_set, metric):
        super().__init__(BenchmarkSLM_RWT, configurations, training_set, validation_set, testing_set, metric)

    def _get_learner_meta(self, learner):
        learner_meta = super()._get_learner_meta(learner)
        learner_meta['nr_generations'] = learner.current_generation
        learner_meta['training_value'] = learner.champion.value
        learner_meta['neural_network'] = learner.champion.neural_network
        learner_meta['training_value_evolution'] = self._get_training_value_evolution(learner)
        # learner_meta['testing_value_evolution'] = self._get_testing_value_evolution(learner)
        learner_meta['processing_time'] = self._get_processing_time(learner)
        learner_meta['topology'] = self._get_topology(learner)
        return learner_meta

    def _get_solutions(self, learner):
        return learner.log['solution_log']

    """calculates accuracy for testing set"""

    def _calculate_accuracy(self, learner, dataset):
        prediction = learner.predict(get_input_variables(dataset).values)
        target = get_target_variable(dataset).values
        return Accuracy.evaluate(prediction, target.astype(int))

    def _get_training_value_evolution(self, learner):
        solutions = self._get_solutions(learner)
        return [solution.value for solution in solutions]

    def _get_testing_value_evolution(self, learner):
        solutions = self._get_solutions(learner)
        return [self._calculate_network_value(solution.neural_network, self.testing_set) for solution in solutions]

    def _calculate_network_value(self, network, dataset):
        predictions = network.predict(get_input_variables(dataset).values)
        target = get_target_variable(dataset).values
        return self.metric.evaluate(predictions, target)

    def _get_processing_time(self, learner):
        return learner.log['time_log']

    def _get_topology(self, learner):
        solutions = self._get_solutions(learner)
        return [solution.neural_network.get_topology() for solution in solutions]

    def get_corresponding_algo():
        return BenchmarkSLM_RWT


class EvaluatorNEAT(Evaluator):

    def __init__(self, configurations, training_set, validation_set, testing_set, metric):
        super().__init__(BenchmarkNEAT, configurations, training_set, validation_set, testing_set, metric)

    def _get_learner_meta(self, learner):
        learner_meta = super()._get_learner_meta(learner)
        learner_meta['training_value'] = self._get_solution_value(learner.champion)
        learner_meta['training_value_evolution'] = self._get_training_value_evolution(learner)
        # learner_meta['testing_value_evolution'] = self._get_testing_value_evolution(learner)
        learner_meta['processing_time'] = self._get_processing_time(learner)
        learner_meta['topology'] = self._get_topology(learner)
        return learner_meta

    def _get_solutions(self, learner):
        return learner.log['solution_log']

    def _get_solution_value(self, solution):
        return solution.fitness if self.metric.greater_is_better else 1 / solution.fitness

    def _get_training_value_evolution(self, learner):
        solutions = self._get_solutions(learner)
        return [self._get_solution_value(solution) for solution in solutions]

    def _get_testing_value_evolution(self, learner):
        solutions = self._get_solutions(learner)
        return [self._calculate_solution_value(solution, self.testing_set, learner) for solution in solutions]

    def _calculate_solution_value(self, solution, dataset, learner):
        X = get_input_variables(dataset).values
        target = get_target_variable(dataset).values
        neural_network = FeedForwardNetwork.create(solution, learner.configuration)
        prediction = self._predict_neural_network(neural_network, X)
        return self.metric.evaluate(prediction, target)

    def _predict_neural_network(self, neural_network, X):
        predictions = array([])
        for data in X:
            predictions = append(predictions, float(neural_network.activate(data)[0]))
        return predictions

    def _get_processing_time(self, learner):
        return learner.log['time_log']

    def _get_topology(self, learner):
        solutions = self._get_solutions(learner)
        return [self._get_genome_topology(solution) for solution in solutions]

    def _get_genome_topology(self, genome):
        return {
            'neurons': len(genome.nodes),
            'connections': len(genome.connections)
        }


class EvaluatorFTNE(EvaluatorSLM):

    def __init__(self, configurations, training_set, validation_set, testing_set, metric):
        Evaluator.__init__(self, BenchmarkSGA, configurations, training_set,
                           validation_set, testing_set, metric)


class EvaluatorSklearn(Evaluator):

    def _get_learner_meta(self, learner):
        learner_meta = super()._get_learner_meta(learner)
        learner_meta['training_value'] = self._calculate_value(learner, self.training_set)
        return learner_meta


class EvaluatorSVC(EvaluatorSklearn):

    def __init__(self, configurations, training_set, validation_set, testing_set, metric):
        super().__init__(SVC, configurations, training_set, validation_set, testing_set, metric)


class EvaluatorSVR(EvaluatorSklearn):

    def __init__(self, configurations, training_set, validation_set, testing_set, metric):
        super().__init__(SVR, configurations, training_set, validation_set, testing_set, metric)


class EvaluatorMLPC(EvaluatorSklearn):

    def __init__(self, configurations, training_set, validation_set, testing_set, metric):
        super().__init__(MLPClassifier, configurations, training_set, validation_set, testing_set, metric)

    def _select_best_learner(self, time_limit=TIME_LIMIT_SECONDS, time_buffer=TIME_BUFFER, verbose=False):
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            return super()._select_best_learner(time_limit, time_buffer, verbose)
    
    def get_corresponding_algo():
        return MLPClassifier


class EvaluatorMLPR(EvaluatorSklearn):

    def __init__(self, configurations, training_set, validation_set, testing_set, metric):
        super().__init__(MLPRegressor, configurations, training_set, validation_set, testing_set, metric)

    def _select_best_learner(self, time_limit=TIME_LIMIT_SECONDS, time_buffer=TIME_BUFFER, verbose=False):
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            return super()._select_best_learner(time_limit, time_buffer, verbose)

    def get_corresponding_algo():
        return MLPRegressor


class EvaluatorRFC(EvaluatorSklearn):

    def __init__(self, configurations, training_set, validation_set, testing_set, metric):
        super().__init__(RandomForestClassifier, configurations,
                         training_set, validation_set, testing_set, metric)


class EvaluatorRFR(EvaluatorSklearn):

    def __init__(self, configurations, training_set, validation_set, testing_set, metric):
        super().__init__(RandomForestRegressor, configurations,
                         training_set, validation_set, testing_set, metric)


class EvaluatorEnsemble(Evaluator):

    def __init__(self, configurations, training_set, validation_set, testing_set, metric):
        super().__init__(Ensemble, configurations, training_set, validation_set, testing_set, metric)

    def _get_learner_meta(self, learner):
        learner_meta = super()._get_learner_meta(learner)
        learner_meta['training_value'] = self._calculate_value(learner, self.training_set)
        return learner_meta


class EvaluatorEnsembleBagging(Evaluator):
    
    def __init__(self, configurations, training_set, validation_set, testing_set, metric):
        super().__init__(EnsembleBagging, configurations, training_set, validation_set, testing_set, metric)

    def _get_learner_meta(self, learner):
        learner_meta = super()._get_learner_meta(learner)
        learner_meta['training_value'] = self._calculate_value(learner, self.training_set)
        return learner_meta


class EvaluatorEnsembleRandomIndependentWeighting(Evaluator):
    
    def __init__(self, configurations, training_set, validation_set, testing_set, metric):
        super().__init__(EnsembleRandomIndependentWeighting, configurations, training_set, validation_set, testing_set, metric)

    def _get_learner_meta(self, learner):
        learner_meta = super()._get_learner_meta(learner)
        learner_meta['training_value'] = self._calculate_value(learner, self.training_set)
        return learner_meta


class EvaluatorEnsembleBoosting(Evaluator):
    
    def __init__(self, configurations, training_set, validation_set, testing_set, metric):
        super().__init__(EnsembleBoosting, configurations, training_set, validation_set, testing_set, metric)

    def _get_learner_meta(self, learner):
        learner_meta = super()._get_learner_meta(learner)
        learner_meta['training_value'] = self._calculate_value(learner, self.training_set)
        return learner_meta
     

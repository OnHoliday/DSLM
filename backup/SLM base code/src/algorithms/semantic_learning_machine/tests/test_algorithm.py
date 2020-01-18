from data.io_plm import load_samples
from data.extract import get_input_variables, get_target_variable
from timeit import default_timer 
from algorithms.semantic_learning_machine.algorithm import SemanticLearningMachine
from algorithms.semantic_learning_machine.mutation_operator import Mutation2
from algorithms.common.stopping_criterion import MaxGenerationsCriterion, ErrorDeviationVariationCriterion, TrainingImprovementEffectivenessCriterion
from algorithms.common.metric import RootMeanSquaredError, Accuracy
import unittest


class TestAlgorithm(unittest.TestCase):

    def setUp(self):
        self.training, self.validation, self.testing = load_samples('c_cancer', 0)

    def test_fit(self):
        print("Basic tests of fit()...")
        def time_seconds(): return default_timer()
        start_time = time_seconds()
        algorithm = SemanticLearningMachine(100, MaxGenerationsCriterion(200), 3, 0.01, 50, Mutation2(), RootMeanSquaredError, True)
        X = get_input_variables(self.training).values
        y = get_target_variable(self.training).values
        start_time = time_seconds()
        algorithm.fit(X, y, RootMeanSquaredError, verbose=False)
        print("time to train algorithm: ", (time_seconds()-start_time))
        self.assertTrue(expr=algorithm.champion)
        print()

    def test_ols(self):
        print('OLS tests of fit()...')
        def time_seconds(): return default_timer()
        start_time = time_seconds()
        algorithm = SemanticLearningMachine(100, MaxGenerationsCriterion(200), 3, 'optimized', 50, Mutation2(), RootMeanSquaredError, True)
        X = get_input_variables(self.training).values
        y = get_target_variable(self.training).values
        start_time = time_seconds()
        algorithm.fit(X, y, RootMeanSquaredError, verbose=False)
        print("time to train algorithm: ", (time_seconds()-start_time))
        self.assertTrue(expr=algorithm.champion)
        print()

    def test_edv(self):
        print('EDV tests of fit()...')
        def time_seconds(): return default_timer()
        start_time = time_seconds()
        algorithm = SemanticLearningMachine(100, ErrorDeviationVariationCriterion(0.25), 3, 0.01, 50, Mutation2(), RootMeanSquaredError, True)
        X = get_input_variables(self.training).values
        y = get_target_variable(self.training).values
        start_time = time_seconds()
        algorithm.fit(X, y, RootMeanSquaredError, verbose=False)
        print("time to train algorithm: ", (time_seconds()-start_time))
        self.assertTrue(expr=algorithm.champion)
        print()
    
    def test_tie(self):
        print('TIE tests of fit()...')
        def time_seconds(): return default_timer()
        start_time = time_seconds()
        algorithm = SemanticLearningMachine(100, TrainingImprovementEffectivenessCriterion(0.25), 3, 0.01, 50, Mutation2(), RootMeanSquaredError, True)
        X = get_input_variables(self.training).values
        y = get_target_variable(self.training).values
        start_time = time_seconds()
        algorithm.fit(X, y, RootMeanSquaredError, verbose=False)
        print("time to train algorithm: ", (time_seconds()-start_time))
        self.assertTrue(expr=algorithm.champion)
        print()

if __name__ == '__main__':
    unittest.main()
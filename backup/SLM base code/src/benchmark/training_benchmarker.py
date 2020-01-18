import datetime
from random import randint

from sklearn.model_selection import KFold, StratifiedKFold

from algorithms.common.metric import RootMeanSquaredError, is_better, AUROC
from algorithms.common.stopping_criterion import MaxGenerationsCriterion
from benchmark.new_configuration import DEFAULT_NUMBER_OF_ITERATIONS
from data.extract import is_classification
from data.io_plm import benchmark_to_pickle, load_standardized_samples
import pandas as pd

# Returns the current date and time.
_now = datetime.datetime.now()

_OUTER_FOLDS = 30
_INNER_FOLDS = 2


class TrainingBenchmarker():

    def __init__(self, dataset_name, learning_metric=None, selection_metric=None, models=None, ensembles=None, benchmark_id=None):
        """Initializes benchmark environment."""
        
        self.benchmark_id = benchmark_id
        self.dataset_name = dataset_name
        # Creates file name as combination of dataset name and and date
        self.file_name = self.dataset_name + "_" + self.benchmark_id + "__" + _now.strftime("%Y_%m_%d__%H_%M_%S")
        
        # Loads samples into object
        self.samples = load_standardized_samples(dataset_name)
        self.ensembles = ensembles
        self.models = models
        
        # If dataset is classification problem, remove regression models. Else, vice versa.
        if is_classification(self.samples):  # original self.samples[0][0] new self.samples
            self.classification = True
            
            if learning_metric != None:
                self.learning_metric = learning_metric
            else:
                self.learning_metric = RootMeanSquaredError
            
            if selection_metric != None:
                self.selection_metric = selection_metric
            else:
                self.selection_metric = AUROC
            
            if 'mlpr_lbfgs' in self.models.keys():
                del self.models['mlpr_lbfgs']
            if 'mlpr_adam' in self.models.keys():
                del self.models['mlpr_adam']
            if 'mlpr_sgd' in self.models.keys():
                del self.models['mlpr_sgd']
        else:
            self.classification = False
            
            if learning_metric != None:
                self.learning_metric = learning_metric
            else:
                self.learning_metric = RootMeanSquaredError
            
            if selection_metric != None:
                self.selection_metric = selection_metric
            else:
                self.selection_metric = RootMeanSquaredError

            if 'mlpc_lbfgs' in self.models.keys():
                del self.models['mlpc_lbfgs']
            if 'mlpc_adam' in self.models.keys():
                del self.models['mlpc_adam']
            if 'mlpc_sgd' in self.models.keys(): 
                del self.models['mlpc_sgd']
        
        # if models = MLP, remove Random Independent Weighting 
        if self.ensembles != None:
            if 'mlpc_lbfgs' in self.models.keys() or 'mlpr_lbfgs' in self.models.keys(): 
                if 'riw' in self.ensembles.keys(): 
                    del self.ensembles['riw']
        
        # Create results dictionary with models under study.
        self.results = {k: [None for i in range(_OUTER_FOLDS)] for k in self.models.keys()}
        
        if self.ensembles != None:
            self.results_ensemble = {ensemble: [None for i in range(_OUTER_FOLDS)] for ensemble in self.ensembles.keys()}
        
        self.best_result = [None for i in range(_OUTER_FOLDS)]
        
        # Serialize benchmark environment.
        benchmark_to_pickle(self)
    
    def _evaluate_algorithm(self, algorithm, configurations, training_set, validation_set, testing_set, metric):
        """Creates evaluator, based on algorithms and configurations."""
        evaluator = algorithm(configurations, training_set, validation_set, testing_set, metric)
        return evaluator.run_nested_cv()
    
    def get_dataset_size(self, dataset): 
        return dataset.shape[0]

    def _get_inner_folds(self, outer_iteration):
        if self.classification:
            return StratifiedKFold(n_splits=_INNER_FOLDS, random_state=outer_iteration, shuffle=True)
        return KFold(n_splits=_INNER_FOLDS, random_state=outer_iteration, shuffle=True)

    def _get_outer_folds(self, outer_iteration):
        if self.classification:
            return StratifiedKFold(n_splits=_OUTER_FOLDS, random_state=outer_iteration, shuffle=True)
        return KFold(n_splits=_OUTER_FOLDS, random_state=outer_iteration, shuffle=True)

    def run_nested_cv(self):
        """ runs benchmark study on a nested cross-validation environment """
        
        #=======================================================================
        # print('self.learning_metric =', self.learning_metric)
        # print('self.selection_metric =', self.selection_metric)
        #=======================================================================
        
        """ N configuration for each method, trained on all data, selected from the same data """
        
        print('Entering training benchmark for dataset:', self.dataset_name)
        
        training_data = pd.DataFrame(self.samples.values)
        
        for outer_cv in range(_OUTER_FOLDS):
            
            print('\n\tIndex of outer fold:', outer_cv)
            
            for key in self.models.keys():
                
                print('\t\tAlgorithm with key:', key)
                
                if not self.results[key][outer_cv]:
                    
                    if self.classification:
                        best_training_value = float('-Inf')
                    else:
                        best_training_value = float('-Inf') if self.selection_metric.greater_is_better else float('Inf')
                    
                    training_value_list = list()
                    for configuration in range(self.models[key]['max_combinations']):
                        
                        print('\n\t\t\tIndex of algorithm configuration:', len(training_value_list))
                        
                        if(len(self.models[key]['algorithms'])) > 1:
                            option = randint(0, 2)
                            algorithm = self.models[key]['algorithms'][option]
                            config = self.models[key]['configuration_method'](option)
                        else:
                            algorithm = self.models[key]['algorithms'][0]
                            #===================================================
                            # if (key == 'mlpc_sgd' or key == 'mlpc_adam' or key == 'mlpr_sgd' or key == 'mlpr_adam'):
                            #===================================================
                            if key.startswith('mlp'):
                                # version from 01-22
                                # config = self.models[key]['configuration_method'](self.get_dataset_size(training_outer))
                                # version from 01-25
                                batch_size = int(training_data.shape[0])
                                # batch_size = int(training_outer.shape[0] / _INNER_FOLDS) * 2
                                config = self.models[key]['configuration_method'](batch_size)
                            else:
                                config = self.models[key]['configuration_method']()
                        
                        if key.startswith('mlp'):
                            config['max_iter'] = DEFAULT_NUMBER_OF_ITERATIONS
                        else:
                            config['stopping_criterion'] = MaxGenerationsCriterion(DEFAULT_NUMBER_OF_ITERATIONS)
                        
                        results = self._evaluate_algorithm(algorithm=algorithm, configurations=config,
                                                           training_set=training_data, validation_set=None, testing_set=training_data, metric=self.learning_metric)
                        
                        if self.classification:
                            training_value = results['training_accuracy']
                        else:
                            training_value = results['training_value']
                        
                        if self.classification:
                            print("\t\t\tAUROC training: %.3f" % (training_value))
                        else:
                            print("\t\t\tRMSE training: %.3f" % (training_value))
                        
                        if self.classification:
                            if training_value > best_training_value:
                                #===============================================
                                # print('\n\t\t\t\t\tClassification: %.3f is better than %.3f\n' % (training_value, best_training_value))
                                #===============================================
                                best_algorithm = algorithm
                                best_key = key
                                best_configuration = config
                                best_training_value = training_value
                                
                                self.results[key][outer_cv] = results
                                self.results[key][outer_cv]['best_configuration'] = best_configuration
                                self.results[key][outer_cv]['avg_inner_validation_error'] = best_training_value
                                self.results[key][outer_cv]['avg_inner_training_error'] = best_training_value
                                
                                best_overall_algorithm = best_algorithm
                                best_overall_configuration = best_configuration
                                best_overall_key = best_key
                                
                                self.best_result[outer_cv] = self.results[key][outer_cv]
                                self.best_result[outer_cv]['best_overall_algorithm'] = best_overall_algorithm
                                self.best_result[outer_cv]['best_overall_configuration'] = best_overall_configuration
                                self.best_result[outer_cv]['best_overall_key'] = best_overall_key
                                
                            #===================================================
                            # else:
                            #     print('\n\t\t\t\t\tClassification: %.3f is worse (!) than %.3f\n' % (training_value, best_training_value))
                            #===================================================
                        else:
                            if is_better(training_value, best_training_value, self.selection_metric):
                                #===============================================
                                # print('\n\t\t\t\t\tRegression: %.3f is better than %.3f\n' % (training_value, best_training_value))
                                #===============================================
                                best_algorithm = algorithm
                                best_key = key
                                best_configuration = config
                                best_training_value = training_value
                                
                                self.results[key][outer_cv] = results
                                self.results[key][outer_cv]['best_configuration'] = best_configuration
                                self.results[key][outer_cv]['avg_inner_validation_error'] = best_training_value
                                self.results[key][outer_cv]['avg_inner_training_error'] = best_training_value
                                
                                best_overall_algorithm = best_algorithm
                                best_overall_configuration = best_configuration
                                best_overall_key = best_key
                                
                                self.best_result[outer_cv] = self.results[key][outer_cv]
                                self.best_result[outer_cv]['best_overall_algorithm'] = best_overall_algorithm
                                self.best_result[outer_cv]['best_overall_configuration'] = best_overall_configuration
                                self.best_result[outer_cv]['best_overall_key'] = best_overall_key
                                
                            #===================================================
                            # else:
                            #     print('\n\t\t\t\t\tRegression: %.3f is worse (!) than %.3f\n' % (training_value, best_training_value))
                            #===================================================
                        
                        training_value_list.append((configuration, training_value))
                    
                    if self.classification:
                        print("\n\t\tAUROC training: %.3f" % (self.results[key][outer_cv]['training_accuracy']))
                    else:
                        print("\n\t\tRMSE training: %.3f" % (self.results[key][outer_cv]['training_value']))
        
        # Serialize benchmark 
        benchmark_to_pickle(self)
        
        print('Leaving training benchmark for dataset:', self.dataset_name)

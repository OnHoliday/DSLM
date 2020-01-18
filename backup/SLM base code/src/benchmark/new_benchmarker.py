import datetime
from random import randint

from numpy import mean
from sklearn.model_selection import KFold, StratifiedKFold

from algorithms.common.metric import RootMeanSquaredError, is_better, AUROC
from data.extract import is_classification, get_input_variables, get_target_variable, is_binary
from data.io_plm import benchmark_to_pickle, load_standardized_samples
import pandas as pd

# Returns the current date and time.
_now = datetime.datetime.now()

_OUTER_FOLDS = 30
_INNER_FOLDS = 2


class Benchmarker():

    def __init__(self, dataset_name, learning_metric=None, selection_metric=None, models=None, ensembles=None, benchmark_id=None, file_path=None):
        """Initializes benchmark environment."""

        self.benchmark_id = benchmark_id
        self.dataset_name = dataset_name
        # Creates file name as combination of dataset name and and date
        self.file_name = self.dataset_name + "_" + self.benchmark_id + "__" + _now.strftime("%Y_%m_%d__%H_%M_%S")

        # Loads samples into object
        self.samples = load_standardized_samples(dataset_name, file_path)
        self.ensembles = ensembles
        self.models = models

        if self.dataset_name == 'data_batch_1':  # todo zrob to bardziej generalnie !!!!
            print(self.samples.keys())
            b = self.samples[b'labels']
            self.samples = pd.DataFrame(self.samples[b'data'])
            self.samples = self.samples/255
            self.samples[3072] = b
            self.samples = self.samples.head(1000) # change it to get all data not sample
            print(self.samples.shape)

        # If dataset is classification problem, remove regression models. Else, vice versa.
        if is_binary(self.samples):  # original self.samples[0][0] new self.samples #todo zrob tu walidacje ze to faktycznie klasyfikacja, ale problem taki ze nie dataframe tylko dict
            self.classification = True
            self.binary = True

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

        elif is_classification(self.samples):
            self.classification = True
            self.binary = False

            if learning_metric != None:
                self.learning_metric = learning_metric
            else:
                self.learning_metric = RootMeanSquaredError

            if selection_metric != None:
                self.selection_metric = selection_metric
            else:
                self.selection_metric = AUROC

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
        
        print('Entering run_nested_cv for dataset:', self.dataset_name)
        
        outer_cv = 0
        outer_folds = self._get_outer_folds(outer_cv)
        for training_outer_index, testing_index in outer_folds.split(get_input_variables(self.samples).values, get_target_variable(self.samples).values):
            
            print('\n\tIndex of outer fold:', outer_cv)
            
            training_outer, testing = pd.DataFrame(self.samples.values[training_outer_index]), pd.DataFrame(self.samples.values[testing_index])
            
            if self.classification:
                best_overall_validation_value = float('-Inf')
            else:
                best_overall_validation_value = float('-Inf') if self.selection_metric.greater_is_better else float('Inf')
            
            for key in self.models.keys():
                
                print('\t\tAlgorithm with key:', key)
                
                if not self.results[key][outer_cv]:
                    
                    if self.classification:
                        best_validation_value = float('-Inf')
                    else:
                        best_validation_value = float('-Inf') if self.selection_metric.greater_is_better else float('Inf')
                    
                    validation_value_list = list()
                    for configuration in range(self.models[key]['max_combinations']):
                        
                        print('\n\t\t\tIndex of algorithm configuration:', len(validation_value_list))
                        
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
                                batch_size = int(training_outer.shape[0] / _INNER_FOLDS)
                                # batch_size = int(training_outer.shape[0] / _INNER_FOLDS) * 2
                                config = self.models[key]['configuration_method'](batch_size)
                            else:
                                config = self.models[key]['configuration_method']()
                        
                        inner_folds = self._get_inner_folds(outer_cv)
                        tmp_valid_training_values_list = list()
                        for training_inner_index, validation_index in inner_folds.split(get_input_variables(training_outer).values, get_target_variable(training_outer).values):
                            
                            print('\t\t\t\tIndex of inner fold:', len(tmp_valid_training_values_list))
                            
                            training_inner, validation = pd.DataFrame(training_outer.values[training_inner_index]), pd.DataFrame(training_outer.values[validation_index])
                            
                            results = self._evaluate_algorithm(algorithm=algorithm, configurations=config,
                                                               training_set=training_inner, validation_set=None, testing_set=validation, metric=self.learning_metric)
                            
                            # print('results[testing_value] =', results['testing_value'], ', results[training_value] =', results['training_value'])
                            
                            if self.classification:
                                tmp_valid_training_values_list.append((results['testing_accuracy'], results['training_accuracy']))
                            else:
                                tmp_valid_training_values_list.append((results['testing_value'], results['training_value']))
                        
                        # Calculate average validation value and check if the current value is better than the best one 
                        average_validation_value = mean(tmp_valid_training_values_list, axis=0)[0]
                        average_training_value = mean(tmp_valid_training_values_list, axis=0)[1]
                        
                        if self.classification:
                            print("\t\t\tAverage AUROC training vs. validation: %.3f vs. %.3f" % (average_training_value, average_validation_value))
                        else:
                            print("\t\t\tAverage RMSE training vs. validation: %.3f vs. %.3f" % (average_training_value, average_validation_value))
                        
                        if self.classification:
                            if average_validation_value > best_validation_value:
                                #===============================================
                                # print('\n\t\t\t\t\tClassification: %.3f is better than %.3f\n' % (average_validation_value, best_validation_value))
                                #===============================================
                                best_algorithm = algorithm
                                best_key = key
                                best_configuration = config
                                best_validation_value = average_validation_value
                                best_training_value = average_training_value
                            #===================================================
                            # else:
                            #     print('\n\t\t\t\t\tClassification: %.3f is worse (!) than %.3f\n' % (average_validation_value, best_validation_value))
                            #===================================================
                        else:
                            if is_better(average_validation_value, best_validation_value, self.selection_metric):
                                #===============================================
                                # print('\n\t\t\t\t\tRegression: %.3f is better than %.3f\n' % (average_validation_value, best_validation_value))
                                #===============================================
                                best_algorithm = algorithm
                                best_key = key
                                best_configuration = config
                                best_validation_value = average_validation_value
                                best_training_value = average_training_value
                            #===================================================
                            # else:
                            #     print('\n\t\t\t\t\tRegression: %.3f is worse (!) than %.3f\n' % (average_validation_value, best_validation_value))
                            #===================================================
                        
                        # Add configuration and validation error to validation error list.
                        validation_value_list.append((configuration, average_validation_value))
                    
                    """ all allowed configurations assessed of a given variant/algorithm/method (key) """
                    print('\n\t\tEvaluating best configuration in outer fold with index', outer_cv)
                    self.results[key][outer_cv] = self._evaluate_algorithm(algorithm=best_algorithm, configurations=best_configuration,
                                                                    training_set=training_outer, validation_set=None, testing_set=testing, metric=self.learning_metric)
                    self.results[key][outer_cv]['best_configuration'] = best_configuration
                    self.results[key][outer_cv]['avg_inner_validation_error'] = best_validation_value
                    self.results[key][outer_cv]['avg_inner_training_error'] = best_training_value
                    if self.classification:
                        self.results[key][outer_cv]['avg_inner_validation_accuracy'] = best_validation_value
                        self.results[key][outer_cv]['avg_inner_training_accuracy'] = best_training_value                    
                    
                    if self.classification:
                        print("\n\t\tAUROC training vs. test: %.3f vs. %.3f" % (self.results[key][outer_cv]['training_accuracy'], self.results[key][outer_cv]['testing_accuracy']))
                        #=======================================================
                        # print("\n\t\tAlgorithm %s, AUROC training vs. test: %.3f vs. %.3f" % (key, self.results[key][outer_cv]['training_accuracy'], self.results[key][outer_cv]['testing_accuracy']))
                        #=======================================================
                    else:
                        print("\n\t\tRMSE training vs. test: %.3f vs. %.3f" % (self.results[key][outer_cv]['training_value'], self.results[key][outer_cv]['testing_value']))
                        #=======================================================
                        # print("\n\t\tAlgorithm %s, RMSE training vs. test: %.3f vs. %.3f" % (key, self.results[key][outer_cv]['training_value'], self.results[key][outer_cv]['testing_value']))
                        #=======================================================
                    
                    best_overall_algorithm = best_algorithm
                    best_overall_configuration = best_configuration
                    best_overall_key = best_key
                    
                    self.best_result[outer_cv] = self.results[key][outer_cv]
                    self.best_result[outer_cv]['best_overall_algorithm'] = best_overall_algorithm
                    self.best_result[outer_cv]['best_overall_configuration'] = best_overall_configuration
                    self.best_result[outer_cv]['best_overall_key'] = best_overall_key
                    
                    # # Serialize benchmark 
                    # benchmark_to_pickle(self)
            
            outer_cv += 1
        
        # Serialize benchmark 
        benchmark_to_pickle(self)
        
        print('Leaving run_nested_cv for dataset:', self.dataset_name)

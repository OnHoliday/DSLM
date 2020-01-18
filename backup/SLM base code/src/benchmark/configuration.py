import random

from numpy import mean, median, linspace

from algorithms.common.neural_network.neural_network import create_network_from_topology
from algorithms.common.stopping_criterion import MaxGenerationsCriterion, ErrorDeviationVariationCriterion, TrainingImprovementEffectivenessCriterion
from algorithms.semantic_learning_machine.algorithm import SemanticLearningMachine
from algorithms.semantic_learning_machine.mutation_operator import Mutation2, Mutation3, Mutation4
from algorithms.simple_genetic_algorithm.crossover_operator import CrossoverOperatorArithmetic
from algorithms.simple_genetic_algorithm.mutation_operator import MutationOperatorGaussian
from algorithms.simple_genetic_algorithm.selection_operator import SelectionOperatorTournament
from benchmark.algorithm import BenchmarkSLM, BenchmarkSLM_RST, BenchmarkSLM_RWT


def get_random_config_slm_fls_grouped(option=None, init_maximum_layers=5, maximum_iterations=200, maximum_learning_step=1, maximum_neuron_connection_weight=0.5, maximum_bias_connection_weight=1.0, mutation_maximum_new_neurons_per_layer=3):
    config = {}
    
    config['stopping_criterion'] = MaxGenerationsCriterion(random.randint(1, maximum_iterations))
    
    config['population_size'] = 10
    
    config['layers'] = init_maximum_layers
    
    # config['learning_step'] = random.uniform(0.1, maximum_learning_step)
    config['learning_step'] = random.uniform(0.001, maximum_learning_step)
   
    config['maximum_neuron_connection_weight'] = random.uniform(0.1, maximum_neuron_connection_weight)
    config['maximum_bias_connection_weight'] = random.uniform(0.1, maximum_bias_connection_weight)
    
    config['mutation_operator'] = Mutation4(maximum_new_neurons_per_layer=mutation_maximum_new_neurons_per_layer, maximum_bias_connection_weight=config['maximum_bias_connection_weight'])
    
    config['random_sampling_technique'] = False
    config['random_weighting_technique'] = False
    """
    if option == 0:  # no RST and no RWT
        config['random_sampling_technique'] = False
        config['random_weighting_technique'] = False
    elif option == 1:  # RST
        config['random_sampling_technique'] = True
        config['random_weighting_technique'] = False
        config['subset_ratio'] = random.uniform(0.01, 0.99)
    elif option == 2:  # RWT
        config['random_sampling_technique'] = False
        config['random_weighting_technique'] = True
        config['weight_range'] = 1
    """
    
    return config


def get_random_config_slm_ols_grouped(option=None, init_maximum_layers=5, maximum_iterations=200, maximum_neuron_connection_weight=0.5, maximum_bias_connection_weight=1.0, mutation_maximum_new_neurons_per_layer=3): 
    config = {}
    
    config['stopping_criterion'] = MaxGenerationsCriterion(random.randint(1, maximum_iterations))
    
    config['population_size'] = 10
    
    config['layers'] = init_maximum_layers
    
    config['learning_step'] = 'optimized'

    config['maximum_neuron_connection_weight'] = random.uniform(0.1, maximum_neuron_connection_weight)
    config['maximum_bias_connection_weight'] = random.uniform(0.1, maximum_bias_connection_weight)
    
    config['mutation_operator'] = Mutation4(maximum_new_neurons_per_layer=mutation_maximum_new_neurons_per_layer, maximum_bias_connection_weight=config['maximum_bias_connection_weight'])
    
    config['random_sampling_technique'] = False
    config['random_weighting_technique'] = False
    """
    if option == 0:  # no RST and no RWT
        config['random_sampling_technique'] = False
        config['random_weighting_technique'] = False
    elif option == 1:  # RST
        config['random_sampling_technique'] = True
        config['random_weighting_technique'] = False
        config['subset_ratio'] = random.uniform(0.01, 0.99)
    elif option == 2:  # RWT
        config['random_sampling_technique'] = False
        config['random_weighting_technique'] = True
        config['weight_range'] = 1
    """
    
    return config


def get_random_config_slm_fls_tie_edv(init_maximum_layers=5, maximum_iterations=200, maximum_learning_step=1, maximum_neuron_connection_weight=0.5, maximum_bias_connection_weight=1.0, mutation_maximum_new_neurons_per_layer=3):
    config = {}
    
    stopping_crit = random.randint(1, 2)
    # EDV
    if stopping_crit == 1:
        config['stopping_criterion'] = ErrorDeviationVariationCriterion(maximum_iterations=maximum_iterations)
    # TIE
    else:
        config['stopping_criterion'] = TrainingImprovementEffectivenessCriterion(maximum_iterations=maximum_iterations)
    
    config['population_size'] = 100
    
    config['layers'] = init_maximum_layers
    
    # config['learning_step'] = random.uniform(0.1, maximum_learning_step)
    config['learning_step'] = random.uniform(0.001, maximum_learning_step)
    
    config['maximum_neuron_connection_weight'] = random.uniform(0.1, maximum_neuron_connection_weight)
    config['maximum_bias_connection_weight'] = random.uniform(0.1, maximum_bias_connection_weight)
    
    config['mutation_operator'] = Mutation4(maximum_new_neurons_per_layer=mutation_maximum_new_neurons_per_layer, maximum_bias_connection_weight=config['maximum_bias_connection_weight'])
    
    config['random_sampling_technique'] = False
    config['random_weighting_technique'] = False
    
    return config


def get_random_config_slm_ols_edv(init_maximum_layers=5, maximum_iterations=200, maximum_neuron_connection_weight=0.5, maximum_bias_connection_weight=1.0, mutation_maximum_new_neurons_per_layer=3):
    config = {}
    
    config['stopping_criterion'] = ErrorDeviationVariationCriterion(maximum_iterations=maximum_iterations)
    config['population_size'] = 100
    
    config['layers'] = init_maximum_layers
    
    config['learning_step'] = 'optimized'
    
    config['maximum_neuron_connection_weight'] = random.uniform(0.1, maximum_neuron_connection_weight)
    config['maximum_bias_connection_weight'] = random.uniform(0.1, maximum_bias_connection_weight)
    
    config['mutation_operator'] = Mutation4(maximum_new_neurons_per_layer=mutation_maximum_new_neurons_per_layer, maximum_bias_connection_weight=config['maximum_bias_connection_weight'])
    
    config['random_sampling_technique'] = False
    config['random_weighting_technique'] = False
    
    return config


""" Ensemble configurations """


def get_config_simple_bagging_ensemble(base_learner, best_configuration, training_outer, testing, metric): 
    config = {}
    config['base_learner'] = _create_base_learner(base_learner, best_configuration, training_outer, testing, metric)
    config['number_learners'] = 30
    config['meta_learner'] = mean
    return config


def get_config_riw_ensemble(base_learner, best_configuration, training_outer, testing, metric):
    config = {} 
    config['base_learner'] = _create_base_learner(base_learner, best_configuration, training_outer, testing, metric)
    config['number_learners'] = 30
    config['meta_learner'] = mean
    config['weight_range'] = 1
    return config


def get_config_boosting_ensemble(base_learner, best_configuration, nr_ensemble, training_outer, testing, metric): 
    nr_ensemble = int(nr_ensemble)
    config = {} 
    config['base_learner'] = _create_base_learner(base_learner, best_configuration, training_outer, testing, metric)
    config['number_learners'] = 30
    if (nr_ensemble == 1 or nr_ensemble == 2):
        config['meta_learner'] = median
    else:  # 3 or 4
        config['meta_learner'] = mean
    if (nr_ensemble == 1 or nr_ensemble == 3):
        config['learning_rate'] = 1
    else: 
        config['learning_rate'] = 'random'
    return config 


def _create_base_learner(algorithm, configurations, training_outer, testing, metric):
    return algorithm(**configurations)


""" MLP configurations """ 


def get_config_mlp_lbfgs():
    config = {}
    config['solver'] = 'lbfgs'
    
    activation = random.randint(0, 2)
    if activation == 0:
        config['activation'] = 'logistic'
    elif activation == 1: 
        config['activation'] = 'tanh'
    else: 
        config['activation'] = 'relu'
    
    nr_hidden_layers = random.randint(1, 5)
    neurons = [random.randint(1, 200) for x in range(nr_hidden_layers)]
    config['hidden_layer_sizes'] = tuple(neurons)
    
    #===========================================================================
    # alpha = random.randint(0, 1)
    # if alpha == 0:
    #     config['alpha'] = 0
    # else: 
    #     config['alpha'] = random.uniform(0.00001, 10)
    #===========================================================================
    
    config['alpha'] = random.uniform(0.1, 10)
    
    config['max_iter'] = random.randint(1, 200)
    
    return config


def get_config_mlp_sgd(nr_instances):
    config = {}
    config['solver'] = 'sgd'
    config['learning_rate'] = 'constant'
    config['learning_rate_init'] = random.uniform(0.001, 1)
    
    activation = random.randint(0, 2)
    if activation == 0:
        config['activation'] = 'logistic'
    elif activation == 1: 
        config['activation'] = 'tanh'
    else: 
        config['activation'] = 'relu'
    
    nr_hidden_layers = random.randint(1, 5)
    neurons = [random.randint(1, 200) for x in range(nr_hidden_layers)]
    config['hidden_layer_sizes'] = tuple(neurons)
    
    #===========================================================================
    # alpha = random.randint(0, 1)
    # if alpha == 0:
    #     config['alpha'] = 0
    # else: 
    #     config['alpha'] = random.uniform(0.00001, 10)
    #===========================================================================
    
    config['alpha'] = random.uniform(0.1, 10)
    
    config['max_iter'] = random.randint(1, 200)
    
    config['batch_size'] = random.randint(50, nr_instances)
    
    shuffle_option = random.randint(0, 1)
    if shuffle_option == 0:
        config['shuffle'] = False
    else: 
        config['shuffle'] = True
    
    config['momentum'] = random.uniform(10 ** -7, 1)
    
    nesterov = random.randint(0, 1)
    if nesterov == 0:
        config['nesterovs_momentum'] = False
    else:
        config['nesterovs_momentum'] = True

    return config


def get_config_mlp_adam(nr_instances):
    config = {}
    config['solver'] = 'adam'
    config['learning_rate_init'] = random.uniform(0.001, 1)
    
    activation = random.randint(0, 2)
    if activation == 0:
        config['activation'] = 'logistic'
    elif activation == 1: 
        config['activation'] = 'tanh'
    else: 
        config['activation'] = 'relu'
    
    nr_hidden_layers = random.randint(1, 5)
    neurons = [random.randint(1, 200) for x in range(nr_hidden_layers)]
    config['hidden_layer_sizes'] = tuple(neurons)
    
    #===========================================================================
    # alpha = random.randint(0, 1)
    # if alpha == 0:
    #     config['alpha'] = 0
    # else: 
    #     config['alpha'] = random.uniform(0.00001, 10)
    #===========================================================================
    
    config['alpha'] = random.uniform(0.1, 10)
        
    config['max_iter'] = random.randint(1, 200)
    
    config['batch_size'] = random.randint(50, nr_instances)
    
    shuffle_option = random.randint(0, 1)
    if shuffle_option == 0:
        config['shuffle'] = False
    else: 
        config['shuffle'] = True
    
    config['beta_1'] = random.uniform(0, 1 - 10 ** -7)
    config['beta_2'] = random.uniform(0, 1 - 10 ** -7)
    
    return config

"""



_BASE_PARAMETERS = {
    'number_generations_ols': 20, #changed from 200
    'number_generations_fls': 100, 
    'population_size': 100
}

iterations_ols = list(range(1, 200 + 1, 1))
stopping_criterion_ols = [MaxGenerationsCriterion(i) for i in iterations_ols]

iterations_fls = list(range(1, 200 + 1, 1))
stopping_criterion_fls = [MaxGenerationsCriterion(i) for i in iterations_fls]

fixed_learning_steps = list(linspace(0.00001, 2, 1000))
subset_ratios = list(linspace(0.01, 0.99, 99))



_SLM_FLS_PARAMETERS = {
    'stopping_criterion': stopping_criterion_fls,
    'population_size': [10],
    'layers': [1, 2, 3, 4, 5],
    'learning_step': fixed_learning_steps,
    'mutation_operator': [Mutation2()],
    'random_sampling_technique': [False],
    'random_weighting_technique': [False]
}

_SLM_OLS_EDV_PARAMETERS = { 
    'stopping_criterion': [ErrorDeviationVariationCriterion(0.25)], 
    'population_size': [100],
    'layers': [1, 2, 3, 4, 5],
    'learning_step': ['optimized'], 
    'mutation_operator': [Mutation2()],
    'random_sampling_technique': [False],
    'random_weighting_technique': [False]
}

_SLM_FLS_TIE_PARAMETERS = {
    'stopping_criterion': [TrainingImprovementEffectivenessCriterion(0.25)], 
    'population_size': [100],
    'layers': [1, 2, 3, 4, 5],
    'learning_step': fixed_learning_steps,
    'mutation_operator': [Mutation2()],
    'random_sampling_technique': [False],
    'random_weighting_technique': [False]
}

_SLM_FLS_EDV_PARAMETERS = { 
    'stopping_criterion': [ErrorDeviationVariationCriterion(0.25)], 
    'population_size': [100],
    'layers': [1, 2, 3, 4, 5],
    'learning_step': fixed_learning_steps, 
    'mutation_operator': [Mutation2()],
    'random_sampling_technique': [False],
    'random_weighting_technique': [False]
}



#SLM OLS with Random Weighting Technique
_SLM_OLS_RWT_PARAMETERS = {
    'stopping_criterion': stopping_criterion_ols, 
    'population_size': [10],
    'layers': [1, 2, 3, 4, 5],
    'learning_step': ['optimized'],
    'mutation_operator': [Mutation2()],
    'random_sampling_technique': [False],
    'random_weighting_technique': [True],
    'weight_range' : [1, 2] 
}

#SLM FLS with Random Sampling Technique 
_SLM_FLS_RST_PARAMETERS = { 
    'stopping_criterion': stopping_criterion_fls, 
    'population_size': [10],
    'layers': [1, 2, 3, 4, 5],
    'learning_step': fixed_learning_steps,
    'mutation_operator': [Mutation2()],
    'random_sampling_technique': [True],
    'random_weighting_technique': [False],
    'subset_ratio': subset_ratios 
}

#SLM FLS with Random Weighting Technique 
_SLM_FLS_RWT_PARAMETERS = { 
    'stopping_criterion': stopping_criterion_fls, 
    'population_size': [10],
    'layers': [1, 2, 3, 4, 5],
    'learning_step': fixed_learning_steps,
    'mutation_operator': [Mutation2()],
    'random_sampling_technique': [False],
    'random_weighting_technique': [True],
    'weight_range': [1, 2]
}

_NEAT_PARAMETERS = {
    'stopping_criterion': [MaxGenerationsCriterion(_BASE_PARAMETERS.get(('number_generations')))],
    'population_size': [_BASE_PARAMETERS.get('population_size')],
    'compatibility_threshold': [3, 4],
    'compatibility_disjoint_coefficient': [1],
    'compatibility_weight_coefficient': [1],
    'conn_add_prob': [0.1, 0.25],
    'conn_delete_prob': [0.1],
    'node_add_prob': [0.1, 0.25],
    'node_delete_prob': [0.1],
    'weight_mutate_rate': [0.25],
    'weight_mutate_power': [0.25]
}

_FTNE_PARAMETERS = {
    'stopping_criterion': [MaxGenerationsCriterion(_BASE_PARAMETERS.get('number_generations'))],
    'population_size': [_BASE_PARAMETERS.get('population_size')],
    'topology': [create_network_from_topology(topology) for topology in [[1], [2], [2, 2], [3, 3, 3], [5, 5, 5]]],
    'selection_operator': [SelectionOperatorTournament(5)],
    'mutation_operator': [MutationOperatorGaussian(0.01), MutationOperatorGaussian(0.1)],
    'crossover_operator': [CrossoverOperatorArithmetic()],
    'mutation_rate': [0.25, 0.5],
    'crossover_rate': [0.01, 0.1]
}

_SVM_PARAMETERS = {
    'C': [c / 10 for c in range(1, 11)],
    'kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
    'epsilon': [e / 10 for e in range(1, 6)],
    'degree': [d for d in range(1, 5)],
    'gamma': [g / 10 for g in range(1, 6)],
    'coef0': [co / 10 for co in range(1, 6)],
    'probability': [True]
}

_MLP_PARAMETERS = {
    'hidden_layer_sizes': [(1), (2), (2, 2), (3, 3, 3), (5, 5, 5), (2, 2, 2, 2)],
    'activation': ['identity', 'logistic', 'tanh', 'relu'],
    'alpha': [10 ** -x for x in range(1, 7)],
    'learning_rate_init': [10 ** -x for x in range(1, 7)]
}

_RF_PARAMETERS = {
    'n_estimators': [25],
    'max_depth': [1, 2, 5, None],
    'min_samples_split': [0.01, 0.02, 0.05]
}



def _create_svc_configuration_list(list_dict):
    configuration_list = list()
    for kernel in list_dict.get('kernel'):
        if kernel == 'linear':
            keys = []
        if kernel == 'poly':
            keys = ['degree', 'gamma', 'coef0']
        if kernel == 'rbf':
            keys = ['gamma']
        if kernel == 'sigmoid':
            keys = ['gamma', 'coef0']
        keys.append('C')
        keys.append('probability')
        sub_dict = {k: list_dict[k] for k in keys if k in list_dict}
        sub_dict['kernel'] = [kernel]
        configuration_list.extend(_create_configuration_list(sub_dict))
    return configuration_list

def _create_svr_configuration_list(list_dict):
    configuration_list = list()
    for kernel in list_dict.get('kernel'):
        if kernel == 'linear':
            keys = []
        if kernel == 'poly':
            keys = ['degree', 'gamma', 'coef0']
        if kernel == 'rbf':
            keys = ['gamma']
        if kernel == 'sigmoid':
            keys = ['gamma', 'coef0']
        keys.append('C')
        keys.append('epsilon')
        sub_dict = {k: list_dict[k] for k in keys if k in list_dict}
        sub_dict['kernel'] = [kernel]
        configuration_list.extend(_create_configuration_list(sub_dict))
    return configuration_list

SLM_FLS_CONFIGURATIONS = _create_configuration_list(_SLM_FLS_PARAMETERS)
SLM_OLS_CONFIGURATIONS = _create_configuration_list(_SLM_OLS_PARAMETERS)

SLM_OLS_EDV_CONFIGURATIONS = _create_configuration_list(_SLM_OLS_EDV_PARAMETERS)
SLM_FLS_TIE_CONFIGURATIONS = _create_configuration_list(_SLM_FLS_TIE_PARAMETERS)
SLM_FLS_EDV_CONFIGURATIONS = _create_configuration_list(_SLM_FLS_EDV_PARAMETERS)


SLM_OLS_RWT_CONFIGURATIONS = _create_configuration_list(_SLM_OLS_RWT_PARAMETERS)

SLM_FLS_RST_CONFIGURATIONS = _create_configuration_list(_SLM_FLS_RST_PARAMETERS)
SLM_FLS_RWT_CONFIGURATIONS = _create_configuration_list(_SLM_FLS_RWT_PARAMETERS)

NEAT_CONFIGURATIONS = _create_configuration_list(_NEAT_PARAMETERS)
FTNE_CONFIGURATIONS = _create_configuration_list(_FTNE_PARAMETERS)
SVC_CONFIGURATIONS = _create_svc_configuration_list(_SVM_PARAMETERS)
SVR_CONFIGURATIONS = _create_svr_configuration_list(_SVM_PARAMETERS)
MLP_CONFIGURATIONS = _create_configuration_list(_MLP_PARAMETERS)
RF_CONFIGURATIONS = _create_configuration_list(_RF_PARAMETERS)

"""

"""
def _create_configuration_list(list_dict):
    return [{k:v for k, v in zip(list_dict.keys(), configuration)}
            for configuration in list(product(*[list_dict[key] for key in list_dict.keys()]))]


def _create_base_learner(algorithm, configurations):
    return [algorithm(**configuration) for configuration in configurations]

#SLM OLS with Random Sampling Technique
_SLM_OLS_RST_PARAMETERS = { 
    'stopping_criterion': [MaxGenerationsCriterion(20)], 
    'population_size': [10],
    'layers': [1, 2, 3, 4, 5],
    'learning_step': ['optimized'],
    'mutation_operator': [Mutation2()],
    'random_sampling_technique': [True],
    'random_weighting_technique': [False],
    'subset_ratio': [0.25, 0.75]
}

_SLM_OLS_PARAMETERS = {
    'stopping_criterion': [MaxGenerationsCriterion(20)], 
    'population_size': [10],
    'layers': [1, 2, 3, 4, 5],
    'learning_step': ['optimized'],
    'mutation_operator': [Mutation2()],
    'random_sampling_technique': [False],
    'random_weighting_technique': [False]
}

SLM_OLS_CONFIGURATIONS = _create_configuration_list(_SLM_OLS_PARAMETERS)
SLM_OLS_RST_CONFIGURATIONS = _create_configuration_list(_SLM_OLS_RST_PARAMETERS)

_ENSEMBLE_PARAMETERS = {
    'base_learner': _create_base_learner(SemanticLearningMachine, SLM_OLS_CONFIGURATIONS),
    'number_learners': [25, 50, 75, 100],
    'meta_learner': [mean]
}

# _ENSEMBLE_FLS_PARAMETERS = { 
#     'base_learner': _create_base_learner(SemanticLearningMachine, SLM_FLS_CONFIGURATIONS),
#     'number_learners': [25, 50, 75, 100],
#     'meta_learner': [mean]
# }

_ENSEMBLE_RST_PARAMETERS = {
    'base_learner': _create_base_learner(BenchmarkSLM_RST, SLM_OLS_RST_CONFIGURATIONS),
    'number_learners': [25, 50, 75, 100],
    'meta_learner': [mean]
}

# _ENSEMBLE_RWT_PARAMETERS = {
#     'base_learner': _create_base_learner(BenchmarkSLM_RWT, SLM_OLS_RWT_CONFIGURATIONS),
#     'number_learners': [25, 50, 75, 100],
#     'meta_learner': [mean]
# }


ENSEMBLE_CONFIGURATIONS = _create_configuration_list(_ENSEMBLE_PARAMETERS)
# ENSEMBLE_FLS_CONFIGURATIONS = _create_configuration_list(_ENSEMBLE_FLS_PARAMETERS)

ENSEMBLE_RST_CONFIGURATIONS = _create_configuration_list(_ENSEMBLE_RST_PARAMETERS)
# ENSEMBLE_RWT_CONFIGURATIONS = _create_configuration_list(_ENSEMBLE_RWT_PARAMETERS)

# _ENSEMBLE_BAGGING_PARAMETERS = { 
#     'base_learner': _create_base_learner(SemanticLearningMachine, SLM_OLS_CONFIGURATIONS), 
#     'number_learners': [25, 50, 75, 100], 
#     'meta_learner': [mean]
# }

# _ENSEMBLE_BAGGING_FLS_PARAMETERS = { 
#     'base_learner': _create_base_learner(SemanticLearningMachine, SLM_FLS_CONFIGURATIONS), 
#     'number_learners': [25, 50, 75, 100], 
#     'meta_learner': [mean]
# }

# _ENSEMBLE_BAGGING_RST_PARAMETERS = { 
#     'base_learner': _create_base_learner(BenchmarkSLM_RST, SLM_OLS_RST_CONFIGURATIONS), 
#     'number_learners': [25, 50, 75, 100], 
#     'meta_learner': [mean]
# }

# _ENSEMBLE_BAGGING_RWT_PARAMETERS = { 
#     'base_learner': _create_base_learner(BenchmarkSLM_RWT, SLM_OLS_RWT_CONFIGURATIONS), 
#     'number_learners': [25, 50, 75, 100], 
#     'meta_learner': [mean]
# }

# _ENSEMBLE_RANDOM_INDEPENDENT_WEIGHTING_PARAMETERS = {
#     'base_learner': _create_base_learner(SemanticLearningMachine, SLM_OLS_CONFIGURATIONS),
#     'number_learners': [25, 50, 75, 100],
#     'meta_learner': [mean],
#     'weight_range': [1, 2]
# }

# _ENSEMBLE_RANDOM_INDEPENDENT_WEIGHTING_FLS_PARAMETERS = {
#     'base_learner': _create_base_learner(SemanticLearningMachine, SLM_FLS_CONFIGURATIONS),
#     'number_learners': [25, 50, 75, 100],
#     'meta_learner': [mean],
#     'weight_range': [1, 2]
# }

# _ENSEMBLE_BOOSTING_PARAMETERS = {
#     'base_learner': _create_base_learner(SemanticLearningMachine, SLM_OLS_CONFIGURATIONS),
#     'number_learners': [25, 50, 75, 100],
#     'meta_learner': [mean, median], 
#     'learning_rate': [1, 'random']
# }

# _ENSEMBLE_BOOSTING_FLS_PARAMETERS = {
#     'base_learner': _create_base_learner(SemanticLearningMachine, SLM_FLS_CONFIGURATIONS),
#     'number_learners': [25, 50, 75, 100],
#     'meta_learner': [mean, median], 
#     'learning_rate': [1, 'random']
# }

# ENSEMBLE_BAGGING_CONFIGURATIONS = _create_configuration_list(_ENSEMBLE_BAGGING_PARAMETERS)
# ENSEMBLE_BAGGING_FLS_CONFIGURATIONS = _create_configuration_list(_ENSEMBLE_BAGGING_FLS_PARAMETERS)

# ENSEMBLE_BAGGING_RST_CONFIGURATIONS = _create_configuration_list(_ENSEMBLE_BAGGING_RST_PARAMETERS)
# ENSEMBLE_BAGGING_RWT_CONFIGURATIONS = _create_configuration_list(_ENSEMBLE_BAGGING_RWT_PARAMETERS)

# ENSEMBLE_RANDOM_INDEPENDENT_WEIGHTING_CONFIGURATIONS = _create_configuration_list(_ENSEMBLE_RANDOM_INDEPENDENT_WEIGHTING_PARAMETERS)
# ENSEMBLE_RANDOM_INDEPENDENT_WEIGHTING_FLS_CONFIGURATIONS = _create_configuration_list(_ENSEMBLE_RANDOM_INDEPENDENT_WEIGHTING_FLS_PARAMETERS)

# ENSEMBLE_BOOSTING_CONFIGURATIONS = _create_configuration_list(_ENSEMBLE_BOOSTING_PARAMETERS)
# ENSEMBLE_BOOSTING_FLS_CONFIGURATIONS = _create_configuration_list(_ENSEMBLE_BOOSTING_FLS_PARAMETERS)

# """

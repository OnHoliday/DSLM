import random

# from algorithms.common.neural_network.neural_network import create_network_from_topology
from algorithms.common.stopping_criterion import MaxGenerationsCriterion, \
    ErrorDeviationVariationCriterion, TrainingImprovementEffectivenessCriterion
from algorithms.semantic_learning_machine.mutation_operator import Mutation4, Mutation_CNN_1
from algorithms.simple_genetic_algorithm.crossover_operator import CrossoverOperatorArithmetic
from algorithms.simple_genetic_algorithm.mutation_operator import MutationOperatorGaussian
from algorithms.simple_genetic_algorithm.selection_operator import SelectionOperatorTournament
from benchmark.evaluator import EvaluatorSLM, EvaluatorNEAT, EvaluatorFTNE, \
    EvaluatorMLPC, EvaluatorMLPR, EvaluatorDSLM

DEFAULT_NUMBER_OF_COMBINATIONS = 3

DEFAULT_POPULATION_SIZE = 5
DEFAULT_NUMBER_OF_ITERATIONS = 10


def generate_random_slm_bls_configuration(option=None, init_maximum_layers=5, maximum_iterations=10, maximum_learning_step=10, maximum_neuron_connection_weight=0.5, maximum_bias_connection_weight=1.0, mutation_maximum_new_neurons_per_layer=3):
    configuration = {}
    
    configuration['stopping_criterion'] = MaxGenerationsCriterion(random.randint(1, maximum_iterations))
    #===========================================================================
    # configuration['stopping_criterion'] = MaxGenerationsCriterion(DEFAULT_NUMBER_OF_ITERATIONS)
    #===========================================================================
    
    configuration['population_size'] = DEFAULT_POPULATION_SIZE
    
    configuration['layers'] = init_maximum_layers
    
    configuration['learning_step'] = random.uniform(0.001, maximum_learning_step)
   
    configuration['maximum_neuron_connection_weight'] = random.uniform(0.1, maximum_neuron_connection_weight)
    configuration['maximum_bias_connection_weight'] = random.uniform(0.1, maximum_bias_connection_weight)
    
    configuration['mutation_operator'] = Mutation4(maximum_new_neurons_per_layer=mutation_maximum_new_neurons_per_layer, maximum_bias_connection_weight=configuration['maximum_bias_connection_weight'])
    
    configuration['random_sampling_technique'] = False
    configuration['random_weighting_technique'] = False
    
    configuration['protected_ols'] = False
    
    configuration['bootstrap_ols'] = False
    
    configuration['store_ls_history'] = True
    
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
    
    return configuration


def generate_random_slm_bls_configuration_training(option=None, init_maximum_layers=5, maximum_iterations=10, maximum_learning_step=100, maximum_neuron_connection_weight=0.5, maximum_bias_connection_weight=1.0, mutation_maximum_new_neurons_per_layer=3):
    configuration = generate_random_slm_bls_configuration(init_maximum_layers=init_maximum_layers, maximum_iterations=maximum_iterations, maximum_learning_step=maximum_learning_step, maximum_neuron_connection_weight=maximum_neuron_connection_weight, maximum_bias_connection_weight=maximum_bias_connection_weight, mutation_maximum_new_neurons_per_layer=mutation_maximum_new_neurons_per_layer)
    configuration['learning_step'] = random.uniform(0.1, maximum_learning_step)
    return configuration


def generate_random_dslm_ols_configuration(option=None, init_maximum_layers=3, maximum_iterations=30,
                                          maximum_neuron_connection_weight=0.5, maximum_bias_connection_weight=1.0,
                                          mutation_maximum_new_neurons_per_layer=3,
                                          init_maximum_cnn_neuron_per_layer=3, init_maximum_cnn_layers=3):
    configuration = {}

    configuration['stopping_criterion'] = MaxGenerationsCriterion(random.randint(1, maximum_iterations))
    configuration['population_size'] = DEFAULT_POPULATION_SIZE

    configuration['layers'] = init_maximum_layers
    configuration['cnn_layers'] = init_maximum_cnn_layers
    configuration['cnn_neurons_per_layer'] = init_maximum_cnn_neuron_per_layer

    configuration['learning_step'] = 'optimized'

    configuration['maximum_neuron_connection_weight'] = random.uniform(0.1, maximum_neuron_connection_weight)
    configuration['maximum_bias_connection_weight'] = random.uniform(0.1, maximum_bias_connection_weight)

    # configuration['mutation_operator'] = Mutation4(maximum_new_neurons_per_layer=mutation_maximum_new_neurons_per_layer,
    #                                                maximum_bias_connection_weight=configuration[
    #                                                    'maximum_bias_connection_weight'])
    configuration['mutation_operator'] = Mutation_CNN_1(maximum_new_cnn_neurons_per_layer=1)
    configuration['random_sampling_technique'] = False
    configuration['random_weighting_technique'] = False
    configuration['protected_ols'] = False
    configuration['bootstrap_ols'] = False
    configuration['store_ls_history'] = True

    return configuration

def generate_random_slm_ols_configuration(option=None, init_maximum_layers=3, maximum_iterations=30, maximum_neuron_connection_weight=0.5, maximum_bias_connection_weight=1.0, mutation_maximum_new_neurons_per_layer=3):
#def generate_random_slm_ols_configuration(option=None, init_maximum_layers=5, maximum_iterations=100, maximum_neuron_connection_weight=0.5, maximum_bias_connection_weight=1.0, mutation_maximum_new_neurons_per_layer=3): 
    configuration = {}
    
    configuration['stopping_criterion'] = MaxGenerationsCriterion(random.randint(1, maximum_iterations))
    #===========================================================================
    # configuration['stopping_criterion'] = MaxGenerationsCriterion(maximum_iterations)
    #===========================================================================
    
    configuration['population_size'] = DEFAULT_POPULATION_SIZE
    
    configuration['layers'] = init_maximum_layers
    
    configuration['learning_step'] = 'optimized'
    
    configuration['maximum_neuron_connection_weight'] = random.uniform(0.1, maximum_neuron_connection_weight)
    configuration['maximum_bias_connection_weight'] = random.uniform(0.1, maximum_bias_connection_weight)
    
    configuration['mutation_operator'] = Mutation4(maximum_new_neurons_per_layer=mutation_maximum_new_neurons_per_layer, maximum_bias_connection_weight=configuration['maximum_bias_connection_weight'])
    
    configuration['random_sampling_technique'] = False
    configuration['random_weighting_technique'] = False
    
    configuration['protected_ols'] = False
    
    configuration['bootstrap_ols'] = False
    
    configuration['store_ls_history'] = True
    
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
    
    return configuration


def generate_random_slm_lr_ls_configuration(init_maximum_layers=5, maximum_iterations=100, maximum_neuron_connection_weight=0.5, maximum_bias_connection_weight=1.0, mutation_maximum_new_neurons_per_layer=3):
    configuration = generate_random_slm_ols_configuration(init_maximum_layers=init_maximum_layers, maximum_iterations=maximum_iterations, maximum_neuron_connection_weight=maximum_neuron_connection_weight, maximum_bias_connection_weight=maximum_bias_connection_weight, mutation_maximum_new_neurons_per_layer=mutation_maximum_new_neurons_per_layer)
    configuration['learning_step'] = 'lr-ls'
    return configuration


def generate_random_1_1_slm_ols_configuration(init_maximum_layers=5, maximum_iterations=100, maximum_neuron_connection_weight=0.5, maximum_bias_connection_weight=1.0, mutation_maximum_new_neurons_per_layer=3):
    configuration = generate_random_slm_ols_configuration(init_maximum_layers=init_maximum_layers, maximum_iterations=maximum_iterations, maximum_neuron_connection_weight=maximum_neuron_connection_weight, maximum_bias_connection_weight=maximum_bias_connection_weight, mutation_maximum_new_neurons_per_layer=mutation_maximum_new_neurons_per_layer)
    configuration['population_size'] = 1
    return configuration


def generate_random_1_5_slm_ols_configuration(init_maximum_layers=5, maximum_iterations=100, maximum_neuron_connection_weight=0.5, maximum_bias_connection_weight=1.0, mutation_maximum_new_neurons_per_layer=3):
    configuration = generate_random_slm_ols_configuration(init_maximum_layers=init_maximum_layers, maximum_iterations=maximum_iterations, maximum_neuron_connection_weight=maximum_neuron_connection_weight, maximum_bias_connection_weight=maximum_bias_connection_weight, mutation_maximum_new_neurons_per_layer=mutation_maximum_new_neurons_per_layer)
    configuration['population_size'] = 5
    return configuration

def generate_random_1_10_slm_ols_configuration(init_maximum_layers=5, maximum_iterations=250, maximum_neuron_connection_weight=0.5, maximum_bias_connection_weight=1.0, mutation_maximum_new_neurons_per_layer=3):
#def generate_random_1_10_slm_ols_configuration(init_maximum_layers=5, maximum_iterations=100, maximum_neuron_connection_weight=0.5, maximum_bias_connection_weight=1.0, mutation_maximum_new_neurons_per_layer=3):
    configuration = generate_random_slm_ols_configuration(init_maximum_layers=init_maximum_layers, maximum_iterations=maximum_iterations, maximum_neuron_connection_weight=maximum_neuron_connection_weight, maximum_bias_connection_weight=maximum_bias_connection_weight, mutation_maximum_new_neurons_per_layer=mutation_maximum_new_neurons_per_layer)
    configuration['population_size'] = 10
    
    #===========================================================================
    # configuration['stopping_criterion'] = MaxGenerationsCriterion(10000)
    #===========================================================================
    configuration['stopping_criterion'] = MaxGenerationsCriterion(random.randint(250, 1000))
    configuration['random_weighting_technique'] = True
    configuration['weight_range'] = 1
    
    return configuration


def generate_random_1_1_slm_lr_ls_configuration(init_maximum_layers=5, maximum_iterations=100, maximum_neuron_connection_weight=0.5, maximum_bias_connection_weight=1.0, mutation_maximum_new_neurons_per_layer=3):
    configuration = generate_random_slm_lr_ls_configuration(init_maximum_layers=init_maximum_layers, maximum_iterations=maximum_iterations, maximum_neuron_connection_weight=maximum_neuron_connection_weight, maximum_bias_connection_weight=maximum_bias_connection_weight, mutation_maximum_new_neurons_per_layer=mutation_maximum_new_neurons_per_layer)
    configuration['population_size'] = 1
    return configuration


def generate_random_1_5_slm_lr_ls_configuration(init_maximum_layers=5, maximum_iterations=100, maximum_neuron_connection_weight=0.5, maximum_bias_connection_weight=1.0, mutation_maximum_new_neurons_per_layer=3):
    configuration = generate_random_slm_lr_ls_configuration(init_maximum_layers=init_maximum_layers, maximum_iterations=maximum_iterations, maximum_neuron_connection_weight=maximum_neuron_connection_weight, maximum_bias_connection_weight=maximum_bias_connection_weight, mutation_maximum_new_neurons_per_layer=mutation_maximum_new_neurons_per_layer)
    configuration['population_size'] = 5
    return configuration


def generate_random_1_10_slm_lr_ls_configuration(init_maximum_layers=5, maximum_iterations=100, maximum_neuron_connection_weight=0.5, maximum_bias_connection_weight=1.0, mutation_maximum_new_neurons_per_layer=3):
    configuration = generate_random_slm_lr_ls_configuration(init_maximum_layers=init_maximum_layers, maximum_iterations=maximum_iterations, maximum_neuron_connection_weight=maximum_neuron_connection_weight, maximum_bias_connection_weight=maximum_bias_connection_weight, mutation_maximum_new_neurons_per_layer=mutation_maximum_new_neurons_per_layer)
    configuration['population_size'] = 10
    return configuration


def generate_random_slm_protected_ols_configuration(init_maximum_layers=5, maximum_iterations=100, maximum_neuron_connection_weight=0.5, maximum_bias_connection_weight=1.0, mutation_maximum_new_neurons_per_layer=3):
    configuration = generate_random_slm_ols_configuration(init_maximum_layers=init_maximum_layers, maximum_iterations=maximum_iterations, maximum_neuron_connection_weight=maximum_neuron_connection_weight, maximum_bias_connection_weight=maximum_bias_connection_weight, mutation_maximum_new_neurons_per_layer=mutation_maximum_new_neurons_per_layer)
    configuration['protected_ols'] = True
    return configuration


def generate_random_slm_bootstrap_ols_configuration(init_maximum_layers=5, maximum_iterations=100, maximum_neuron_connection_weight=0.5, maximum_bias_connection_weight=1.0, mutation_maximum_new_neurons_per_layer=3):
    configuration = generate_random_slm_ols_configuration(init_maximum_layers=init_maximum_layers, maximum_iterations=maximum_iterations, maximum_neuron_connection_weight=maximum_neuron_connection_weight, maximum_bias_connection_weight=maximum_bias_connection_weight, mutation_maximum_new_neurons_per_layer=mutation_maximum_new_neurons_per_layer)
    configuration['bootstrap_ols'] = True
    configuration['bootstrap_ols_samples'] = 10
    #===========================================================================
    # configuration['bootstrap_ols_samples'] = 30
    #===========================================================================
    #===========================================================================
    # configuration['high_absolute_ls_difference'] = 1
    #===========================================================================
    return configuration


def generate_random_slm_bootstrap_ols_median_configuration(init_maximum_layers=5, maximum_iterations=100, maximum_neuron_connection_weight=0.5, maximum_bias_connection_weight=1.0, mutation_maximum_new_neurons_per_layer=3):
    configuration = generate_random_slm_bootstrap_ols_configuration(init_maximum_layers=init_maximum_layers, maximum_iterations=maximum_iterations, maximum_neuron_connection_weight=maximum_neuron_connection_weight, maximum_bias_connection_weight=maximum_bias_connection_weight, mutation_maximum_new_neurons_per_layer=mutation_maximum_new_neurons_per_layer)
    configuration['bootstrap_ols_criterion'] = 'median'
    return configuration

    
def generate_random_slm_bootstrap_ols_mean_configuration(init_maximum_layers=5, maximum_iterations=100, maximum_neuron_connection_weight=0.5, maximum_bias_connection_weight=1.0, mutation_maximum_new_neurons_per_layer=3):
    configuration = generate_random_slm_bootstrap_ols_configuration(init_maximum_layers=init_maximum_layers, maximum_iterations=maximum_iterations, maximum_neuron_connection_weight=maximum_neuron_connection_weight, maximum_bias_connection_weight=maximum_bias_connection_weight, mutation_maximum_new_neurons_per_layer=mutation_maximum_new_neurons_per_layer)
    configuration['bootstrap_ols_criterion'] = 'mean'
    return configuration


def generate_random_slm_bls_tie_edv_configuration(init_maximum_layers=5, maximum_iterations=100, maximum_learning_step=10, maximum_neuron_connection_weight=0.5, maximum_bias_connection_weight=1.0, mutation_maximum_new_neurons_per_layer=3):
    configuration = generate_random_slm_bls_configuration(init_maximum_layers=init_maximum_layers, maximum_iterations=maximum_iterations, maximum_learning_step=maximum_learning_step, maximum_neuron_connection_weight=maximum_neuron_connection_weight, maximum_bias_connection_weight=maximum_bias_connection_weight, mutation_maximum_new_neurons_per_layer=mutation_maximum_new_neurons_per_layer)
    
    stopping_crit = random.randint(1, 2)
    # EDV
    if stopping_crit == 1:
        configuration['stopping_criterion'] = ErrorDeviationVariationCriterion(maximum_iterations=maximum_iterations)
    # TIE
    else:
        configuration['stopping_criterion'] = TrainingImprovementEffectivenessCriterion(maximum_iterations=maximum_iterations)
    
    configuration['population_size'] = 100
    
    return configuration

def generate_random_slm_ols_edv_configuration(init_maximum_layers=5, maximum_iterations=250, maximum_neuron_connection_weight=0.5, maximum_bias_connection_weight=1.0, mutation_maximum_new_neurons_per_layer=3):
#def generate_random_slm_ols_edv_configuration(init_maximum_layers=5, maximum_iterations=100, maximum_neuron_connection_weight=0.5, maximum_bias_connection_weight=1.0, mutation_maximum_new_neurons_per_layer=3):
    configuration = generate_random_slm_ols_configuration(init_maximum_layers=init_maximum_layers, maximum_iterations=maximum_iterations, maximum_neuron_connection_weight=maximum_neuron_connection_weight, maximum_bias_connection_weight=maximum_bias_connection_weight, mutation_maximum_new_neurons_per_layer=mutation_maximum_new_neurons_per_layer)
    
    configuration['stopping_criterion'] = ErrorDeviationVariationCriterion(maximum_iterations=maximum_iterations)
    
    configuration['population_size'] = 100
    
    return configuration


def generate_random_neat_configuration(maximum_iterations=100):
    configuration = {}
    
    """ https://neat-python.readthedocs.io/en/latest/config_file.html """
    
    configuration['stopping_criterion'] = MaxGenerationsCriterion(random.randint(1, maximum_iterations))
    #===========================================================================
    # configuration['stopping_criterion'] = MaxGenerationsCriterion(DEFAULT_NUMBER_OF_ITERATIONS)
    #===========================================================================
    
    configuration['population_size'] = DEFAULT_POPULATION_SIZE
    
    # Jan: [3, 4]
    """" Individuals whose genomic distance is less than this threshold are considered to be in the same species """ 
    configuration['compatibility_threshold'] = random.randint(1, 10)
    
    # Jan: [1]
    """ The coefficient for the disjoint and excess gene counts’ contribution to the genomic distance. """
    configuration['compatibility_disjoint_coefficient'] = random.uniform(0.001, 1)
    
    # Jan: [1]
    """ The coefficient for each weight, bias, or response multiplier difference’s contribution to the genomic distance (for homologous nodes or connections). This is also used as the value to add for differences in activation functions, aggregation functions, or enabled/disabled status. """
    configuration['compatibility_weight_coefficient'] = random.uniform(0.001, 1)
    
    # Jan: [0.1, 0.25]
    """ The probability that mutation will add a connection between existing nodes. Valid values are in [0.0, 1.0]. """
    configuration['conn_add_prob'] = random.uniform(0.001, 1)
    
    # Jan: [0.1]
    """ The probability that mutation will delete an existing connection. Valid values are in [0.0, 1.0]. """
    configuration['conn_delete_prob'] = random.uniform(0.001, 1)
    
    # Jan: [0.1, 0.25]
    """ The probability that mutation will add a new node (essentially replacing an existing connection, the enabled status of which will be set to False). Valid values are in [0.0, 1.0]. """
    configuration['node_add_prob'] = random.uniform(0.001, 1)
    
    # Jan: [0.1]
    """ The probability that mutation will delete an existing node (and all connections to it). Valid values are in [0.0, 1.0]. """
    configuration['node_delete_prob'] = random.uniform(0.001, 1)
    
    # Jan: [0.25]
    """ The probability that mutation will change the weight of a connection by adding a random value. """
    configuration['weight_mutate_rate'] = random.uniform(0.001, 1)
    
    # Jan: [0.25]
    """ The standard deviation of the zero-centered normal/gaussian distribution from which a weight value mutation is drawn. """
    configuration['weight_mutate_power'] = random.uniform(0.001, 5)
    
    return configuration


def generate_random_ftne_configuration(maximum_iterations=100, maximum_number_of_layers=5, maximum_neurons_per_layer=5):
    configuration = {}
    
    configuration['stopping_criterion'] = MaxGenerationsCriterion(random.randint(1, maximum_iterations))
    #===========================================================================
    # configuration['stopping_criterion'] = MaxGenerationsCriterion(DEFAULT_NUMBER_OF_ITERATIONS)
    #===========================================================================
    
    configuration['population_size'] = DEFAULT_POPULATION_SIZE
    
    # Jan: [create_network_from_topology(topology) for topology in [[1], [2], [2, 2], [3, 3, 3], [5, 5, 5]]]
    number_of_layers = random.randint(1, maximum_number_of_layers)
    neurons_per_layer = [random.randint(1, maximum_neurons_per_layer) for i in range(number_of_layers)]
    configuration['topology'] = create_network_from_topology(neurons_per_layer)
    
    # Jan: [SelectionOperatorTournament(5)]
    tournament_size = random.randint(1, configuration['population_size'])
    configuration['selection_operator'] = SelectionOperatorTournament(tournament_size)
     
    # Jan: [MutationOperatorGaussian(0.01), MutationOperatorGaussian(0.1)]
    standard_deviation = random.uniform(0.001, 5)
    configuration['mutation_operator'] = MutationOperatorGaussian(standard_deviation)
     
    # Jan: [CrossoverOperatorArithmetic()]
    configuration['crossover_operator'] = CrossoverOperatorArithmetic()
     
    # Jan: [0.25, 0.5]
    configuration['mutation_rate'] = random.uniform(0.001, 1)
     
    # Jan: [0.01, 0.1]
    configuration['crossover_rate'] = random.uniform(0.001, 1)
    
    return configuration


def generate_random_sgd_configuration(nr_instances):
    configuration = {}
    configuration['solver'] = 'sgd'
    configuration['learning_rate'] = 'constant'
    configuration['learning_rate_init'] = random.uniform(0.001, 10)
    
    activation = random.randint(0, 2)
    if activation == 0:
        configuration['activation'] = 'logistic'
    elif activation == 1: 
        configuration['activation'] = 'tanh'
    else: 
        configuration['activation'] = 'relu'
    
    nr_hidden_layers = random.randint(1, 5)
    neurons = [random.randint(1, 200) for x in range(nr_hidden_layers)]
    configuration['hidden_layer_sizes'] = tuple(neurons)
    
    #===========================================================================
    # alpha = random.randint(0, 1)
    # if alpha == 0:
    #     configuration['alpha'] = 0
    # else: 
    #     configuration['alpha'] = random.uniform(0.00001, 10)
    #===========================================================================
    
    configuration['alpha'] = random.uniform(0.1, 10)
    
    configuration['max_iter'] = random.randint(1, 100)
    
    configuration['batch_size'] = random.randint(50, nr_instances)
    
    shuffle_option = random.randint(0, 1)
    if shuffle_option == 0:
        configuration['shuffle'] = False
    else: 
        configuration['shuffle'] = True
    
    configuration['momentum'] = random.uniform(10 ** -7, 1)
    
    nesterov = random.randint(0, 1)
    if nesterov == 0:
        configuration['nesterovs_momentum'] = False
    else:
        configuration['nesterovs_momentum'] = True

    return configuration


def generate_random_adam_configuration(nr_instances):
    configuration = {}
    configuration['solver'] = 'adam'
    configuration['learning_rate_init'] = random.uniform(0.001, 10)
    
    activation = random.randint(0, 2)
    if activation == 0:
        configuration['activation'] = 'logistic'
    elif activation == 1: 
        configuration['activation'] = 'tanh'
    else: 
        configuration['activation'] = 'relu'
    
    nr_hidden_layers = random.randint(1, 5)
    neurons = [random.randint(1, 200) for x in range(nr_hidden_layers)]
    configuration['hidden_layer_sizes'] = tuple(neurons)
    
    #===========================================================================
    # alpha = random.randint(0, 1)
    # if alpha == 0:
    #     configuration['alpha'] = 0
    # else: 
    #     configuration['alpha'] = random.uniform(0.00001, 10)
    #===========================================================================
    
    configuration['alpha'] = random.uniform(0.1, 10)
        
    configuration['max_iter'] = random.randint(1, 100)
    
    configuration['batch_size'] = random.randint(50, nr_instances)
    
    shuffle_option = random.randint(0, 1)
    if shuffle_option == 0:
        configuration['shuffle'] = False
    else: 
        configuration['shuffle'] = True
    
    configuration['beta_1'] = random.uniform(0, 1 - 10 ** -7)
    configuration['beta_2'] = random.uniform(0, 1 - 10 ** -7)
    
    return configuration


def generate_random_mlp_configuration(nr_instances):
    solver_option = random.randint(0, 1)
    if solver_option == 0:
        configuration = generate_random_sgd_configuration(nr_instances)
    else:
        configuration = generate_random_adam_configuration(nr_instances)
    return configuration


def generate_random_mlp_configuration_training(nr_instances):
    configuration = generate_random_mlp_configuration(nr_instances)
    configuration['learning_rate_init'] = random.uniform(0.1, 100)
    configuration['alpha'] = 0
    return configuration

DSLM_OLS = {
    'dslm_ols': {
        'name_long': 'Deep Semantic Learning Machine - Optimized Learning Step',
        'name_short': 'DSLM-OLS',
        'algorithms': [EvaluatorDSLM],
        'configuration_method': generate_random_dslm_ols_configuration,
        'max_combinations': DEFAULT_NUMBER_OF_COMBINATIONS
    }
}

SLM_BLS = {
    'slm_bls': {
        'name_long': 'Semantic Learning Machine - Bounded Learning Step',
        'name_short': 'SLM-BLS',
        'algorithms': [EvaluatorSLM],
        'configuration_method': generate_random_slm_bls_configuration,
        'max_combinations': DEFAULT_NUMBER_OF_COMBINATIONS
    }
}

SLM_BLS_TRAINING = {
    'slm_bls_training': {
        'name_long': 'Semantic Learning Machine - Bounded Learning Step',
        'name_short': 'SLM-BLS',
        'algorithms': [EvaluatorSLM],
        'configuration_method': generate_random_slm_bls_configuration_training,
        'max_combinations': DEFAULT_NUMBER_OF_COMBINATIONS
    }
}

SLM_OLS = {
    'slm_ols': {
        'name_long': 'Semantic Learning Machine - Optimized Learning Step',
        'name_short': 'SLM-OLS',
        'algorithms': [EvaluatorSLM],
        'configuration_method': generate_random_slm_ols_configuration,
        'max_combinations': DEFAULT_NUMBER_OF_COMBINATIONS
    }
}

SLM_LR_LS = {
    'slm_lr_ls': {
        'name_long': 'Semantic Learning Machine - Linear Regression Learning Step',
        'name_short': 'SLM-LR-LS',
        'algorithms': [EvaluatorSLM],
        'configuration_method': generate_random_slm_lr_ls_configuration,
        'max_combinations': DEFAULT_NUMBER_OF_COMBINATIONS
    }
}

SLM_OLS_1_1 = {
    'slm_ols_1_1': {
        'name_long': '(1 + 1)-Semantic Learning Machine - Optimized Learning Step',
        'name_short': '(1 + 1)-SLM-OLS',
        'algorithms': [EvaluatorSLM],
        'configuration_method': generate_random_1_1_slm_ols_configuration,
        'max_combinations': DEFAULT_NUMBER_OF_COMBINATIONS
    }
}

SLM_OLS_1_5 = {
    'slm_ols_1_5': {
        'name_long': '(1 + 5)-Semantic Learning Machine - Optimized Learning Step',
        'name_short': '(1 + 5)-SLM-OLS',
        'algorithms': [EvaluatorSLM],
        'configuration_method': generate_random_1_5_slm_ols_configuration,
        'max_combinations': DEFAULT_NUMBER_OF_COMBINATIONS
    }
}

SLM_OLS_1_10 = {
    'slm_ols_1_10': {
        'name_long': '(1 + 10)-Semantic Learning Machine - Optimized Learning Step',
        'name_short': '(1 + 10)-SLM-OLS',
        'algorithms': [EvaluatorSLM],
        'configuration_method': generate_random_1_10_slm_ols_configuration,
        'max_combinations': DEFAULT_NUMBER_OF_COMBINATIONS
    }
}

SLM_LR_LS_1_1 = {
    'slm_lr_ls_1_1': {
        'name_long': '(1 + 1)-Semantic Learning Machine - Linear Regression Learning Step',
        'name_short': '(1 + 1)-SLM-LR-LS',
        'algorithms': [EvaluatorSLM],
        'configuration_method': generate_random_1_1_slm_lr_ls_configuration,
        'max_combinations': DEFAULT_NUMBER_OF_COMBINATIONS
    }
}

SLM_LR_LS_1_5 = {
    'slm_lr_ls_1_5': {
        'name_long': '(1 + 5)-Semantic Learning Machine - Linear Regression Learning Step',
        'name_short': '(1 + 5)-SLM-LR-LS',
        'algorithms': [EvaluatorSLM],
        'configuration_method': generate_random_1_5_slm_lr_ls_configuration,
        'max_combinations': DEFAULT_NUMBER_OF_COMBINATIONS
    }
}

SLM_LR_LS_1_10 = {
    'slm_lr_ls_1_10': {
        'name_long': '(1 + 10)-Semantic Learning Machine - Linear Regression Learning Step',
        'name_short': '(1 + 10)-SLM-LR-LS',
        'algorithms': [EvaluatorSLM],
        'configuration_method': generate_random_1_10_slm_lr_ls_configuration,
        'max_combinations': DEFAULT_NUMBER_OF_COMBINATIONS
    }
}

SLM_PROTECTED_OLS = {
    'slm_protected_ols': {
        'name_long': 'Semantic Learning Machine - Protected Optimized Learning Step',
        'name_short': 'SLM-P-OLS',
        'algorithms': [EvaluatorSLM],
        'configuration_method': generate_random_slm_protected_ols_configuration,
        'max_combinations': DEFAULT_NUMBER_OF_COMBINATIONS
    }
}

SLM_BOOTSTRAP_OLS_MEDIAN = {
    'slm_bootstrap_ols_median': {
        'name_long': 'Semantic Learning Machine - Bootstrap Optimized Learning Step (median)',
        'name_short': 'SLM-B-OLS-median',
        'algorithms': [EvaluatorSLM],
        'configuration_method': generate_random_slm_bootstrap_ols_median_configuration,
        'max_combinations': DEFAULT_NUMBER_OF_COMBINATIONS
    }
}

SLM_BOOTSTRAP_OLS_MEAN = {
    'slm_bootstrap_ols_mean': {
        'name_long': 'Semantic Learning Machine - Bootstrap Optimized Learning Step (mean)',
        'name_short': 'SLM-B-OLS-mean',
        'algorithms': [EvaluatorSLM],
        'configuration_method': generate_random_slm_bootstrap_ols_mean_configuration,
        'max_combinations': DEFAULT_NUMBER_OF_COMBINATIONS
    }
}

NEAT = {
    'neat': {
        'name_long': 'NeuroEvolution of Augmented Topologies',
        'name_short': 'NEAT',
        'algorithms': [EvaluatorNEAT],
        'configuration_method': generate_random_neat_configuration,
        'max_combinations': DEFAULT_NUMBER_OF_COMBINATIONS
    }
}

FTNE = {
    'ftne': {
        'name_long': 'Fixed-Topology NeuroEvolution',
        'name_short': 'FTNE',
        'algorithms': [EvaluatorFTNE],
        'configuration_method': generate_random_ftne_configuration,
        'max_combinations': DEFAULT_NUMBER_OF_COMBINATIONS
    }
}

SLM_BLS_SSC = {
    'slm_bls_ssc': {
        'name_long': 'Semantic Learning Machine - Bounded Learning Step with Semantic Stopping Criterion',
        'name_short': 'SLM-BLS-SSC',
        'algorithms': [EvaluatorSLM],
        'configuration_method': generate_random_slm_bls_tie_edv_configuration,
        'max_combinations': DEFAULT_NUMBER_OF_COMBINATIONS
    }
}

SLM_OLS_SSC = {
    'slm_ols_ssc': {
        'name_long': 'Semantic Learning Machine - Optimized Learning Step with Semantic Stopping Criterion',
        'name_short': 'SLM-OLS-SSC',
        'algorithms': [EvaluatorSLM],
        'configuration_method': generate_random_slm_ols_edv_configuration,
        'max_combinations': DEFAULT_NUMBER_OF_COMBINATIONS
    }
}

MLP_C = {
    'mlp_c': {
        'name_long': 'Multilayer Perceptron (SGD or ADAM solver)',
        'name_short': 'MLP (SGD or ADAM)',
        'algorithms': [EvaluatorMLPC],
        'configuration_method': generate_random_mlp_configuration,
        'max_combinations': DEFAULT_NUMBER_OF_COMBINATIONS
    }
}

MLP_C_TRAINING = {
    'mlp_c_training': {
        'name_long': 'Multilayer Perceptron (SGD or ADAM solver)',
        'name_short': 'MLP (SGD or ADAM)',
        'algorithms': [EvaluatorMLPC],
        'configuration_method': generate_random_mlp_configuration_training,
        'max_combinations': DEFAULT_NUMBER_OF_COMBINATIONS
    }
}

MLP_R = {
    'mlp_r': {
        'name_long': 'Multilayer Perceptron (SGD or ADAM solver)',
        'name_short': 'MLP (SGD or ADAM)',
        'algorithms': [EvaluatorMLPR],
        'configuration_method': generate_random_mlp_configuration,
        'max_combinations': DEFAULT_NUMBER_OF_COMBINATIONS
    }
}

MLP_R_TRAINING = {
    'mlp_r_training': {
        'name_long': 'Multilayer Perceptron (SGD or ADAM solver)',
        'name_short': 'MLP (SGD or ADAM)',
        'algorithms': [EvaluatorMLPR],
        'configuration_method': generate_random_mlp_configuration_training,
        'max_combinations': DEFAULT_NUMBER_OF_COMBINATIONS
    }
}

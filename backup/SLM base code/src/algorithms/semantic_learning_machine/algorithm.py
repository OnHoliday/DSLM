from copy import copy, deepcopy
from numpy.random import choice as np_choice
from random import uniform, sample, choice, randint
from statistics import median, mean

from numpy import array, dot, resize, shape
from numpy import std
from numpy.core.multiarray import arange
from numpy.linalg import pinv
from sklearn.linear_model import LinearRegression

import algorithms.common
from algorithms.common.algorithm import EvolutionaryAlgorithm
from algorithms.common.neural_network.activation_function import _NON_LINEAR_ACTIVATION_FUNCTIONS
from algorithms.common.neural_network.connection import Connection
from algorithms.common.neural_network.neural_network import ConvNeuralNetwork, create_neuron, \
    create_output_neuron
from algorithms.common.neural_network.node import Sensor
from algorithms.semantic_learning_machine.mutation_operator import Mutation4
from algorithms.semantic_learning_machine.solution import Solution


class SemanticLearningMachine(EvolutionaryAlgorithm):
    """
    Class represents Semantic Learning Machine (SLM) algorithms:
    https://www.researchgate.net/publication/300543369_Semantic_Learning_Machine_
    A_Feedforward_Neural_Network_Construction_Algorithm_Inspired_by_Geometric_Semantic_Genetic_Programming
    Attributes:
        layer: Number of layers for base topology.
        learning_step: Weight for connection to output neuron.
        max_connections: Maximum connections for neuron.
        mutation_operator: Operator that augments neural network.
        next_champion: Solution that will replace champion.
    Notes:
        learning_step can be positive numerical value of 'optimized' for optimized learning step.
    """

    def __init__(self, population_size, stopping_criterion, layers, learning_step,
                max_connections=None, mutation_operator=Mutation4(), init_minimum_layers=2, init_maximum_neurons_per_layer=5, maximum_neuron_connection_weight=0.5, maximum_bias_connection_weight=1.0, subset_ratio=1, weight_range=1.0,
                random_sampling_technique=False, random_weighting_technique=False, protected_ols=False, bootstrap_ols=False, bootstrap_ols_samples=10, bootstrap_ols_criterion='median', high_absolute_ls_difference=1, store_ls_history=False):
        super().__init__(population_size, stopping_criterion)
        self.layers = layers
        self.learning_step = learning_step
        self.max_connections = max_connections
        self.mutation_operator = mutation_operator
        self.init_minimum_layers = init_minimum_layers
        self.init_maximum_neurons_per_layer = init_maximum_neurons_per_layer
        self.maximum_neuron_connection_weight = maximum_neuron_connection_weight
        self.maximum_bias_connection_weight = maximum_bias_connection_weight
        self.next_champion = None
        self.random_sampling_technique = random_sampling_technique
        self.random_weighting_technique = random_weighting_technique
        self.subset_ratio = subset_ratio
        self.weight_range = weight_range
        self.protected_ols = protected_ols
        self.bootstrap_ols = bootstrap_ols
        self.bootstrap_ols_samples = bootstrap_ols_samples
        self.bootstrap_ols_criterion = bootstrap_ols_criterion
        if self.bootstrap_ols:
            self.high_absolute_ls_difference = high_absolute_ls_difference
            self.high_absolute_differences_history = []
        self.store_ls_history = store_ls_history
        if self.store_ls_history:
            self.ls_history = []
        self.zero_ls_by_activation_function = {}
        self.zero_ls_history = []
        self.lr_intercept = None

    def _get_learning_step(self, partial_semantics):
        """Returns learning step."""
        
        if self.learning_step == 'lr-ls':
            ls = self._get_linear_regression_learning_step(partial_semantics)
        # If learning step is 'optimized', calculate optimized learning step.
        elif self.learning_step == 'optimized':
            ls = self._get_optimized_learning_step(partial_semantics)
        # Else, return numerical learning step.
        else:
            ls = uniform(-self.learning_step, self.learning_step)
            # ls = self.learning_step
        
        if self.store_ls_history:
            self.ls_history += [ls]
        
        return ls

    def _get_linear_regression_learning_step(self, partial_semantics):
        
        delta_target = copy(self.target_vector).astype(float)
        if self.champion:
            delta_target -= self.champion.neural_network.get_predictions()
        X = partial_semantics.reshape(-1, 1)
        y = delta_target.reshape(-1, 1)
        lr = LinearRegression().fit(X, y)
        ls = lr.coef_[0][0]
        self.lr_intercept = lr.intercept_[0]
        #=======================================================================
        # print('Score:', lr.score(X, y))
        # print('lr.coef_:', lr.coef_)
        # print('lr.intercept_:', lr.intercept_)
        #=======================================================================
        return ls
    
    def _get_optimized_learning_step(self, partial_semantics):
        """Calculates optimized learning step."""
        
        """ bootstrap samples; compute OLS for each; use desired criterion to select the final LS """
        if self.bootstrap_ols:
            
            weights = []
            size = self.target_vector.shape[0]
            
            for sample in range(self.bootstrap_ols_samples):
                
                idx = np_choice(arange(size), size, replace=True)
                
                bootstrap_delta_target = copy(self.target_vector[idx]).astype(float)
                if self.champion:
                    full_predictions = self.champion.neural_network.get_predictions()
                    bootstrap_delta_target -= full_predictions[idx]
                
                bootstrap_partial_semantics = partial_semantics[idx]
                inverse = array(pinv(resize(bootstrap_partial_semantics, (1, bootstrap_partial_semantics.size))))
                ols = dot(inverse.transpose(), bootstrap_delta_target)[0]
                
                weights += [ols]
            
            ols_median = median(weights)
            ols_mean = mean(weights)
            ols = self._compute_ols(partial_semantics)
            abs_dif = abs(ols_median - ols_mean)
            
            if abs_dif >= self.high_absolute_ls_difference:
                self.high_absolute_differences_history.append([abs_dif, ols_median, ols_mean, ols])
                #===============================================================
                # print('Absolute difference: %.3f, median vs. mean: %.3f vs. %.3f' % (abs_dif, ols_median, ols_mean))
                #===============================================================
                #===============================================================
                # print('Absolute difference: %.3f, median vs. mean vs. OLS: %.3f vs. %.3f vs. %.3f' % (abs_dif, ols_median, ols_mean, ols))
                # print()
                #===============================================================
            
            if self.bootstrap_ols_criterion == 'median':
                return median(weights)
            else:
                return mean(weights)
        
        else:
            return self._compute_ols(partial_semantics)

    def _compute_ols(self, partial_semantics):
            # Calculates distance to target vector.
            delta_target = copy(self.target_vector).astype(float)
            if self.champion:
                """ version to use when no memory issues exist """
                delta_target -= self.champion.neural_network.get_predictions()
                """ version attempting to circumvent the memory issues """
                #===============================================================
                # predictions = self.champion.neural_network.get_predictions()
                # for i in range(delta_target.shape[0]):
                #     delta_target[i] -= predictions[i]
                #===============================================================
            # Calculates pseudo-inverse of partial_semantics.
            inverse = array(pinv(resize(partial_semantics, (1, partial_semantics.size))))
            # inverse = array(pinv(matrix(partial_semantics)))
            # Returns dot product between inverse and delta.
            ols = dot(inverse.transpose(), delta_target)[0]
            
            if ols == 0:
                self.zero_ls_history.append([partial_semantics, delta_target, None])
            
            if self.protected_ols:
                if self._valid_ols(delta_target, partial_semantics, ols) == False:
                    ols = 0
            
            return ols
    
    def _valid_ols(self, delta_target, partial_semantics, ols):
        size = delta_target.shape[0]
        absolute_ideal_weights = []
        for i in range(size):
            if partial_semantics[i] != 0:
                absolute_ideal_weight = delta_target[i] / partial_semantics[i]
            else:
                absolute_ideal_weight = 0
            
            absolute_ideal_weights += [abs(absolute_ideal_weight)]
        
        #=======================================================================
        # print(median(absolute_ideal_weights))
        # print(mean(absolute_ideal_weights))
        # print(std(absolute_ideal_weights))
        # print(abs(ols))
        #=======================================================================
        
        upper_bound = mean(absolute_ideal_weights) + 2 * std(absolute_ideal_weights)
        lower_bound = mean(absolute_ideal_weights) - 2 * std(absolute_ideal_weights)
        if abs(ols) > upper_bound or abs(ols) < lower_bound:
            #===================================================================
            # print('\tInvalid OLS')
            #===================================================================
            return False
        else:
            return True
    
    def _get_connection_weight(self, weight):
        """Returns connection weight if defined, else random value between -1 and 1."""
        
        if weight:
            return weight
        else:
            return uniform(-self.maximum_neuron_connection_weight, self.maximum_neuron_connection_weight)
    
    def _connect_nodes(self, from_nodes, to_nodes, weight=None, random=False):
        """
        Connects list of from_nodes with list of to_nodes.
        Args:
            from_nodes: List of from_nodes.
            to_nodes: List of to_nodes.
            weight: Weight from connection.
            random: Flag if random number of connections.
        Notes:
            If weight is None, then weight will be chosen at random between -1 and 1.
        """

        for to_node in to_nodes:
            # If random, create random sample of connection partners
            if random:
                max_connections = len(from_nodes)
                random_connections = randint(1, max_connections)
                from_nodes_sample = sample(from_nodes, random_connections)
            else:
                from_nodes_sample = from_nodes
            # Connect to_node to each node in from_node_sample.
            for from_node in from_nodes_sample:
                Connection(from_node, to_node, self._get_connection_weight(weight))

    def _connect_nodes_mutation(self, hidden_layers):
        """Connects new mutation neurons to remainder of network."""

        # Sets reference to champion neural network.
        neural_network = self.champion.neural_network
        # Create hidden origin layer.
        from_layers = [copy(hidden_layer) for hidden_layer in hidden_layers]
        for hidden_layer_new, hidden_layer_old in zip(from_layers, neural_network.hidden_layers):
            hidden_layer_new.extend(hidden_layer_old)
        # Establish connections.
        self._connect_nodes(neural_network.sensors, hidden_layers[0], random=True)
        previous_neurons = from_layers[0]
        for from_layer, to_layer in zip(from_layers[1:], hidden_layers[1:]):
            self._connect_nodes(previous_neurons, to_layer, random=True)
            previous_neurons = from_layer

    def _connect_learning_step(self, neural_network):
        """Connects last hidden neuron with defined learning step."""

        # Get last hidden neuron.
        last_neuron = neural_network.hidden_layers[-1][-1]
        # Get semantics of last neuron.
        last_semantics = last_neuron.semantics
        # Connect last neuron to output neuron.
        
        ls = self._get_learning_step(last_semantics)
        if self.learning_step == 'optimized' and self.bootstrap_ols == False and ls == 0:
            # print('\tActivation function:', last_neuron.activation_function)
            if len(self.zero_ls_history) > 0 and self.zero_ls_history[-1][2] == None:
                self.zero_ls_history[-1][2] = last_neuron.activation_function
                if last_neuron.activation_function in self.zero_ls_by_activation_function:
                    count = self.zero_ls_by_activation_function[last_neuron.activation_function]
                    self.zero_ls_by_activation_function[last_neuron.activation_function] = count + 1
                else:
                    self.zero_ls_by_activation_function[last_neuron.activation_function] = 1
            
            #===================================================================
            # if last_neuron.activation_function != 'relu':
            #     print('\tActivation function:', last_neuron.activation_function)
            #     print(self.zero_ls_history[-1])
            #     print()
            #===================================================================
        
        if self.stopping_criterion.__class__ == algorithms.common.stopping_criterion.MaxGenerationsCriterion: 
            if self.current_generation == self.stopping_criterion.max_generation:
                
                #===============================================================
                # if self.bootstrap_ols:
                #     if len(self.high_absolute_differences_history) > 0:
                #         print(self.high_absolute_differences_history)
                #         print('Number of high absolute differences:', len(self.high_absolute_differences_history))
                #===============================================================
                
                #===============================================================
                # if len(self.zero_ls_by_activation_function) > 0:
                #     print(self.zero_ls_by_activation_function)
                #     print()
                #===============================================================
                
                #===================================================================
                # if self.store_ls_history:
                #     print(self.ls_history)
                #===================================================================
                pass
        
        if self.lr_intercept:
            neural_network.output_neuron.input_connections[0].weight += self.lr_intercept #todo never used so didn't adjust for now
        
        self._connect_nodes([last_neuron], neural_network.output_layer, ls)

    def _create_solution(self, neural_network):
        """Creates solution for population."""

        # Creates solution object.
        solution = Solution(neural_network, None, None)
        # Calculates error.
        solution.value = self.metric.evaluate(neural_network.get_predictions(), self.target_vector)
        # Checks, if solution is better than parent.
        solution.better_than_ancestor = self._is_better_solution(solution, self.champion)
        # After the output semantics are updated, we can remove the semantics from the final hidden neuron.
        # neural_network.output_neuron.input_connections[-1].from_node.semantics = None
        for neuron in neural_network.get_output_neurons(): neuron.input_connections[-1].from_node.semantics = None

        # Returns solution.
        return solution

    def _initialize_sensors(self):
        """Initializes sensors based on input matrix."""

        return [Sensor(input_data) for input_data in self.input_matrix.T]

    def _initialize_bias(self, neural_network):
        """Initializes biases with same length as sensors."""

        return Sensor(resize(array([1]), shape(neural_network.sensors[0].semantics)))

    def _initialize_hidden_layers(self, neural_network):
        """Initializes hidden layers, based on defined number of layers."""
        
        number_of_layers = randint(self.init_minimum_layers, self.layers)
        neurons_per_layer = [randint(1, self.init_maximum_neurons_per_layer) for i in range(number_of_layers - 1)]
        hidden_layers = [[create_neuron(None, neural_network.bias, maximum_bias_connection_weight=self.maximum_bias_connection_weight) for neuron in range(neurons_per_layer[layer])] for layer in range(number_of_layers - 1)]
        
        # From Jan: Create hidden layers with one neuron with random activation function each.
        # hidden_layers = [[create_neuron(None, neural_network.bias)] for i in range(self.layers - 1)]
        
        # Add final hidden layer with one neuron
        activation_function = choice(list(_NON_LINEAR_ACTIVATION_FUNCTIONS.keys()))
        hidden_layers.append([create_neuron(activation_function, neural_network.bias, maximum_bias_connection_weight=self.maximum_bias_connection_weight)])
        # Returns hidden layers.
        return hidden_layers

    def _initialize_output_layer(self, neural_network):
        """Initializes hidden layers, based on defined number of layers."""

        if self.nr_of_classes > 2:
            output_layer = [create_output_neuron('identity', neural_network.bias) for neuron in range(self.nr_of_classes)]
        else:
            output_layer = [create_output_neuron('identity', neural_network.bias)]

        return output_layer

    def _initialize_topology(self):
        """Initializes topology."""

        # Create sensors.
        sensors = self._initialize_sensors()
        # Create neural network.
        neural_network = ConvNeuralNetwork(sensors, None, None, None, None)
        # Create bias.
        neural_network.bias = self._initialize_bias(neural_network)
        # Return neural network.
        return neural_network

    def _initialize_neural_network(self, topology):
        """Creates neural network from initial topology."""

        # Create shallow copy of topology.
        neural_network = copy(topology)
        # Create output neuron.
        neural_network.output_layer = self._initialize_output_layer(neural_network)

        # neural_network.output_neuron = create_output_neuron('identity', neural_network.bias)
        #=======================================================================
        # neural_network.output_neuron = create_neuron('identity', None)
        #=======================================================================
        
        # Create cnn layer.
        neural_network.cnn_layers = self._initialize_convolution_layers(neural_network)
        # Create hidden layer.
        neural_network.hidden_layers = self._initialize_hidden_layers(neural_network)
        # Establish connections
        self._connect_nodes(neural_network.sensors, neural_network.hidden_layers[0], random=True)
        previous_neurons = neural_network.hidden_layers[0]
        for hidden_layer in neural_network.hidden_layers[1:]:
            self._connect_nodes(previous_neurons, hidden_layer, random=True)
            previous_neurons = hidden_layer
        # Calculate hidden neurons.
        for layer in neural_network.hidden_layers:
            for neuron in layer:
                neuron.calculate()
        # Connect last neuron to output neuron with learning step.
        self._connect_learning_step(neural_network)
        # Calculate output semantics.
        for neuron in neural_network.output_layer:
            neuron.calculate()
        # Return neural network.
        return neural_network

    def _initialize_solution(self, topology):
        """Creates solution for initial population."""

        # Initialize neural network.
        neural_network = self._initialize_neural_network(topology)
        # Create solution.
        solution = self._create_solution(neural_network)
        # Return solution.
        return solution

    def _initialize_population(self):
        """Initializes population in first generation."""
        # def time_seconds(): return default_timer()
        # start_time = time_seconds() 
        # Initializes neural network topology.
        topology = self._initialize_topology()
        # Create initial population from topology.
        for i in range(self.population_size):
            solution = self._initialize_solution(topology)
            
            # -IG-
            # print('\t\tInit, individual:', i, ', topology:', solution.neural_network.get_topology())
            
            if not self.next_champion:
                self.next_champion = solution
            elif self._is_better_solution(solution, self.next_champion):
                self.next_champion.neural_network = None
                self.next_champion = solution
            else:
                solution.neural_network = None
            self.population.append(solution)
        # print("time to initialize population: ", time_seconds()-start_time)

    def _mutate_network(self):
        """Creates mutated offspring from champion neural network."""

        # Create shallow copy of champion neural network.
        neural_network = deepcopy(self.champion.neural_network)
        # Create mutated hidden layers.
        mutation_layers = self.mutation_operator.mutate_network(self)
        # Connect hidden neurons to remainder of network.
        self._connect_nodes_mutation(mutation_layers)
        # Calculate mutated hidden layer.
        for mutation_layer in mutation_layers:
            for neuron in mutation_layer:
                neuron.calculate()

        # Extend hidden layers.
        for hidden_layer, mutation_layers in zip(neural_network.hidden_layers, mutation_layers):
            hidden_layer.extend(mutation_layers)
        # Connect final hidden neuron to output neuron.
        self._connect_learning_step(neural_network)
        # Get most recent connection.
        for neuron in neural_network.output_layer:
            connection = neuron.input_connections[-1]

            # Update semantics of output neuron.
            if self.lr_intercept:
                #===================================================================
                # neural_network.output_neuron.semantics2 = copy(neural_network.output_neuron.semantics)
                # neural_network.output_neuron.semantics2 += connection.from_node.semantics * connection.weight
                #===================================================================
                neuron.semantics += connection.from_node.semantics * connection.weight + self.lr_intercept
                #===================================================================
                # print(neural_network.output_neuron.semantics2 - neural_network.output_neuron.semantics)
                # print(self.lr_intercept)
                # print()
                #===================================================================
            else:
                neuron.semantics = connection.from_node.semantics * connection.weight
        
        # Return neural network.
        return neural_network

    def _mutate_solution(self):
        """Applies mutation operator to current champion solution."""

        # Created mutated offspring of champion neural network.
        neural_network = self._mutate_network()
        # Create solution.
        solution = self._create_solution(neural_network)
        # Return solution.
        return solution

    def _mutate_population(self):
        """ ... """
        if self.random_sampling_technique:
            # calculate the new predictions and update the champion's error according to the new input matrix previously generated
            champ_predictions = self.champion.neural_network.predict(self.input_matrix)
            self.champion.predictions = champ_predictions 
            self.champion.value = self.metric.evaluate(self.champion.predictions, self.target_vector)
        
        if self.random_weighting_technique:
            # calculate the new predictions and update the champion's error according to the new input matrix previously generated
            self.champion.value = self.metric.evaluate(self.champion.predictions, self.target_vector)
        
        # print('\t\tMutation, champion topology:', self.champion.neural_network.get_topology())
        
        for i in range(self.population_size):
            solution = self._mutate_solution()
            
            # -IG-
            # print('\t\tMutation, individual:', i, ', topology:', solution.neural_network.get_topology())
            
            if not self.next_champion:
                if self._is_better_solution(solution, self.champion):
                    self.next_champion = solution
                else:
                    solution.neural_network = None
            elif self._is_better_solution(solution, self.next_champion):
                self.next_champion.neural_network = None
                self.next_champion = solution
            else:
                solution.neural_network = None
            self.population.append(solution)

    def _wipe_population(self):
        self.population = list()

    def _override_current_champion(self):
        if self.next_champion:
            self.champion = self.next_champion
            self.next_champion = None

    def _epoch(self):
        if self.current_generation == 0:
            self._initialize_population()
        else:
            self._mutate_population()
        stopping_criterion = self.stopping_criterion.evaluate(self)
        self._override_current_champion()
        self._wipe_population()
        return stopping_criterion

    def fit(self, input_matrix, target_vector, metric, verbose=False):
        super().fit(input_matrix, target_vector, metric, verbose)
        self.champion.neural_network = deepcopy(self.champion.neural_network)

    def predict(self, input_matrix):
        neural_network = self.champion.neural_network
        neural_network.load_sensors(input_matrix)
        neural_network.calculate()
        return neural_network.get_predictions()

    def __repr__(self):
        return 'SemanticLearningMachine'

class DeepSemanticLearningMachine(SemanticLearningMachine):
    def __init__(self, mutation_operator, max_depth, max_length, population_size, stopping_criterion, layers, learning_step,
                max_connections,  init_minimum_layers, init_maximum_neurons_per_layer, maximum_neuron_connection_weight, maximum_bias_connection_weight, subset_ratio, weight_range,
                random_sampling_technique, random_weighting_technique, protected_ols, bootstrap_ols, bootstrap_ols_samples, bootstrap_ols_criterion, high_absolute_ls_difference, store_ls_history):
        super().__init__(population_size, stopping_criterion, population_size, stopping_criterion, layers, learning_step,
                max_connections,  init_minimum_layers, init_maximum_neurons_per_layer, maximum_neuron_connection_weight, maximum_bias_connection_weight, subset_ratio, weight_range,
                random_sampling_technique, random_weighting_technique, protected_ols, bootstrap_ols, bootstrap_ols_samples, bootstrap_ols_criterion, high_absolute_ls_difference, store_ls_history)
        self.max_depth = max_depth
        self.max_length = max_length


# Each mutation can add elements either to existing line or next one ( net is like tree )
    def _initialize_population(self):
        """Initializes population in first generation."""
        # def time_seconds(): return default_timer()
        # start_time = time_seconds()
        # Initializes neural network topology.
        topology = self._initialize_topology() #todo zmien to na CNN tolopogy
        # Create initial population from topology.
        for i in range(self.population_size):
            solution = self._initialize_solution(topology)

            if not self.next_champion:
                self.next_champion = solution
            elif self._is_better_solution(solution, self.next_champion):
                self.next_champion.neural_network = None
                self.next_champion = solution
            else:
                solution.neural_network = None
            self.population.append(solution)
        # print("time to initialize population: ", time_seconds()-start_time)

    def _create_solution(self, neural_network):
        """Creates solution for population."""

        # Creates solution object.
        solution = Solution(neural_network, None, None)
        # Calculates error.
        solution.value = self.metric.evaluate(neural_network.get_predictions(), self.target_vector)
        # Checks, if solution is better than parent.
        solution.better_than_ancestor = self._is_better_solution(solution, self.champion)
        # After the output semantics are updated, we can remove the semantics from the final hidden neuron.
        neural_network.output_neuron.input_connections[-1].from_node.semantics = None
        # Returns solution.
        return solution

    def _initialize_sensors(self):
        """Initializes sensors based on input matrix."""

        return [Sensor(input_data) for input_data in self.input_matrix.T]

    def _initialize_flat_sensors(self):
        """Initializes sensors based on input matrix."""

        return [Sensor(input_data) for input_data in self.input_matrix.T]

    def _initialize_bias(self, neural_network):
        """Initializes biases with same length as sensors."""

        return Sensor(resize(array([1]), shape(neural_network.sensors[0].semantics)))

    def _initialize_hidden_layers(self, neural_network):
        """Initializes hidden layers, based on defined number of layers."""

        number_of_layers = randint(self.init_minimum_layers, self.layers)
        neurons_per_layer = [randint(1, self.init_maximum_neurons_per_layer) for i in range(number_of_layers - 1)]
        hidden_layers = [[create_neuron(None, neural_network.bias,
                                        maximum_bias_connection_weight=self.maximum_bias_connection_weight) for neuron
                          in range(neurons_per_layer[layer])] for layer in range(number_of_layers - 1)]

        # From Jan: Create hidden layers with one neuron with random activation function each.
        # hidden_layers = [[create_neuron(None, neural_network.bias)] for i in range(self.layers - 1)]

        # Add final hidden layer with one neuron
        activation_function = choice(list(_NON_LINEAR_ACTIVATION_FUNCTIONS.keys()))
        hidden_layers.append([create_neuron(activation_function, neural_network.bias,
                                            maximum_bias_connection_weight=self.maximum_bias_connection_weight)])
        # Returns hidden layers.
        return hidden_layers

    def _initialize_cnn_topology(self):
        """Initializes topology."""
        #todo dodaj tutaj kod na cnn i nn layersy nie tylko nn
        # Create sensors.
        sensors = self._initialize_sensors()
        # Create neural network.
        neural_network = ConvNeuralNetwork(sensors, None, None, None, None)
        # Create bias.
        neural_network.bias = self._initialize_bias(neural_network)
        # Return neural network.
        return neural_network

    def _initialize_neural_network(self, topology):
        """Creates neural network from initial topology."""

        # Create shallow copy of topology.
        neural_network = copy(topology)
        # Create output neuron.
        neural_network.output_neuron = create_output_neuron('identity', neural_network.bias)
        # =======================================================================
        # neural_network.output_neuron = create_neuron('identity', None)
        # =======================================================================

        # Create hidden layer.
        neural_network.hidden_layers = self._initialize_hidden_layers(neural_network)
        # Establish connections
        self._connect_nodes(neural_network.sensors, neural_network.hidden_layers[0], random=True)
        previous_neurons = neural_network.hidden_layers[0]
        for hidden_layer in neural_network.hidden_layers[1:]:
            self._connect_nodes(previous_neurons, hidden_layer, random=True)
            previous_neurons = hidden_layer
        # Calculate hidden neurons.
        for layer in neural_network.hidden_layers:
            for neuron in layer:
                neuron.calculate()
        # Connect last neuron to output neuron with learning step.
        self._connect_learning_step(neural_network)
        # Calculate output semantics.
        neural_network.output_neuron.calculate()
        # Return neural network.
        return neural_network

    def _initialize_solution(self, topology):
        """Creates solution for initial population."""

        # Initialize neural network.
        neural_network = self._initialize_neural_network(topology)
        # Create solution.
        solution = self._create_solution(neural_network)
        # Return solution.
        return solution
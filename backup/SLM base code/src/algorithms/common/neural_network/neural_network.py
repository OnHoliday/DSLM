from copy import copy, deepcopy
from random import choice, uniform
from numpy import array, dot, resize, shape, concatenate
import numpy as np

from algorithms.common.neural_network.activation_function import _NON_LINEAR_ACTIVATION_FUNCTIONS
from algorithms.common.neural_network.connection import Connection
from algorithms.common.neural_network.node import Neuron, Sensor, ConvNeuron, PoolNeuron


class NeuralNetwork(object):
    """
    Class represents neural network.
    Attributes:
        sensors: List of input sensors.
        bias: Bias neuron.
        hidden_layers: List of layers, containing hidden neurons.
        output_neuron: Output neuron.
    """

    def __init__(self, sensors, bias, hidden_layers, output_layer):
        self.sensors = sensors
        self.bias = bias
        self.hidden_layers = hidden_layers
        self.output_layer = output_layer

    def __copy__(self):
        # Bias can be referenced.
        bias = self.bias
        # Sensors can be referenced.
        sensors = self.sensors
        # Creates shallow copy of every hidden layer, while only referencing the contained neurons.
        hidden_layers = [copy(hidden_layer) for hidden_layer in self.hidden_layers] if self.hidden_layers else list()
        # Copies output layer.
        output_layer = copy(self.output_layer) if self.output_layer else None
        # Returns shallow copy of self.
        return NeuralNetwork(sensors, bias, hidden_layers, output_layer)

    def __deepcopy__(self, memodict={}):
        bias = deepcopy(self.bias, memodict)
        sensors = deepcopy(self.sensors, memodict)
        hidden_layers = deepcopy(self.hidden_layers, memodict)
        output_layers = deepcopy(self.output_layer, memodict)
        neural_network = NeuralNetwork(sensors, bias, hidden_layers, output_layers)
        memodict[id(self)] = neural_network
        return neural_network

    def get_weights(self):
        return np.array([connection.weight for connection in self.get_connections()])

    def calculate(self):
        """Calculates semantics of all hidden neurons and output neuron."""
        for hidden_layer in self.hidden_layers:
            for neuron in hidden_layer:
                neuron.calculate()
        for neuron in self.output_layer:
            neuron.calculate()


    def get_predictions(self):
        """Returns semantics of output neuron."""
        final_preds = 0
        for neuron in self.output_layer:
            final_preds += neuron.semantics

        return final_preds

    def load_sensors(self, X):
        """Loads input variables of dataset into sensors. Adjusts length of bias."""
        for sensor, sensor_data in zip(self.sensors, X.T):
            sensor.semantics = sensor_data
        self.bias.semantics = np.array([1 for i in range(sensor_data.shape[0])])
        # self.bias.semantics.resize(shape(self.sensors[0].semantics), refcheck = False)

    def get_hidden_neurons(self):
        """Returns list of hidden neurons."""
        neurons = list()
        [neurons.extend(hidden_neurons) for hidden_neurons in self.hidden_layers]
        return neurons

    def get_output_neurons(self):
        """Returns list of hidden neurons."""
        neurons = list()
        [neurons.append(output_neurons) for output_neurons in self.output_layer]
        return neurons

    def get_connections(self):
        """Returns list of connections."""
        neurons = list()
        neurons.extend(self.get_hidden_neurons())
        neurons.extend(self.get_output_neurons())
        connections = list()
        [connections.extend(neuron.input_connections) for neuron in neurons]
        return connections

    def get_topology(self):
        """Returns number of hidden layers, number of hidden neurons and number of connections."""
        return {
            'layers': len(self.hidden_layers),
            'neurons': len(self.get_hidden_neurons()),
            'connections': len(self.get_connections())
        }

    def predict(self, X):
        self.load_sensors(X)
        self.calculate()
        return self.get_predictions()

    def add_sensors(self, X):
        """Adds sensors to neural network and loads input data."""
        self.sensors = [Sensor(np.array([])) for sensor_data in X.T]
        self.load_sensors(X)
        # Connects nodes to first level hidden layer.
        _connect_nodes(self.sensors, self.hidden_layers[0])

    def wipe_semantics(self):
        """Sets semantics of all neurons to empty numpy array."""
        for neuron in self.get_hidden_neurons(): neuron.semantics = np.array([])
        for neuron in self.get_output_neurons(): neuron.semantics = np.array([])


def create_neuron(activation_function=None, bias=None, maximum_bias_connection_weight=1.0):
	"""Creates neuron with defined activation function and bias."""
	# If activation function not defined, choose activation function at random.
	if not activation_function:
		activation_function = choice(list(_NON_LINEAR_ACTIVATION_FUNCTIONS.keys()))
		# activation_function = choice(list(_ACTIVATION_FUNCTIONS.keys()))
	neuron = Neuron(np.array([]), list(), activation_function)
	# If is biased, connect to bias with random weight.
	if bias:
		Connection(bias, neuron, uniform(-maximum_bias_connection_weight, maximum_bias_connection_weight))
	return neuron


def create_output_neuron(activation_function, bias, initial_bias_connection_weight=0.0):
    neuron = Neuron(np.array([]), list(), activation_function)
    Connection(bias, neuron, initial_bias_connection_weight)
    return neuron

def _connect_nodes(from_nodes, to_nodes, weight=0):
    """Connects all from nodes with all to nodes with determined weight."""
    for to_node in to_nodes:
        for from_node in from_nodes:
            Connection(from_node, to_node, weight)

def create_cnn_neuron(activation_function=None):
    if not activation_function:
        activation_function = choice(list(_NON_LINEAR_ACTIVATION_FUNCTIONS.keys()))
    what_layer = np.random.rand()

    if what_layer >= 0.5:
        neuron = ConvNeuron(np.array([]), list(),activation_function)
    else:
        neuron = PoolNeuron(np.array([]), list(),activation_function)

    return neuron

def get_final_output(x):
    """Compute softmax values for each sets of scores in x."""
    # print(1)
    arr = []
    for neuron in x:
        arr.append(neuron.semantics)
    np_arr = np.array(arr)
    e_x = np.exp(np_arr - np.max(np_arr))
    return e_x / e_x.sum(axis=0)



class ConvNeuralNetwork(object):
    """
    Class represents convolutional neural network.
    Attributes:
        sensors: List of input sensors.
        bias: Bias neuron.
        hidden_layers: List of layers, containing hidden neurons.
        output_neuron: Output neuron.
    """

    def __init__(self, sensors, bias, cnn_layers, flatten_layer, hidden_layers, output_layer):
        self.sensors = sensors
        self.bias = bias
        self.cnn_layers = cnn_layers
        self.flatten_layer = flatten_layer
        self.hidden_layers = hidden_layers
        self.output_layer = output_layer

    def __copy__(self):
        # Bias can be referenced.
        bias = self.bias
        # Sensors can be referenced.
        sensors = self.sensors
        # Creates shallow copy of every conv layer, while only referencing the contained neurons.
        cnn_layers = [copy(cnn_layers) for cnn_layers in self.cnn_layers] if self.cnn_layers else list()
        # Creates shallow copy of every hidden layer, while only referencing the contained neurons.
        flatten_layer = copy(self.flatten_layer) if self.flatten_layer else list()
        # Creates shallow copy of every hidden layer, while only referencing the contained neurons.
        hidden_layers = [copy(hidden_layer) for hidden_layer in self.hidden_layers] if self.hidden_layers else list()
        # Copies output layer.
        output_layer = copy(self.output_layer) if self.output_layer else list()

        # output_neuron = copy(self.output_neuron) if self.output_neuron else None
        # Returns shallow copy of self.
        return ConvNeuralNetwork(sensors, bias, cnn_layers, flatten_layer, hidden_layers, output_layer)

    def __deepcopy__(self, memodict={}):
        bias = deepcopy(self.bias, memodict)
        sensors = deepcopy(self.sensors, memodict)
        cnn_layers = deepcopy(self.cnn_layers, memodict)
        flatten_layer = deepcopy(self.flatten_layer, memodict)
        hidden_layers = deepcopy(self.hidden_layers, memodict)
        output_layer = deepcopy(self.output_layer, memodict)
        neural_network = ConvNeuralNetwork(sensors, bias, cnn_layers,flatten_layer,  hidden_layers, output_layer)
        memodict[id(self)] = neural_network
        return neural_network

    def get_weights(self):
        return np.array([connection.weight for connection in self.get_connections()])

    def calculate(self):
        """Calculates semantics of all hidden neurons and output neuron."""
        i = 0
        for layer in self.cnn_layers:
            if i == 0:
                for neuron in layer:
                    neuron.calculate()
            else:
                for neuron in layer:
                    neuron.calculate2()
            i += 1

        last_layer = self.cnn_layers[-1]
        flatten_layer = None
        for neuron in last_layer:
            temp_arr = neuron.semantics
            temp_arr2 = temp_arr.reshape(-1, neuron.semantics.shape[3])

            if flatten_layer is not None:
                flatten_layer = concatenate((flatten_layer, temp_arr2), axis=0)
            else:
                flatten_layer = temp_arr2


        self.flatten_layer = array([Sensor(d) for d in flatten_layer])
        #todo tak zupelnie nie moze byc liczba innput_connections powinna byc dokladnie idealna !!!! cos tu nie gra do naprawy!!!
        self._connect_flat_hidden(self.flatten_layer, self.hidden_layers[0])
        self.load_semantic_after_flat_layer()

        for hidden_layer in self.hidden_layers:
            for neuron in hidden_layer:
                neuron.calculate()

        for neuron in self.output_layer:
            neuron.calculate()

    def get_predictions(self):
        """Returns semantics of output neuron."""
        final_output = get_final_output(self.output_layer)
        output = np.argmax(final_output, axis=0)
        self.final_output = np.transpose(output)
        return self.final_output

    def load_semantic_after_flat_layer(self):
        """Loads input variables of dataset into sensors. Adjusts length of bias."""

        for hidden_neuron in self.hidden_layers[0]:
            for connection, flat_data in zip(hidden_neuron.input_connections, self.flatten_layer.T):
                connection.from_node.semantics = flat_data.semantics
            self.bias.semantics = np.array([1 for i in range(flat_data.semantics.shape[0])])

    def _connect_flat_hidden(self, from_nodes, to_nodes):

        for to_node in to_nodes:
            #todo to trzeba zdecydowanie wywalic to jest bardzo bardzo zle !!!! nie mozna tak
            to_node.input_connections = []
            for from_node in from_nodes:
                Connection(from_node, to_node, uniform(-2, 2))

    def load_sensors(self, X):
        """Loads input variables of dataset into sensors. Adjusts length of bias."""
        for sensor, sensor_data in zip(self.sensors, X.T):
            sensor.semantics = sensor_data
        self.bias.semantics = np.array([1 for i in range(sensor_data.shape[0])])
        # self.bias.semantics.resize(shape(self.sensors[0].semantics), refcheck = False)

    def get_cnn_neurons(self):
        """Returns list of hidden neurons."""
        neurons = list()
        [neurons.extend(cnn_neurons) for cnn_neurons in self.cnn_layers]
        return neurons

    def get_hidden_neurons(self):
        """Returns list of hidden neurons."""
        neurons = list()
        [neurons.extend(hidden_neurons) for hidden_neurons in self.hidden_layers]
        return neurons

    def get_output_layer(self):
        """Returns list of hidden neurons."""
        neurons = list()
        [neurons.append(output_neurons) for output_neurons in self.output_layer]
        return neurons


    # def get_output_neurons(self):
    #     """Returns list of hidden neurons."""
    #     neurons = list()
    #     [neurons.append(output_neurons) for output_neurons in self.output_layer]
    #     return neurons

    def get_connections(self): ##TODO TOPRZEROBIC !!!!!!!!!!
        """Returns list of connections."""
        neurons = list()
        neurons.extend(self.get_cnn_neurons())
        neurons.extend(self.get_hidden_neurons())
        neurons.extend(self.get_output_layer())
        # neurons.append(self.output_neuron)
        connections = list()
        [connections.extend(neuron.input_connections) for neuron in neurons]
        return connections

    def get_topology(self):
        """Returns number of hidden layers, number of hidden neurons and number of connections."""
        return {
            'cnn_layers': len(self.cnn_layers),
            'hidden_layers': len(self.hidden_layers),
            'cnn_neurons': len(self.get_cnn_neurons()),
            'hidden_neurons': len(self.get_hidden_neurons()),
            'connections': len(self.get_connections())
        }

    def predict(self, X):
        self.load_sensors(X)
        self.calculate()
        return self.get_predictions()

    def add_sensors(self, X):
        """Adds sensors to neural network and loads input data."""
        self.sensors = [Sensor(np.array([])) for sensor_data in X.T]
        self.load_sensors(X)
        # Connects nodes to first level hidden layer.
        _connect_nodes(self.sensors, self.hidden_layers[0])

    def wipe_semantics(self):
        """Sets semantics of all neurons to empty numpy array."""
        for neuron in self.get_hidden_neurons(): neuron.semantics = np.array([])
        self.output_neuron.semantics = np.array([])

import numpy as np
from algorithms.common.neural_network.activation_function import calculate_output
from copy import copy, deepcopy


class Node(object):
    """
    Class represents abstract node in neural network.

    Attributes:
        semantics: Semantic vector
    """

    def __init__(self, semantics):
        self.semantics = semantics


class Sensor(Node):
    """
    Class represents input sensor in neural network.
    """

    def __deepcopy__(self, memodict={}):
        sensor = Sensor(np.array([]))
        memodict[id(self)] = sensor
        return sensor

class Neuron(Node):
    """
    Class represents neuron in neural network.

    Attributes:
        input_connections = Set of input connections
        activation_function = String for activation function id
    """

    def __init__(self, semantics, input_connections, activation_function):
        super().__init__(semantics)
        self.input_connections = input_connections
        self.activation_function = activation_function

    def __copy__(self):
        copy_semantics = copy(self.semantics)
        copy_input_connections = copy(self.input_connections)
        copy_activation_function = self.activation_function
        return Neuron(copy_semantics, copy_input_connections, copy_activation_function)

    def __deepcopy__(self, memodict={}):
        input_connections = deepcopy(self.input_connections, memodict)
        activation_function = self.activation_function
        semantics = np.array([])
        neuron = Neuron(semantics, input_connections, activation_function)
        memodict[id(self)] = neuron
        return neuron


    def _calculate_weighted_input(self):
        """Calculates weighted input from input connections."""
        # for connection in self.input_connections:
        #     print(len(connection.from_node.semantics))

        ###TODO ultra niebezpieczna sprawa naprawic to najszybcije jka sie bedzie dalo !!!

        # return np.sum([connection.from_node.semantics * connection.weight for connection in self.input_connections],
        #               axis=0)
        try:
            return np.sum([connection.from_node.semantics * connection.weight for connection in self.input_connections], axis=0)
        except ValueError:
            i = 0
            for connection in self.input_connections:
                i += 1

                if len(connection.from_node.semantics) == 0:
                    print(connection.from_node.semantics, i)
                    connection.from_node.semantics = np.ones(shape=self.input_connections[0].from_node.semantics.shape[0])
            return np.sum([connection.from_node.semantics * connection.weight for connection in self.input_connections], axis=0)


    def _calculate_output(self, weighted_input):
        """Calculates semantics, based on weighted input."""
        return calculate_output(weighted_input, self.activation_function)

    def calculate(self):
        """Calculates weighted input, then calculates semantics."""
        weighted_input = self._calculate_weighted_input()
        self.semantics = self._calculate_output(weighted_input)



class ConvNeuron_3dfilter(Neuron):
    def __init__(self, semantics, input_connections, activation_function):
        super().__init__(semantics, input_connections, activation_function)
        self._initialize_parameters()

    def __copy__(self):
        copy_semantics = copy(self.semantics)
        copy_input_connections = copy(self.input_connections)
        copy_activation_function = self.activation_function
        return ConvNeuron_3dfilter(copy_semantics, copy_input_connections, copy_activation_function)

    def __deepcopy__(self, memodict={}):
        input_connections = deepcopy(self.input_connections, memodict)
        activation_function = self.activation_function
        semantics = np.array([])
        neuron = ConvNeuron_3dfilter(semantics, input_connections, activation_function)
        memodict[id(self)] = neuron
        return neuron


    def _initialize_parameters(self):
        self.nr_of_channel = 1
        self.kernel_size = np.random.randint(2, 7)
        self.stride = np.random.randint(1, 3)
        self.filter = np.random.uniform(-1, 1, (self.kernel_size, self.kernel_size, self.dimenstions))

    def get_sizes(self, input_pic_array):
        self.input_width = input_pic_array.shape[0]
        self.input_length = input_pic_array.shape[1]
        self.dimenstions = input_pic_array.shape[2]
        self.output_width = int(np.ceil((input_pic_array.shape[0] - self.kernel_size + 1) / self.stride))
        self.output_length = int(np.ceil((input_pic_array.shape[1] - self.kernel_size + 1) / self.stride))


    def convolv(self):
        all_semantics = []
        sensors = self.input_connections[0].from_node
        print(len(sensors))
        [all_semantics.append(sensor.semantics) for sensor in sensors]
        semantics_array = np.array(all_semantics).reshape((32, 32, 3, sensors[0].semantics.shape[-1]))
        input_pic_array = semantics_array[:, :, :, 0]
        self.get_sizes(input_pic_array)
        dataset_whole_output = np.zeros((self.output_width, self.output_length, self.dimenstions, semantics_array.shape[3]))
        for s in range(semantics_array.shape[3]):
            print(s)
            input_pic_array = semantics_array[:, :, :, s]
            self.get_sizes(input_pic_array)
            whole_output = np.zeros((self.output_width, self.output_length, self.dimenstions))

            dim_output = np.zeros((self.output_width, self.output_length, self.dimenstions))
            col = 0
            for j in range(0, self.input_length - self.kernel_size + 1, self.stride):
                row_output = np.zeros((self.output_width,))
                row = 0
                for i in range(0, self.input_length - self.kernel_size + 1, self.stride):
                    new_array = input_pic_array[i:(i + self.kernel_size), j:(j + self.kernel_size), d]
                    output = np.multiply(new_array, self.filter)
                    row_output[row] = np.sum(output)
                    row += 1
                dim_output[col, :] = row_output
                col += 1


            dataset_whole_output[:, :, :, s] = whole_output
        return dataset_whole_output

    def convolv2(self):
        all_semantics = []
        sensors = self.input_connections[0].from_node
        print(len(sensors))
        [all_semantics.append(sensor.semantics) for sensor in sensors]
        all_semantics = all_semantics[0]
        semantics_array = np.array(all_semantics).reshape((all_semantics.shape[0], all_semantics.shape[1], 3, sensors[0].semantics.shape[-1]))
        input_pic_array = semantics_array[:, :, :, 0]
        self.get_sizes(input_pic_array)
        dataset_whole_output = np.zeros((self.output_width, self.output_length, self.dimenstions, semantics_array.shape[3]))
        for s in range(semantics_array.shape[3]):
            print(s)
            input_pic_array = semantics_array[:, :, :, s]
            self.get_sizes(input_pic_array)
            whole_output = np.zeros((self.output_width, self.output_length, self.dimenstions))
            for d in range(self.dimenstions):
                dim_output = np.zeros((self.output_width, self.output_length))
                col = 0
                for j in range(0, self.input_length - self.kernel_size + 1, self.stride):
                    row_output = np.zeros((self.output_width,))
                    row = 0
                    for i in range(0, self.input_length - self.kernel_size + 1, self.stride):
                        new_array = input_pic_array[i:(i + self.kernel_size), j:(j + self.kernel_size), d]
                        output = np.multiply(new_array, self.filter)
                        row_output[row] = np.sum(output)
                        row += 1
                    dim_output[col, :] = row_output
                    col += 1
                whole_output[:, :, d] = dim_output
            dataset_whole_output[:, :, :, s] = whole_output
        return dataset_whole_output

class ConvNeuron(Neuron):
    def __init__(self, semantics, input_connections, activation_function):
        super().__init__(semantics, input_connections, activation_function)
        self._initialize_parameters()

    def __copy__(self):
        copy_semantics = copy(self.semantics)
        copy_input_connections = copy(self.input_connections)
        copy_activation_function = self.activation_function
        return ConvNeuron(copy_semantics, copy_input_connections, copy_activation_function)

    def __deepcopy__(self, memodict={}):
        input_connections = deepcopy(self.input_connections, memodict)
        activation_function = self.activation_function
        semantics = np.array([])
        neuron = ConvNeuron(semantics, input_connections, activation_function)
        memodict[id(self)] = neuron
        return neuron

    def _initialize_parameters(self):
        self.nr_of_channel = 1
        self.kernel_size = np.random.randint(2, 7)
        self.stride = np.random.randint(1, 3)
        # prepare yourself for different number of channels not only RGB but also 1 dim black/white
        self.filter = np.random.uniform(-1, 1, (self.kernel_size, self.kernel_size, 3))

    def get_sizes(self, input_pic_array):
        self.input_width = input_pic_array.shape[0]
        self.input_length = input_pic_array.shape[1]
        self.dimenstions = input_pic_array.shape[2]
        self.output_width = int(np.ceil((input_pic_array.shape[0] - self.kernel_size + 1) / self.stride))
        self.output_length = int(np.ceil((input_pic_array.shape[1] - self.kernel_size + 1) / self.stride))

    def convolv(self):
        all_semantics = []
        sensors = self.input_connections[0].from_node
        # print(len(sensors))
        [all_semantics.append(sensor.semantics) for sensor in sensors]
        semantics_array = np.array(all_semantics).reshape((32, 32, 3, sensors[0].semantics.shape[-1]))
        input_pic_array = semantics_array[:, :, :, 0]
        self.get_sizes(input_pic_array)
        dataset_whole_output = np.zeros(
            (self.output_width, self.output_length, self.dimenstions, semantics_array.shape[3]))
        for s in range(semantics_array.shape[3]):
            # print(s)
            input_pic_array = semantics_array[:, :, :, s]
            self.get_sizes(input_pic_array)
            whole_output = np.zeros((self.output_width, self.output_length, self.dimenstions))
            col = 0
            for j in range(0, self.input_length - self.kernel_size + 1, self.stride):
                row_output = np.zeros((self.output_width, self.dimenstions))
                row = 0
                for i in range(0, self.input_length - self.kernel_size + 1, self.stride):
                    new_array = input_pic_array[i:(i + self.kernel_size), j:(j + self.kernel_size), :]

                    output = np.multiply(new_array, self.filter)
                    row_output[row, :] = np.sum(output, axis=(0,1))
                    row += 1
                whole_output[:, col, :] = row_output
                col += 1
            dataset_whole_output[:, :, :, s] = whole_output
        return dataset_whole_output

    def convolv2(self):
        all_semantics = []
        sensors = self.input_connections[0].from_node
        # print(len(sensors))
        [all_semantics.append(sensor.semantics) for sensor in sensors]
        all_semantics = all_semantics[0]
        semantics_array = np.array(all_semantics).reshape((all_semantics.shape[0], all_semantics.shape[1], 3, sensors[0].semantics.shape[-1]))
        input_pic_array = semantics_array[:, :, :, 0]
        self.get_sizes(input_pic_array)
        dataset_whole_output = np.zeros(
            (self.output_width, self.output_length, self.dimenstions, semantics_array.shape[3]))
        for s in range(semantics_array.shape[3]):
            # print(s)
            input_pic_array = semantics_array[:, :, :, s]
            self.get_sizes(input_pic_array)
            whole_output = np.zeros((self.output_width, self.output_length, self.dimenstions))
            col = 0
            for j in range(0, self.input_length - self.kernel_size + 1, self.stride):
                row_output = np.zeros((self.output_width, self.dimenstions))
                row = 0
                for i in range(0, self.input_length - self.kernel_size + 1, self.stride):
                    new_array = input_pic_array[i:(i + self.kernel_size), j:(j + self.kernel_size), :]

                    output = np.multiply(new_array, self.filter)
                    row_output[row, :] = np.sum(output, axis=(0,1))
                    row += 1
                whole_output[:, col, :] = row_output
                col += 1
            dataset_whole_output[:, :, :, s] = whole_output
        return dataset_whole_output


    def _calculate_output(self, weighted_input):
        """Calculates semantics, based on weighted input."""
        return calculate_output(weighted_input, self.activation_function)

    def calculate(self):
        """Calculates weighted input, then calculates semantics."""
        self.semantics = self.convolv()

    def calculate2(self):
        """Calculates weighted input, then calculates semantics."""
        self.semantics = self.convolv2()



class PoolNeuron(Neuron):
    def __init__(self, semantics, input_connections, activation_function):
        super().__init__(semantics, input_connections, activation_function)
        self._initialize_parameters()

    def __copy__(self):
        copy_semantics = copy(self.semantics)
        copy_input_connections = copy(self.input_connections)
        copy_activation_function = self.activation_function
        return PoolNeuron(copy_semantics, copy_input_connections, copy_activation_function)

    def __deepcopy__(self, memodict={}):
        input_connections = deepcopy(self.input_connections, memodict)
        activation_function = self.activation_function
        semantics = np.array([])
        neuron = PoolNeuron(semantics, input_connections, activation_function)
        memodict[id(self)] = neuron
        return neuron


    def _initialize_parameters(self):
        self.pool_size = np.random.randint(1, 5)
        self.operation = 'max'


    def get_sizes(self, input_pic_array):
        self.input_width = input_pic_array.shape[0]
        self.input_length = input_pic_array.shape[1]
        self.dimenstions = input_pic_array.shape[2]
        self.output_width = input_pic_array.shape[0] - self.pool_size + 1
        self.output_length = input_pic_array.shape[1] - self.pool_size + 1

    def pool(self):
        all_semantics = []
        sensors = self.input_connections[0].from_node
        [all_semantics.append(sensor.semantics) for sensor in sensors]
        semantics_array = np.array(all_semantics).reshape((int(np.sqrt(len(sensors)/3)), int(np.sqrt(len(sensors)/3)), 3, sensors[0].semantics.shape[-1]))
        input_pic_array = semantics_array[:, :, :, 0]
        self.get_sizes(input_pic_array)
        dataset_whole_output = np.zeros((self.output_width, self.output_length, self.dimenstions, semantics_array.shape[3]))
        for s in range(semantics_array.shape[3]):
            # print(s)
            input_pic_array = semantics_array[:, :, :, s]
            whole_output = np.zeros((self.output_width, self.output_length, self.dimenstions))
            col = 0
            for j in range(self.input_length-self.pool_size+1):
                row_output = np.zeros((self.output_width, self.dimenstions))
                row = 0
                for i in range(self.input_width-self.pool_size+1):
                    new_array = input_pic_array[i:(i + self.pool_size), j:(j + self.pool_size), :]
                    row_output[row, :] = np.max(new_array, axis=(0,1))
                    row += 1
                whole_output[:,col, :] = row_output
                col += 1
            dataset_whole_output[:, :, :, s] = whole_output
        return dataset_whole_output

    def pool2(self):
        all_semantics = []
        sensors = self.input_connections[0].from_node
        [all_semantics.append(sensor.semantics) for sensor in sensors]
        all_semantics = all_semantics[0]
        semantics_array = np.array(all_semantics).reshape((all_semantics.shape[0], all_semantics.shape[1], 3, sensors[0].semantics.shape[-1]))
        input_pic_array = semantics_array[:, :, :, 0]
        self.get_sizes(input_pic_array)
        dataset_whole_output = np.zeros((self.output_width, self.output_length, self.dimenstions, semantics_array.shape[3]))
        for s in range(semantics_array.shape[3]):
            # print(s)
            input_pic_array = semantics_array[:, :, :, s]
            whole_output = np.zeros((self.output_width, self.output_length, self.dimenstions))
            col = 0
            for j in range(self.input_length-self.pool_size+1):
                row_output = np.zeros((self.output_width, self.dimenstions))
                row = 0
                for i in range(self.input_width-self.pool_size+1):
                    new_array = input_pic_array[i:(i + self.pool_size), j:(j + self.pool_size), :]
                    row_output[row, :] = np.max(new_array, axis=(0, 1))
                    row += 1
                whole_output[:, col, :] = row_output
                col += 1
            dataset_whole_output[:, :, :, s] = whole_output
        return dataset_whole_output


    def _calculate_output(self, weighted_input):
        """Calculates semantics, based on weighted input."""
        return calculate_output(weighted_input, self.activation_function)

    def calculate(self):
        """Calculates weighted input, then calculates semantics."""
        self.semantics = self.pool()

    def calculate2(self):
        """Calculates weighted input, then calculates semantics."""
        self.semantics = self.pool2()


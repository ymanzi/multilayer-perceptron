import numpy as np
import random

"IMPLEMENT VARIABLE LEARNING RATE"

class Softmax:
	@staticmethod
	def fct(x: np.ndarray) -> np.ndarray:
		vec = np.exp(x)
		return vec / np.sum(vec)
	
	@staticmethod
	def derivative(x):
		sm = Softmax.fct(x)
		return sm * (1.0 - sm)

class Sigmoid:
	@staticmethod
	def fct(x: np.ndarray) -> np.ndarray:
		if x.size == 0:
			return None
		x = x.astype(float)
		if x.ndim == 0:
			x = np.array(x, ndmin=1)
		return (1.0 / (1.0 + (np.exp(-x))))

	@staticmethod
	def derivative(x):
		sig = Sigmoid.fct(x)
		return sig * (1.0 - sig)

class Tanh:
	@staticmethod
	def fct(x: np.ndarray) -> np.ndarray:
		pos_val = np.exp(x)
		neg_val = np.exp(-x)
		return (pos_val - neg_val) / (pos_val + neg_val)

	@staticmethod
	def derivative(x):
		t = Tanh.fct(x)
		return 1.0 - t**2

class ReLu:
	@staticmethod
	def fct(x: np.ndarray) -> np.ndarray:
		return np.maximum(0.01 * x, x)
	
	@staticmethod
	def derivative(x):
		return np.where(x < 0, 0.01, 1)

def sigmoid(x: np.ndarray) -> np.ndarray:
	if x.size == 0:
		return None
	x = x.astype(float)
	if x.ndim == 0:
		x = np.array(x, ndmin=1)
	return (1.0 / (1.0 + (np.exp(-x))))

def sigmoid_derivative(x):
	sig = sigmoid(x)
	return sig * (1.0 - sig)

class MSE(object):
	"""
		def sigmoid_derivative(x):
			sig = sigmoid(x)
			return sig * (1.0 - sig)
	"""

	@staticmethod
	def value(a, y):
		"""Return the cost associated with an output ``a`` and desired output
		``y``.
		"""
		return 0.5*np.linalg.norm(a-y)**2 / a.shape[0]
		
	@staticmethod
	def delta(z, a, y):
		"""Return the error delta from the output layer."""
		return np.subtract(a, y) * sigmoid_derivative(z) ######## A MODIFIER

class CrossEntropyCost(object):

	@staticmethod
	def value(a, y):
		"""Return the cost associated with an output ``a`` and desired output
		``y``.  Note that np.nan_to_num is used to ensure numerical
		stability for log close to 0 values.  In particular, if both ``a`` and ``y`` have a 1.0
		in the same slot, then the expression (1-y)*np.log(1-a)
		returns nan.  The np.nan_to_num ensures that that is converted
		to the correct value (0.0).
		"""
		return np.sum(np.nan_to_num(-y*np.log(a + 1e-15)-(1-y)*np.log(1-a + 1e-15))) / a.shape[0]

	@staticmethod
	def delta(z, a, y):
		"""Return the error delta from the output layer.  Note that the
        parameter ``z`` is not used by the method.  It is included in
        the method's parameters in order to make the interface
        consistent with the delta method for other cost classes.

		C = −[ylna+(1−y)ln(1−a)]
        """
		return np.subtract(a, y)

class Weight_init:
	@staticmethod
	def std(layers):
		return [np.random.randn(x, y) for x,y in zip(layers[1:], layers[:-1])]

	@staticmethod
	def xavier(layers):
		return [np.random.randn(x, y)/np.sqrt(y + x) * np.sqrt(6) for x,y in zip(layers[1:], layers[:-1])]

class Network(object):
	def __init__(self, layers, cost=CrossEntropyCost, hidden_activation=Sigmoid, output_activation=Sigmoid, w_init='std'):
		''' 
			Exemple of size: [2, 3, 1] 
			if we want to create a Network object with 
				2 neurons in the first layer, 
				3 neurons in the second layer, and 
				1 neuron in the final layer

			in the weights initialization we divide by 'np.sqrt(x)'
			to minimize the value of z because if the activation function
			is sigmoid and we don't do that and x is a large number of vector,
			there are a lot of chances that z will be large numbers and sigmoid(z)
			will saturate in the beginning.
			As the sigmoid graph show it, if z << 0 or z >> 1, a small change in the input
			give a small change in the output, we say the neuron is 'saturated'
		'''
		self.layers = layers
		self.nb_layers = len(layers)
		if w_init == 'std':
			self.weights = Weight_init.std(layers)
		elif w_init == 'xavier':
			self.weights = Weight_init.xavier(layers)
		self.biases = [np.random.randn(x, 1) for x in layers[1:]]
		self.cost = cost
		self.list_training_cost = []
		self.list_test_cost = []
		self.output_a = output_activation
		self.hidden_a = hidden_activation

	# def quadratic_cost_derivative(self, output_activations, y):
	# 	return np.subtract(output_activations, y)

	def feedforward(self, a):
		"""Return the output of the network if "a" is input.
			a′=σ(wa+b)
		"""
		for weight, bias in zip(self.weights[:-1], self.biases[:-1]):
			# a = sigmoid(np.add(np.dot(weight, a), bias))
			a = self.hidden_a.fct(np.add(np.dot(weight, a), bias))
		a = self.output_a.fct(np.add(np.dot(self.weights[-1], a), self.biases[-1]))
		return a

	def backpropagation(self, x, y):
		"""
		Return a tuple ``(nabla_b, nabla_w)`` representing the
		gradient for the cost function C_x.  ``nabla_b`` and
		``nabla_w`` are layer-by-layer lists of numpy arrays, similar
		to ``self.biases`` and ``self.weights``.

		z = wa + b
		a′=σ(wa+b)  activation function
		"""
		nabla_w = [np.zeros(w.shape) for w in self.weights]
		nabla_b = [np.zeros(b.shape) for b in self.biases]

		#feedforward
		list_activation = [x]
		list_z = []
		a = x
		for weight, bias in zip(self.weights[:-1], self.biases[:-1]):
			z = np.add(np.dot(weight, a), bias)
			# a = sigmoid(z)
			a = self.hidden_a.fct(z)
			list_activation.append(a)
			list_z.append(z)
		z = np.add(np.dot(self.weights[-1], a), self.biases[-1])
		a = self.output_a.fct(z)
		list_activation.append(a)
		list_z.append(z)

		delta = self.cost.delta(list_z[-1], list_activation[-1], y)
		nabla_b[-1] = delta
		nabla_w[-1] = np.dot(delta, list_activation[-2].transpose())
		for l in range(2, self.nb_layers):
			z = list_z[-l]
			# delta = np.dot(self.weights[-l + 1].transpose(), delta) * sigmoid_derivative(z)
			delta = np.dot(self.weights[-l + 1].transpose(), delta) * self.hidden_a.derivative(z)
			nabla_b[-l] = delta
			nabla_w[-l] = np.dot(delta, list_activation[-l -1].transpose())
		return (nabla_w, nabla_b)
	
	def train_(self, training_data, epochs, mini_batch_size, learning_rate = 5.0,\
			lambda_=0.0, test_data=None, n_epoch_early_stop = 0):
		"""Train the neural network using mini-batch stochastic
		gradient descent.  The "training_data" is a list of tuples
		"(x, y)" representing the training inputs and the desired
		outputs.  The other non-optional parameters are
		self-explanatory.  If "test_data" is provided then the
		network will be evaluated against the test data after each
		epoch, and partial progress printed out.  This is useful for
		tracking progress, but slows things down substantially."""
		
		#EarlyStop Initialize
		best_accuracy = 1
		no_change_best_accuracy = 0

		best_cost = 1e20
		no_change_best_cost = 0

		training_data = list(training_data)
		training_size = len(training_data)
		if test_data:
			test_data = list(test_data)
			test_size = len(test_data)
		for j in range(epochs):
			random.shuffle(training_data)
			for n in range(0, training_size, mini_batch_size):
				self.update_minibatch(training_data[n: n + mini_batch_size], learning_rate, lambda_, training_size)
			if test_data:
				accuracy = self.evaluate(test_data)
				test_cost = self.get_cost(test_data)
				training_cost = self.get_cost(training_data)
				self.list_test_cost.append(test_cost)
				self.list_training_cost.append(training_cost)

				if test_cost < best_cost:
					best_cost = test_cost
					no_change_best_cost = 0
				else:
					no_change_best_cost += 1
					if no_change_best_cost == 2:
						best_cost = test_cost
				
				# if no_change_best_cost > 2 and learning_rate > 0.05:
				# 	learning_rate /= 10
				# 	no_change_best_cost = 0
				
				print("Epoch {}: {} / {} Training Cost: {}  Test Cost: {}  learning_rate: {}".format(
					j, accuracy, test_size, training_cost, test_cost, learning_rate))
				if n_epoch_early_stop > 0:
					if best_accuracy < accuracy:
						best_accuracy = accuracy
						no_change_best_accuracy = 0
					else:
						no_change_best_accuracy += 1
					if no_change_best_accuracy == n_epoch_early_stop:
						print("Early stop activated")
						return (self.list_training_cost, self.list_test_cost)
			else:
				print("Epoch {0} complete".format(j))
		return (self.list_training_cost, self.list_test_cost)

	
	def update_minibatch(self, batch, learning_rate, lambda_, n):
		"""
			Update the network's weights and biases by applying
			gradient descent using backpropagation to a single mini batch.
			The "mini_batch" is a list of tuples "(x, y)", and "learning_rate"
			is the learning rate.
			``lambda_`` is the L2 regularization parameter who reduce overfitting,
			and ``n`` is the total size of the training data set.
		"""

		nabla_w = [np.zeros(w.shape) for w in self.weights]
		nabla_b = [np.zeros(b.shape) for b in self.biases]

		for x, y in batch:
			delta_nabla_w, delta_nabla_b = self.backpropagation(x, y)
			nabla_b = [nb + dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
			nabla_w = [nw + dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]
		self.weights = [(1-(learning_rate* lambda_ /n))*w - (learning_rate / len(batch) * dw) for w, dw in zip(self.weights, nabla_w)]
		self.biases = [b - (learning_rate / len(batch) * db) for b, db in zip(self.biases, nabla_b)]


	def evaluate(self, test_data):
		"""Return the number of test inputs for which the neural
		network outputs the correct result. Note that the neural
		network's output is assumed to be the index of whichever
		neuron in the final layer has the highest activation."""
		test_results = [(np.argmax(self.feedforward(x)), np.argmax(y))
						for (x, y) in test_data]
		return sum(int(x == y) for (x, y) in test_results)

	def get_cost(self, test_data):
		"""Return the cost. Note that the neural
		network's output is assumed to be the index of whichever
		neuron in the final layer has the highest activation."""

		list_res = [(self.feedforward(x), [np.argmax(y)]) for (x,y) in test_data]
		test_results = list(zip(*[(1 - x[0], y) if np.argmax(x) == 0 else (x[1], y)
						for (x, y) in list_res]))
		a = np.array(test_results[0])
		y = np.array(test_results[1])
		
		return self.cost.value(a , y)

# nn = Network([3, 2, 1])
# print(nn.feedforward(np.random.randn(3, 1)), "\n\n")
# print(nn.weights)
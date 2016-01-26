import numpy as np
import scipy.optimize
from sklearn import preprocessing


def computeNNgradients(costfunc, weights):
    """
    Compute numerical estimate of gradient for testing
    the implementation of NN
    :param costfunc: reference to the cost function
    :param weights: parameters for the network
    :return:
    """
    grad = np.zeros(weights.shape)
    epsilon_matrix = np.zeros(weights.shape)
    epsilon = 1e-4
    for p in range(len(weights)):
        epsilon_matrix[p] = epsilon
        loss1 = costfunc(weights - epsilon_matrix)[0]
        loss2 = costfunc(weights + epsilon_matrix)[0]
        grad[p] = (loss2 - loss1) / 2 / epsilon
        epsilon_matrix[p] = 0
    return grad


class NeuralNetwork:
    def __init__(self, X, y, num_labels, num_features, dataset_size, hidden_layer_size, lambda_value):
        """
        Neural network with 1 hidden layer using Sigmoid function
        :param X: a np.array size m x n
        :param y: a np.array size m x 1 range from 1 to num_labels
        :param num_labels:
        :param num_features:
        :param dataset_size:
        :param hidden_layer_size:
        :param lambda_value: regularization constant
        :return:
        """
        self.X = X
        self.n = num_features
        self.y = y
        self.K = num_labels
        self.m = dataset_size
        self.hidden_layer_size = hidden_layer_size
        self.lambda_value = lambda_value
        self.trained_weights = None
        self.all_weights = None

    def random_init_weights(self, row, col):
        """
        Randomly initialize weights for symmetry breaking
        :param row:
        :param col:
        :return: a np array in range of -epsilon to epsilon
        """
        epsilon_init = 0.12
        weights = np.random.rand(row, col) * 2 * epsilon_init - epsilon_init
        scaler = preprocessing.StandardScaler().fit(weights)
        weights = scaler.transform(weights)
        normalizer = preprocessing.Normalizer().fit(weights)
        weights = normalizer.transform(weights)
        return weights

    def sigmoid(self, numbers):
        """
        Compute the sigmoid function
        :param numbers: a np.array of floats
        :return: a np.array of floats
        """
        return 1.0 / (1.0 + np.exp(-numbers))

    def sigmoid_gradient(self, numbers):
        """
        Compute the sigmoid gradient
        :param numbers: a np.array
        :return:
        """
        return np.multiply(self.sigmoid(numbers), np.ones(numbers.shape, dtype=float)-self.sigmoid(numbers))

    def insert_bias_values(self, array, value):
        """
        Insert bias values to dataset array
        :param array: 1D or 2D np.array
        :param value: float
        :return:
        """
        try:
            biased = np.array([[0.0]*(len(array[0.0])+1)]*len(array))
            for j in range(len(biased)):
                biased[j] = np.insert(array[j], 0, value)
        except TypeError:
            biased = np.insert(array, 0, value)
        return biased

    def recode_y(self):
        """
        Recode y so y is in the form [[1], [0]] if y is not me,
        otherwise [[0], [1]]
        :return:
        """
        y_recoded = np.array([[0.0]*self.K]*self.m)
        for j in range(len(self.y)):
            y_recoded[j] = np.array(range(1, self.K+1)==self.y[j])
        return y_recoded

    def cost_function(self, weights, onlyJ=False):
        """
        Compute error
        :param weights: parameters for the network
        :param onlyJ: True if we only need to compute J
        :return:
        """
        theta1 = np.reshape(weights[:self.hidden_layer_size*(self.n+1)], (self.hidden_layer_size, self.n+1))
        theta2 = np.reshape(weights[self.hidden_layer_size*(self.n+1):], (self.K, self.hidden_layer_size+1))
        assert theta1.shape == (self.hidden_layer_size, self.n+1)
        assert theta2.shape == (self.K, self.hidden_layer_size+1)
        J = 0
        X_biased = self.insert_bias_values(self.X, 1)
        y_recoded = self.recode_y()

        # Cost function J
        a2 = self.sigmoid(np.dot(X_biased, np.transpose(theta1)))
        a2_biased = self.insert_bias_values(a2, 1)
        a3 = self.sigmoid(np.dot(a2_biased, np.transpose(theta2)))
        J -= np.multiply(y_recoded, np.log(a3)) + np.multiply((1-y_recoded), np.log(1-a3))
        J = np.sum(J[:, 0:self.K])
        J /= self.m

        # Add regularization term to J
        regu_term = 0
        for j in range(self.hidden_layer_size):
            for k in range(1, self.n+1):
                regu_term += theta1[j][k] ** 2
        for j1 in range(self.K):
            for k1 in range(1, self.hidden_layer_size+1):
                regu_term += theta2[j1][k1] ** 2
        regu_term = regu_term * self.lambda_value / 2 / self.m
        J += regu_term

        if onlyJ:
            return J
        else:
            pass

        # Back propagation
        theta1_grad = np.zeros(theta1.shape)
        theta2_grad = np.zeros(theta2.shape)
        Delta1 = np.zeros(theta1.shape)
        Delta2 = np.zeros(theta2.shape)
        for t in range(self.m):
            # Fast forward
            z2 = np.dot(theta1, np.transpose(X_biased[t]))
            a2 = self.sigmoid(z2)
            a2 = self.insert_bias_values(a2, 1)
            a3 = self.sigmoid(np.dot(theta2, a2))
            z2 = self.insert_bias_values(z2, 1)

            # Back prop
            delta3 = a3 - np.transpose(y_recoded[t])
            delta2 = np.multiply(np.dot(np.transpose(theta2), delta3), self.sigmoid_gradient(z2))
            delta2 = delta2[1:]
            Delta2 = Delta2 + np.outer(delta3, a2)
            Delta1 = Delta1 + np.outer(delta2, X_biased[t])

        for j in range(theta1.shape[0]):
            for k in range(theta1.shape[1]):
                if k == 0:
                    theta1_grad[j][k] = Delta1[j][k] / self.m
                else:
                    theta1_grad[j][k] = (Delta1[j][k] + theta1[j][k] * self.lambda_value) / self.m

        for j in range(theta2.shape[0]):
            for k in range(theta2.shape[1]):
                if k == 0:
                    theta2_grad[j][k] = Delta2[j][k] / self.m
                else:
                    theta2_grad[j][k] = (Delta2[j][k] + theta2[j][k] * self.lambda_value) / self.m

        grad = np.append(np.array(theta1_grad.flat), np.array(theta2_grad.flat))
        return J, grad

    def query_J(self, weights):
        """
        Return cost function J
        :param weights: parameters
        :return:
        """
        J = self.cost_function(weights, onlyJ=True)
        return J

    def query_grad(self, weights):
        """
        Return gradients
        :param weights: parameters
        :return:
        """
        J, grad = self.cost_function(weights)
        return grad

    def train(self, iterations):
        """
        Train the network for a number of iterations using fmin_cg
        :param iterations:
        :return:
        """
        initial_theta1 = self.random_init_weights(self.hidden_layer_size, self.n+1)
        initial_theta2 = self.random_init_weights(self.K, self.hidden_layer_size+1)
        initial_weights = np.append(np.array(initial_theta1.flat), np.array(initial_theta2.flat))
        self.trained_weights, self.all_weights = scipy.optimize.fmin_cg(self.query_J, initial_weights,
                                                                        fprime=self.query_grad,
                                                                        retall=True, maxiter=iterations)

    def predict(self, x):
        """
        Predict using the trained parameters
        :param x:
        :return:
        """
        if self.trained_weights is None:
            print "Network has not been trained"
            return
        else:
            theta1 = np.reshape(self.trained_weights[:self.hidden_layer_size*(self.n+1)],
                                (self.hidden_layer_size, self.n+1))
            theta2 = np.reshape(self.trained_weights[self.hidden_layer_size*(self.n+1):],
                                (self.K, self.hidden_layer_size+1))
            assert theta1.shape == (self.hidden_layer_size, self.n+1)
            assert theta2.shape == (self.K, self.hidden_layer_size+1)
            x_biased = self.insert_bias_values(x, 1)
            h1 = self.sigmoid(np.dot(x_biased, np.transpose(theta1)))
            h1_biased = self.insert_bias_values(h1, 1)
            h2 = self.sigmoid(np.dot(h1_biased, np.transpose(theta2)))
            return h2

    def get_error_values(self):
        """
        Get all the error values for debugging
        :return:
        """
        J_values = []
        for w in self.all_weights:
            J_values.append(self.cost_function(w, True))
        return J_values

    def interpret_result(self, result):
        """
        Interpret predicted label by giving out the most likely label
        :param result:
        :return:
        """
        result = list(result)
        return result.index(max(result)) + 1


# if __name__ == "__main__":
#     X = np.array([[1, 2, 3, 4],
#                  [5, 6, 7, 8],
#                  [9, 10, 11, 12]])
#     y = np.array([[2], [1], [2]])
#     t1 = np.array([[0.1, 0.2, 0.3, 0.4, 0.5],
#                    [0.6, 0.7, 0.8, 0.9, 0]])
#     t2 = np.array([[0.1, 0.2, 0.3],
#                    [0.4, 0.5, 0.6]])
#     w = np.append(np.array(t1.flat), np.array(t2.flat))
#     nn = NeuralNetwork(X, y, 2, 4, 3, 2, 0.3)
#     print nn.cost_function(w)

import math
import numpy as np
import tensorflow as tf
from tensorflow.python.framework import ops


class Cnn:
    def __init__(self):
        self.X = None
        self.Y = None
        self.Z3 = None
        self.cost = None

        self.parameters = {}

    def create_placeholders(self, n_H0, n_W0, n_C0, n_y):
        # Creates tensorflow variables for data X(input) and Y(labels)
        self.X = tf.placeholder(tf.float32, shape=(None, n_H0, n_W0, n_C0))
        self.Y = tf.placeholder(tf.float32, shape=(None, n_y))

    def initialize_parameters(self):
        # Initialize variables W1 and W2
        W1 = tf.get_variable('W1', [4, 4, 3, 8], initializer=tf.contrib.layers.xavier_initializer(seed=0))
        W2 = tf.get_variable('W2', [2, 2, 8, 16], initializer=tf.contrib.layers.xavier_initializer(seed=0))
        # Set self.parameters
        self.parameters = {"W1": W1,
                      "W2": W2}

    def forward_propagation(self, X, parameters):
        """
        Implements the forward propagation for the model:
        CONV2D -> RELU -> MAXPOOL -> CONV2D -> RELU -> MAXPOOL -> FLATTEN -> FULLYCONNECTED

        Arguments:
        X -- input dataset placeholder, of shape (input size, number of examples)
        parameters -- python dictionary containing your parameters "W1", "W2"
                      the shapes are given in initialize_parameters
        """

        # Retrieve the parameters from the dictionary "parameters"
        W1 = self.parameters['W1']
        W2 = self.parameters['W2']

        # CONV2D: stride of 1, padding 'SAME'
        Z1 = tf.nn.conv2d(X, W1, strides=[1, 1, 1, 1], padding='SAME')
        # RELU
        A1 = tf.nn.relu(Z1)
        # MAXPOOL: window 8x8, sride 8, padding 'SAME'
        P1 = tf.nn.max_pool(A1, ksize=[1, 8, 8, 1], strides=[1, 8, 8, 1], padding='SAME')
        # CONV2D: filters W2, stride 1, padding 'SAME'
        Z2 = tf.nn.conv2d(P1, W2, strides=[1, 1, 1, 1], padding='SAME')
        # RELU
        A2 = tf.nn.relu(Z2)
        # MAXPOOL: window 4x4, stride 4, padding 'SAME'
        P2 = tf.nn.max_pool(A2, ksize=[1, 4, 4, 1], strides=[1, 4, 4, 1], padding='SAME')
        # FLATTEN
        P2 = tf.contrib.layers.flatten(P2)
        # FULLY-CONNECTED without non-linear activation function (not not call softmax).
        self.Z3 = tf.contrib.layers.fully_connected(P2, 6, activation_fn=None)

    def compute_cost(self, Z3, Y):
        """
        Computes the cost

        Arguments:
        Z3 -- output of forward propagation (output of the last LINEAR unit), of shape (6, number of examples)
        Y -- "true" labels vector placeholder, same shape as Z3

        Returns:
        cost - Tensor of the cost function
        """
        self.cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=Z3, labels=Y))

    def random_minibatches(self, X, Y, batch_size):
        # Naive minibatch generator
        steps = [(i*batch_size)+batch_size for i in range(int(math.ceil(len(X)/64)))]
        minibatches = [(X[j-batch_size:j], Y[j-batch_size:j]) for j in steps]
        np.random.shuffle(minibatches)
        return minibatches

    def model(self, X_train, Y_train, X_test, Y_test, learning_rate=0.009,
              num_epochs=100, minibatch_size=64, print_cost=True):
        """
        Implements a three-layer ConvNet in Tensorflow:
        CONV2D -> RELU -> MAXPOOL -> CONV2D -> RELU -> MAXPOOL -> FLATTEN -> FULLYCONNECTED

        Arguments:
        X_train -- training set, of shape (None, 64, 64, 3)
        Y_train -- test set, of shape (None, n_y = 6)
        X_test -- training set, of shape (None, 64, 64, 3)
        Y_test -- test set, of shape (None, n_y = 6)
        learning_rate -- learning rate of the optimization
        num_epochs -- number of epochs of the optimization loop
        minibatch_size -- size of a minibatch
        print_cost -- True to print the cost every 100 epochs

        Returns:
        train_accuracy -- real number, accuracy on the train set (X_train)
        test_accuracy -- real number, testing accuracy on the test set (X_test)
        parameters -- parameters learnt by the model. They can then be used to predict.
        """

        ops.reset_default_graph()  # to be able to rerun the model without overwriting tf variables
        (m, n_H0, n_W0, n_C0) = X_train.shape
        n_y = Y_train.shape[1]
        costs = []  # To keep track of the cost

        # Create Placeholders of the correct shape
        self.create_placeholders(n_H0, n_W0, n_C0, n_y)
        X, Y = self.X, self.Y

        # Initialize parameters
        self.initialize_parameters()

        # Forward propagation: Build the forward propagation in the tensorflow graph
        self.forward_propagation(X, self.parameters)
        Z3 = self.Z3

        # Cost function: Add cost function to tensorflow graph
        self.compute_cost(Z3, Y)
        cost = self.cost

        # Backpropagation: Define the tensorflow optimizer. Use an AdamOptimizer that minimizes the cost.
        optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost)

        # Initialize all the variables globally
        init = tf.global_variables_initializer()

        # Start the session to compute the tensorflow graph
        with tf.Session() as sess:

            # Run the initialization
            sess.run(init)

            # Do the training loop
            for epoch in range(num_epochs):

                minibatch_cost = 0.
                num_minibatches = int(
                    m / minibatch_size)  # number of minibatches of size minibatch_size in the train set
                minibatches = self.random_mini_batches(X_train, Y_train, minibatch_size)

                for minibatch in minibatches:
                    # Select a minibatch
                    (minibatch_X, minibatch_Y) = minibatch
                    # Run the session to execute the optimizer and the cost
                    _, temp_cost = sess.run([optimizer, cost], feed_dict={X: minibatch_X, Y: minibatch_Y})

                    minibatch_cost += temp_cost / num_minibatches

                # Print the cost every epoch
                if print_cost == True and epoch % 5 == 0:
                    print("Cost after epoch %i: %f" % (epoch, minibatch_cost))
                if print_cost == True and epoch % 1 == 0:
                    costs.append(minibatch_cost)

            # Calculate the correct predictions
            predict_op = tf.argmax(Z3, 1)
            correct_prediction = tf.equal(predict_op, tf.argmax(Y, 1))

            # Calculate accuracy on the test set
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
            print(accuracy)
            train_accuracy = accuracy.eval({X: X_train, Y: Y_train})
            test_accuracy = accuracy.eval({X: X_test, Y: Y_test})
            print("Train Accuracy:", train_accuracy)
            print("Test Accuracy:", test_accuracy)

            return train_accuracy, test_accuracy, self.parameters

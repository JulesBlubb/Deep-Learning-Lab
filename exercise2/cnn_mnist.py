# from __future__ import print_function

import argparse
import gzip
import json
import os
import pickle

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

def one_hot(labels):
    """this creates a one hot encoding from a flat vector:
    i.e. given y = [0,2,1]
     it creates y_one_hot = [[1,0,0], [0,0,1], [0,1,0]]
    """
    classes = np.unique(labels)
    n_classes = classes.size
    one_hot_labels = np.zeros(labels.shape + (n_classes,))
    for c in classes:
        one_hot_labels[labels == c, c] = 1
    return one_hot_labels


def mnist(datasets_dir='./data'):
    if not os.path.exists(datasets_dir):
        os.mkdir(datasets_dir)
    data_file = os.path.join(datasets_dir, 'mnist.pkl.gz')
    if not os.path.exists(data_file):
        print('... downloading MNIST from the web')
        try:
            import urllib
            urllib.urlretrieve('http://google.com')
        except AttributeError:
            import urllib.request as urllib
        url = 'http://www.iro.umontreal.ca/~lisa/deep/data/mnist/mnist.pkl.gz'
        urllib.urlretrieve(url, data_file)

    print('... loading data')
    # Load the dataset
    f = gzip.open(data_file, 'rb')
    try:
        train_set, valid_set, test_set = pickle.load(f, encoding="latin1")
    except TypeError:
        train_set, valid_set, test_set = pickle.load(f)
    f.close()

    test_x, test_y = test_set
    test_x = test_x.astype('float32')
    test_x = test_x.astype('float32').reshape(test_x.shape[0], 28, 28, 1)
    test_y = test_y.astype('int32')
    valid_x, valid_y = valid_set
    valid_x = valid_x.astype('float32')
    valid_x = valid_x.astype('float32').reshape(valid_x.shape[0], 28, 28, 1)
    valid_y = valid_y.astype('int32')
    train_x, train_y = train_set
    train_x = train_x.astype('float32').reshape(train_x.shape[0], 28, 28, 1)
    train_y = train_y.astype('int32')
    print('... done loading data')
    return train_x, one_hot(train_y), valid_x, one_hot(valid_y), test_x, one_hot(test_y)


# 2 convolutional layer

class CNN(object):

    def __init__(self, num_filters, filter_size, batch_size):
        self.x_placeholder = tf.placeholder(tf.float32, shape=[None, 28, 28, 1])
        self.y_placeholder = tf.placeholder(tf.float32, shape=[None, 10])

        self.num_filters = num_filters
        self.filter_size = filter_size
        self.batch_size = batch_size

    def net_graph(self, x):
        # first Convolutional Layers (16 3x3 filter, stride 1) & ReLu
        conv1 = tf.layers.conv2d(
                inputs=x,
                filters=self.num_filters,
                kernel_size=self.filter_size,
                strides=1,
                padding="same",
                activation="relu")

        # max pooling (pool size 2)
        pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=2, strides=2)

        # Convolutional Layer #2 and Pooling Layer #2
        conv2 = tf.layers.conv2d(
                inputs=x,
                filters=self.num_filters,
                kernel_size=self.filter_size,
                strides=1,
                padding="same",
                activation="relu")

        pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=2, strides=2)

        flatten = tf.layers.flatten(pool2)

        # fully connected layer (128 units)
        fully_layer = tf.layers.dense(inputs=flatten, units=128, activation="relu")

        self.logits = tf.layers.dense(inputs=fully_layer, units=10, activation=None)

        return self.logits


    def train(self, x_train, y_train, x_valid, y_valid, num_epochs, lr, batch_size):
        # define the loss
        self.logits = self.net_graph(self.x_placeholder)
        self.out_soft = tf.nn.softmax(self.logits)

        # start Session
        init_var = tf.global_variables_initializer()
        self.sess = tf.Session()
        self.sess.run(init_var)

        # Cross Entropy Loss
        self.loss = tf.losses.softmax_cross_entropy(self.y_placeholder, self.logits)

        # Stochastic Gradient Descent
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=lr)
        train = optimizer.minimize(self.loss)

        # Mini Batches
        num_batches = x_train.shape[0] // batch_size
        valid_error = []

        for e in range(num_epochs):
            train_loss = 0
            for b in range(num_batches):
                # extracting a batch from x_train and y_train
                start = b * self.batch_size
                end = start + self.batch_size
                x_batch = x_train[start:end, ]
                y_batch = y_train[start:end, ]

                _, loss_value = self.sess.run([train, self.loss],
                                              {self.x_placeholder: x_batch,
                                               self.y_placeholder: y_batch})
                train_loss += (loss_value / num_batches)

            valid_loss = self.sess.run(self.loss, {self.x_placeholder: x_valid,
                                                    self.y_placeholder: y_valid})
            valid_acc = self.accuracy(x_valid, y_valid)

            print('Epoche: ', e , ',', 'Validation accuracy: ', valid_acc)
            valid_error.append(1 - valid_acc)

        return valid_error, valid_loss

    def accuracy(self, x, y_placeholder):
        y_pred = self.sess.run(self.logits, {self.x_placeholder: x})
        y_pred_int = np.argmax(y_pred, axis=1)
        y_placeholder_int = np.argmax(y_placeholder, axis=1)

        correct = np.sum(y_pred_int == y_placeholder_int)
        acc = correct / x.shape[0]

        return acc


def train_and_validate(x_train, y_train, x_valid, y_valid, num_epochs, lr, num_filters, batch_size, filter_size):
    # TODO: train and validate your convolutional neural networks with the
    model = CNN(num_filters, filter_size, batch_size)

    # train network
    learning_curve, valid_loss = model.train(x_train, y_train, x_valid, y_valid, num_epochs, lr, batch_size)

    return learning_curve, model , valid_loss # TODO: Return the validation error after each epoch (i.e learning curve) and your model


def test(x_test, y_test, model):
    # TODO: test your network here by evaluating it on the test data
    test_acc = model.accuracy(x_test, y_test)

    return 1 - test_acc


def plot_learning_curves(learning_curves, title, legend):
    # Plot Learning curves in one Plot
    plt.ylabel('validation-error')
    plt.xlabel('epochs')
    plt.title(title)
    for l in range(len(learning_curves)):
            plt.plot(learning_curves[l], label=legend['graph_labels'][l])
    plt.legend()
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_path", default="./", type=str, nargs="?",
                        help="Path where the results will be stored")
    parser.add_argument("--input_path", default="./", type=str, nargs="?",
                        help="Path where the data is located. If the data is not available it will be downloaded first")
    parser.add_argument("--learning_rate", default=1e-3, type=float, nargs="?", help="Learning rate for SGD")
    parser.add_argument("--num_filters", default=32, type=int, nargs="?",
                        help="The number of filters for each convolution layer")
    parser.add_argument("--batch_size", default=128, type=int, nargs="?", help="Batch size for SGD")
    parser.add_argument("--epochs", default=12, type=int, nargs="?",
                        help="Determines how many epochs the network will be trained")
    parser.add_argument("--run_id", default=0, type=int, nargs="?",
                        help="Helps to identify different runs of an experiments")
    parser.add_argument("--filter_size", default=3, type=int, nargs="?",
                        help="Filter width and height")
    args = parser.parse_args()

    # hyperparameters
    lr = args.learning_rate
    num_filters = args.num_filters
    batch_size = args.batch_size
    epochs = args.epochs
    filter_size = args.filter_size

    # train and test convolutional neural network
    x_train, y_train, x_valid, y_valid, x_test, y_test = mnist(args.input_path)


    # set num_filters = 16
    # Task 2, learning rates
    learning_rates = [0.1, 0.01, 0.001, 0.0001]

    learning_curves = []
    for l in learning_rates:
        learning_curve, model, valid_loss  = train_and_validate(x_train, y_train, x_valid, y_valid, epochs, l, num_filters=16, batch_size=128, filter_size=3)
        test_error = test(x_test, y_test, model)
        learning_curves.append(learning_curve)

    plot_learning_curves(learning_curves, "Different Learningrates in Stochastic Gradient Descent", {'graph_labels': learning_rates})

    # Task 3, filter_size
    filter_sizes = [1, 3, 5, 7]

    learning_curves_fs = []
    #for f in filter_sizes:
    #    learning_curve, model, valid_loss = train_and_validate(x_train, y_train, x_valid, y_valid, epochs, 0.1, num_filters=16,
    #                                           batch_size=128, filter_size=f)
    #    test_error = test(x_test, y_test, model)
    #    learning_curves_fs.append(learning_curve)

    #plot_learning_curves(learning_curves_fs, "Different Filter Sizes in Stochastic Gradient Descent", {'graph_labels': filter_sizes})


    # save results in a dictionary and write them into a .json file
    results = dict()
    results["lr"] = lr
    results["num_filters"] = num_filters
    results["batch_size"] = batch_size
    results["filter_size"] = filter_size
    results["learning_curve"] = learning_curve
    results["test_error"] = test_error

    path = os.path.join(args.output_path, "results")
    os.makedirs(path, exist_ok=True)

    fname = os.path.join(path, "results_run_%d.json" % args.run_id)

    fh = open(fname, "w")
    json.dump(results, fh)
    fh.close()

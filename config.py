import tensorflow as tf  


class Config(object):
    """
    define a class to store parameters,
    the input should be feature mat of training and testing
    """

    def __init__(self, X_train, X_test):
        # Input data
        self.train_count = len(X_train)  
        self.test_data_count = len(X_test)  
        self.n_steps = len(X_train[0])  

        # Trainging
        self.learning_rate = 0.0025
        self.lambda_loss_amount = 0.0015
        self.training_epochs = 2000
        self.batch_size = 1500

        # LSTM structure
        self.n_inputs = len(X_train[0][0])  
        self.n_hidden = 100
        self.n_classes = 3
        self.W = {
            'hidden': tf.Variable(tf.random_normal([self.n_inputs, self.n_hidden]), name="W_hidden"),
            'output': tf.Variable(tf.random_normal([self.n_hidden, self.n_classes]), name="W_output")
        }
        self.biases = {
            'hidden': tf.Variable(tf.random_normal([self.n_hidden], mean=1.0), name="b_hidden"),
            'output': tf.Variable(tf.random_normal([self.n_classes]), name="b_output")
        }
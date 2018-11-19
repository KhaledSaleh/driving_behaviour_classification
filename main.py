from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
from sklearn import metrics
from sys import argv
from argparse import ArgumentParser
from config import Config
from utils import *
from model import LSTM_Network


def main(args):
    if args.dataset == "full":
        X_train, X_test, y_train, y_test = load_full_dataset()
    elif args.dataset == "motorway":
        X_train, X_test, y_train, y_test = load_motorway_dataset()
    elsif args.dataset == "secondary":
        X_train, X_test, y_train, y_test = load_secondary_dataset()
    else:
        print("No valid dataset argument was set, will use the full dataset!")
        X_train, X_test, y_train, y_test = load_full_dataset()
        
    print('Training dataset shape: ', X_train.shape, y_train.shape)
    graph=tf.Graph()
    with graph.as_default():
      config = Config(X_train, X_test)
    
      X = tf.placeholder(tf.float32, [None, config.n_steps, config.n_inputs], name="X")
      Y = tf.placeholder(tf.float32, [None, config.n_classes], name="Y")

      pred_Y = LSTM_Network(X, config)

      # Loss,optimizer,evaluation
      l2 = config.lambda_loss_amount * \
          sum(tf.nn.l2_loss(tf_var) for tf_var in tf.trainable_variables())
      # Softmax loss and L2
      cost = tf.reduce_mean(
          tf.nn.softmax_cross_entropy_with_logits(pred_Y, Y), name="cost") + l2
      optimizer = tf.train.AdamOptimizer(
          learning_rate=config.learning_rate).minimize(cost)

      correct_pred = tf.equal(tf.argmax(pred_Y, 1), tf.argmax(Y, 1))
      accuracy = tf.reduce_mean(tf.cast(correct_pred, dtype=tf.float32))
      
      saver = tf.train.Saver()

    with tf.Session(graph=graph,config=tf.ConfigProto(log_device_placement=False)) as sess: 
      
      if args.test == False:
        init_op = tf.global_variables_initializer()
        sess.run(init_op)
        best_accuracy = 0.0
        # Start training for each batch and loop epochs
        for i in range(config.training_epochs):
            for start, end in zip(range(0, config.train_count, config.batch_size),
                                  range(config.batch_size, config.train_count + 1, config.batch_size)):
                sess.run(optimizer, feed_dict={X: X_train[start:end],
                                               Y: one_hot(y_train[start:end])})
                saver.save(sess, os.path.join(args.save_dir,'LSTM_model'))
            # Test completely at every epoch: calculate accuracy
            pred_out, accuracy_out, loss_out = sess.run([pred_Y, accuracy, cost], feed_dict={
                                                    X: X_test, Y: one_hot(y_test)})
            print("Training iter: {},".format(i)+\
                  " Test accuracy : {},".format(accuracy_out)+\
                  " Loss : {}".format(loss_out))
            best_accuracy = max(best_accuracy, accuracy_out)
        print("")
        print("Final test accuracy: {}".format(accuracy_out))
        print("Best epoch's test accuracy: {}".format(best_accuracy))
        print("")
      # start testing the trained model
      else:
          saver.restore(sess, os.path.join(args.save_dir,'LSTM_model'))
          pred_out, accuracy_out, loss_out = sess.run([pred_Y, accuracy, cost], feed_dict={
                                                  X: X_test, Y: one_hot(y_test)})
          print(" Test accuracy : {},".format(accuracy_out)+\
                " Loss : {}".format(loss_out))

      predictions = pred_out.argmax(1)
      print(metrics.classification_report(y_test, predictions))
      print(metrics.confusion_matrix(y_test, predictions))


if __name__ == '__main__':
    parser = ArgumentParser(description=__doc__)
    parser.add_argument('--save_dir', '-s',
                        help='Directory of (to be)-saved model',
                        default= 'saves')
    parser.add_argument('--dataset', '-d',
                        help='Which split of the dataset to train/test the model on?'\
                        '(i.e. full, motorway or secondary)',
                        default= 'full')
    parser.add_argument('--test', action='store_true',
                        help='Start testing the saved model in $save_dir$ '\
                        'othewrwise, it will start the training')
    args = parser.parse_args()
    main(args)

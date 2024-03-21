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

# 加载不同的数据集
def main(args):
    if args.dataset == "full":
        X_train, X_test, y_train, y_test = load_full_dataset()
    elif args.dataset == "motorway":
        X_train, X_test, y_train, y_test = load_motorway_dataset()
    elif args.dataset == "secondary":
        X_train, X_test, y_train, y_test = load_secondary_dataset()
    else:
        print("No valid dataset argument was set, will use the full dataset!")
        X_train, X_test, y_train, y_test = load_full_dataset()
        
    print('Training dataset shape: ', X_train.shape, y_train.shape)

    # 创建一个新的 TensorFlow 计算图
    graph=tf.Graph()

    # 将上下文环境设置为新创建的计算图，以便后续操作都在这个图中进行
    with graph.as_default():

      # 加载参数
      config = Config(X_train, X_test)
      # 创建一个 TensorFlow 占位符 X，用于接收输入数据，[None, config.n_steps, config.n_inputs]是数据形状
      X = tf.placeholder(tf.float32, [None, config.n_steps, config.n_inputs], name="X")
      # 标签同理
      Y = tf.placeholder(tf.float32, [None, config.n_classes], name="Y")

      # 调用 LSTM_Network 函数创建一个 LSTM 网络模型，并将输入数据 X 和配置对象 config 作为参数传递给该函数。函数返回的 pred_Y 是 LSTM 网络的输出结果
      pred_Y = LSTM_Network(X, config)

      # Loss,optimizer,evaluation

      # L2正则化项
      l2 = config.lambda_loss_amount * \
          sum(tf.nn.l2_loss(tf_var) for tf_var in tf.trainable_variables())

      # Softmax loss and L2
      # softmax交叉熵损失函数+L2正则化项
      cost = tf.reduce_mean(
          tf.nn.softmax_cross_entropy_with_logits(pred_Y, Y), name="cost") + l2
      # Adam优化器
      optimizer = tf.train.AdamOptimizer(
          learning_rate=config.learning_rate).minimize(cost)
      
      # 检查预测值中最大概率的索引是否与真实标签中最大值的索引相等，以确定每个样本的预测是否正确。
      correct_pred = tf.equal(tf.argmax(pred_Y, 1), tf.argmax(Y, 1))
        
      # 计算模型在整个数据集上的准确率。首先将 correct_pred 转换为浮点数类型，然后计算其平均值
      accuracy = tf.reduce_mean(tf.cast(correct_pred, dtype=tf.float32))

      # 创建一个 TensorFlow Saver 对象，用于保存和恢复模型的参数
      saver = tf.train.Saver()

    # 创建一个 TensorFlow 会话 sess，并指定使用之前创建的计算图 graph。同时通过 tf.ConfigProto 设置对象，关闭了日志记录设备放置信息。
    with tf.Session(graph=graph,config=tf.ConfigProto(log_device_placement=False)) as sess: 
        
      # 如果不是测试模式，则进行训练
      if args.test == False:

        #创建一个全局变量初始化的操作 init_op，用于初始化所有的模型参数
        init_op = tf.global_variables_initializer()
        # 执行初始化操作，初始化模型的所有参数
        sess.run(init_op)
        best_accuracy = 0.0

        # 开始训练每个批次和循环迭代
        for i in range(config.training_epochs):

            # 划分批次
            for start, end in zip(range(0, config.train_count, config.batch_size),
                                  range(config.batch_size, config.train_count + 1, config.batch_size)):
                # 通过 feed_dict 将训练数据传入模型中
                sess.run(optimizer, feed_dict={X: X_train[start:end],
                                               Y: one_hot(y_train[start:end])})
                                      
                # 每训练完一个批次后保存模型的参数，保存路径为 args.save_dir 下的 'LSTM_model'
                saver.save(sess, os.path.join(args.save_dir,'LSTM_model'))
                                      
            # Test completely at every epoch: calculate accuracy
            # 在每个 epoch 结束后，对测试数据进行测试，计算模型的预测值 pred_out、准确率 accuracy_out 和损失 loss_out
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
      # 测试模式
      else:
          # 恢复之前保存的训练好的模型，使得当前会话中的模型参数和之前保存的模型参数一致
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

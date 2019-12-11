#coding=utf-8
import tensorflow as tf
from tensorflow.contrib.losses.python.metric_learning import metric_loss_ops

class siamese:

    # Create model
    def __init__(self, training=True):
        self.x1 = tf.placeholder(tf.float32, [None, 128, 128, 3], name='input')
        self.x2 = tf.placeholder(tf.float32, [None, 128, 128, 3])
        # self.x3 = tf.placeholder(tf.float32, [None, 128, 128, 3])
        # self.x4 = tf.placeholder(tf.float32, [None, 128, 128, 3])
        self.training = training

        with tf.variable_scope("siamese") as scope:
            self.embeddings_anchor = tf.add( self.network(self.x1), 0.0, name='out_put')

            scope.reuse_variables()
            self.embeddings_positive = self.network(self.x2)

        # Create loss
        self.y_ = tf.placeholder(tf.float32, [None])

        self.l2loss , self.xent_loss = self.n_pairs_loss()
        self.loss = self.l2loss + self.xent_loss

    def n_pairs_loss(self):
        l2loss , xent_loss = metric_loss_ops.npairs_loss(
            labels= self.y_,
            embeddings_anchor= self.embeddings_anchor,
            embeddings_positive=self.embeddings_positive,
            reg_lambda=0.002)
        return l2loss , xent_loss

    def network(self, x):

        net = tf.layers.conv2d(x, filters=64, kernel_size=3, activation=tf.nn.relu,
                               strides=2, padding='valid', name='conv2_1', reuse=tf.AUTO_REUSE)

        net = tf.layers.max_pooling2d(net, pool_size=2, strides=2)

        net = tf.layers.conv2d(net, filters=128, kernel_size=4, activation=tf.nn.relu,
                               strides=1, padding='valid', name='conv2_2', reuse=tf.AUTO_REUSE)

        net = tf.layers.max_pooling2d(net, pool_size=2, strides=2)

        net = tf.layers.conv2d(net, filters=64, kernel_size=3,name='conv2_3', activation=tf.nn.relu,
                               strides=2, padding='valid', reuse=tf.AUTO_REUSE)

        net_max = tf.layers.max_pooling2d(net, pool_size=2, strides=2)

        net = tf.layers.conv2d(net_max, filters=96, kernel_size=2,name='conv2_4', activation=tf.nn.relu,
                               strides=2, padding='valid', reuse=tf.AUTO_REUSE)

        net = tf.layers.flatten(net, name='cnn_flatten')

        # 合起来
        net = tf.concat([net, tf.layers.flatten( net_max )], axis=1)
        net = self.fc_layer(net, 1024, "fc1")
        net = tf.nn.relu(net)

        net = self.fc_layer(net, 1024, "fc2")
        net = tf.nn.relu(net)

        net = self.fc_layer(net, 2, "fc3")

        return net

    def fc_layer(self, bottom, n_weight, name):
        assert len(bottom.get_shape()) == 2
        n_prev_weight = bottom.get_shape()[1]
        initer = tf.truncated_normal_initializer(stddev=0.01)
        W = tf.get_variable(name+'W', dtype=tf.float32, shape=[n_prev_weight, n_weight], initializer=initer)
        b = tf.get_variable(name+'b', dtype=tf.float32, initializer=tf.constant(0.01, shape=[n_weight], dtype=tf.float32))
        fc = tf.nn.bias_add(tf.matmul(bottom, W), b)
        return fc
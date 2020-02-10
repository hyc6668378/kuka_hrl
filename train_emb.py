#coding=utf-8
""" Siamese implementation using Tensorflow with MNIST example.
This siamese network embeds a 28x28 image (a point in 784D)
into a point in 2D.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from builtins import input

import tensorflow as tf
import numpy as np

config=tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True))

#import helpers
from alg.siamese_net import siamese
# import alg.visualize as visualize
from alg.toolkit import read_and_decode, show_embed, test_number
import os

# prepare data and tf.session
sess = tf.InteractiveSession(config=config)

batch_size=64
# class_num = len(os.listdir('image/'))
class_num = 4
epochs = 40
train_data = []
for i in range(class_num):
    train_data.append(read_and_decode(['tf_recoders/'+str(i) + '.tfrecords'], batch_size=batch_size, epochs=epochs))

test_data_size = test_number()
test_image, test_labels, test_iterator = read_and_decode(['tf_recoders/Train_data.tfrecords'], batch_size=test_data_size, epochs=2)

# setup siamese network
siamese = siamese(training=True)

global_step = tf.Variable(0, trainable=False)
initial_learning_rate = 3e-4 #初始学习率

learning_rate = tf.train.exponential_decay(initial_learning_rate,
                                           global_step=global_step,
                                           decay_steps=1000,decay_rate=0.95)
# update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)

assert tf.get_collection(tf.GraphKeys.UPDATE_OPS)==[] #确保没有 batch norml

train_step = tf.compat.v1.train.AdamOptimizer(learning_rate=learning_rate).minimize(siamese.loss, global_step=global_step)

saver = tf.train.Saver(max_to_keep=1)

# if you just want to load a previously trainmodel?
load = False
model_ckpt = 'model/embedding.ckpt.index'
if os.path.isfile(model_ckpt):
    input_var = None
    while input_var not in ['yes', 'no']:
        input_var = input("We found model files. Do you want to load it and continue training [yes/no]?")
    if input_var == 'yes':
        load = True

# init
sess.run(tf.global_variables_initializer())
sess.run(tf.local_variables_initializer())
if load: saver.restore(sess, 'model/embedding.ckpt')

for i in range( len( train_data) ):
    sess.run( train_data[i][-1].initializer )
sess.run(test_iterator.initializer)
_ = os.system("clear")
print('Global_variables:')
for var in tf.global_variables(): print(var)
print('-------------------------------\n')

print('Trainable_variables:')
for var in tf.trainable_variables(): print(var)
print('-------------------------------\n')
print("Test_data_size: {}".format(test_data_size))

anchor = []
positive = []
labels = []
siamese_cnn = {}
siamese_cnn_para = [var for var in tf.trainable_variables() if "siamese/conv" in var.name]
shuffle_index = np.arange( batch_size*len( train_data)//2 )

test_batch_x1, test_batch_y1 = sess.run([test_image, test_labels])

while True:
    try:
        batch_img_lab = [sess.run(train_data[i][0:2]) for i in range(len(train_data))]

        # split images to anchor and positive
        anchor = np.vstack(np.array([batch_img_lab[i][0][0:batch_size // 2] for i in range(len(train_data))]))
        positive = np.vstack(np.array([batch_img_lab[i][0][batch_size // 2:] for i in range(len(train_data))]))
        labels = np.ravel(np.array([batch_img_lab[i][1][0:batch_size // 2] for i in range(len(train_data))]))

        np.random.shuffle(shuffle_index)
        _, loss_v, l2loss, xent_loss, global_Step = sess.run([train_step, siamese.loss,
                              siamese.l2loss, siamese.xent_loss, global_step], feed_dict={
                                siamese.x1: anchor[shuffle_index],
                                siamese.x2: positive[shuffle_index],
                                siamese.y_: labels[shuffle_index]})

        if np.isnan(loss_v):
            print('Model diverged with loss = NaN')
            quit()

        if global_Step % 20 == 0:
            print ('step %d: total_loss %.3f \tl2loss %.3f \txent_loss %.3f ' % (global_Step, loss_v, l2loss, xent_loss))
    except (IndexError):
        continue
    except (tf.errors.OutOfRangeError, AttributeError): #, IndexError
        print('Training stop with step {}'.format(global_Step))

        for var in siamese_cnn_para:
            siamese_cnn[var.name] = sess.run(var)
            np.save('model/siamese_cnn.npy', siamese_cnn, allow_pickle=True)
        print('保存孪生网络cnn参数.')
        break

    if global_Step % 200 == 0 and global_Step > 0:
        saver.save(sess, 'model/embedding.ckpt')
        for var in siamese_cnn_para:
            siamese_cnn[var.name] = sess.run(var)
            np.save('model/siamese_cnn.npy', siamese_cnn, allow_pickle=True)
        print('保存孪生网络cnn参数.')
        # 遍历测试数据  stack在一起  分批过网络，因为一batch跑不完
        begin = 400
        embed = siamese.embeddings_anchor.eval({siamese.x1: test_batch_x1[0:begin]})
        while begin < test_data_size-401:
            embed = np.vstack((embed, siamese.embeddings_anchor.eval({siamese.x1: test_batch_x1[begin:begin+400]})))
            begin = begin+400
        embed = np.vstack((embed, siamese.embeddings_anchor.eval({siamese.x1: test_batch_x1[begin:]})))

        show_embed(embed, test_batch_y1, global_Step, class_num)

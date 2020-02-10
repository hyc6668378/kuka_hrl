#coding=utf-8
import tensorflow as tf
import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID" # see issue #152

import numpy as np
from alg.toolkit import read_and_decode, keras_model_to_frozen_graph
import argparse

def common_arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--recoder',  type=str, default='grasp_success_or_not')
    parser.add_argument('--test_acc_limit',  type=float, default=0.97)
    parser.add_argument('--input_shape',  type=int, default=32)
    parser.add_argument('--class_num',  type=int, default=2)

    return  parser.parse_args()

if __name__ == '__main__':
    """
    用keras训练分类器， 然后直接抽出计算图和参数固化成 .pd文件
    """

    args = common_arg_parser()

    img_size = 'big' if args.input_shape ==128 else 'little'

    path = 'tf_recoders/' + args.recoder
    train_dataset = read_and_decode(path +'_train.tfrecords',
                                    batch_size=64,
                                    img_size=img_size, dataset_as_output=True)
    test_dataset = read_and_decode(path+'_test.tfrecords',
                                   batch_size=1024,
                                   img_size=img_size, dataset_as_output=True)

    model = tf.keras.Sequential(
        [
            # 1.
            tf.keras.layers.Conv2D(input_shape=(args.input_shape, args.input_shape, 3),
                                   filters=32, kernel_size=5, strides=1, padding='same',
                                   activation= 'relu'),
            # tf.keras.layers.BatchNormalization(),
            # tf.keras.layers.ReLU(),
            tf.keras.layers.MaxPool2D(pool_size=2),

            # 2.
            tf.keras.layers.Conv2D(filters=32, kernel_size=30, strides=1, padding='same',
                                   activation='relu'),

            tf.keras.layers.Conv2D(filters=320, kernel_size=30, strides=1, padding='same',
                                   activation='relu'),
            tf.keras.layers.MaxPool2D(pool_size=2),

            tf.keras.layers.Conv2D(filters=32, kernel_size=3, strides=1, padding='same',
                                   activation='relu'),
            tf.keras.layers.MaxPool2D(pool_size=2),

            tf.keras.layers.Flatten(),

            tf.keras.layers.Dropout(rate=0.5),
            tf.keras.layers.Dense(128, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.3)),

            tf.keras.layers.Dropout(rate=0.5),
            tf.keras.layers.Dense(128, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.3)),

            tf.keras.layers.Dropout(rate=0.5),
            tf.keras.layers.Dense(args.class_num, activation='softmax' )
        ])

    model.compile(optimizer=tf.keras.optimizers.Adam(3e-4),
                  loss="sparse_categorical_crossentropy",
                  metrics=['accuracy'])

    print('\n\n-------------------------------------------')
    model.summary()
    step = 0
    while True:
        step += 2
        train_history = model.fit(train_dataset, epochs=2, verbose=1, workers=10)
        if step % 5 == 0:
            test_history = model.evaluate(test_dataset, verbose=1, workers=10)
            print('\n-----------------------------------------------')
            print('\nepoch:  {}   Test acc: {}\n'.format(step, test_history[1]))
            if test_history[1] > args.test_acc_limit:
                print('\nTrain over with Epoch: {}   Test acc: {}\n'.format(step, test_history[1]))
                break
    # model.save('model/'+args.recoder+'.h5')
    # keras_model_to_frozen_graph(model_h5_name='model/'+args.recoder+'.h5',
    #                             model_pd_name=args.recoder+'.pd')
    print('\n\n-------------------------------------------')
    print('在 {} epochs之后, Test acc: {}.\n模型固化到： "{}"\n\n'.format(step,  test_history[1], 'model/'+args.recoder+'.pd'))
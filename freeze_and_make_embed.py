# -*- coding:utf-8 -*-
import os, argparse
import tensorflow as tf
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
from tensorflow.python.framework import graph_util
import numpy as np
dir = os.path.dirname(os.path.realpath(__file__))


def freeze_graph(model_folder):
    # We retrieve our checkpoint fullpath
    checkpoint = tf.train.get_checkpoint_state(model_folder)
    input_checkpoint = checkpoint.model_checkpoint_path

    # We precise the file fullname of our freezed graph
    absolute_model_folder = "/".join(input_checkpoint.split('/')[:-1])
    output_graph = absolute_model_folder + "/frozen_model.pb"

    # Before exporting our graph, we need to precise what is our output node
    # this variables is plural, because you can have multiple output nodes
    # freeze之前必须明确哪个是输出结点,也就是我们要得到推论结果的结点
    # 输出结点可以看我们模型的定义
    # 只有定义了输出结点,freeze才会把得到输出结点所必要的结点都保存下来,或者哪些结点可以丢弃
    # 所以,output_node_names必须根据不同的网络进行修改
    output_node_names = "siamese/out_put"

    # We clear the devices, to allow TensorFlow to control on the loading where it wants operations to be calculated
    clear_devices = True

    # We import the meta graph and retrive a Saver
    saver = tf.train.import_meta_graph(input_checkpoint + '.meta', clear_devices=clear_devices)

    # We retrieve the protobuf graph definition
    graph = tf.get_default_graph()
    input_graph_def = graph.as_graph_def()

    # We start a session and restore the graph weights
    # 这边已经将训练好的参数加载进来,也即最后保存的模型是有图,并且图里面已经有参数了,所以才叫做是frozen
    # 相当于将参数已经固化在了图当中
    with tf.Session() as sess:
        saver.restore(sess, input_checkpoint)

        # We use a built-in TF helper to export variables to constant
        output_graph_def = graph_util.convert_variables_to_constants(
            sess,
            input_graph_def,
            output_node_names.split(",")  # We split on comma for convenience
        )

        # Finally we serialize and dump the output graph to the filesystem
        with tf.gfile.GFile(output_graph, "wb") as f:
            f.write(output_graph_def.SerializeToString())
    print("\n---------------------------------------------------")
    print("%d ops in the final graph." % len(output_graph_def.node))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_folder", type=str, default='model/', help="Model folder to export")
    args = parser.parse_args()

    freeze_graph(args.model_folder)

    from alg.toolkit import load_graph
    embedding_graph = load_graph('model/frozen_model.pb')

    # get input and output
    img_placeholder = embedding_graph.get_tensor_by_name(embedding_graph.get_operations()[0].values()[0].name)
    embed = embedding_graph.get_tensor_by_name(embedding_graph.get_operations()[-1].values()[-1].name)

    from alg.toolkit import read_and_decode

    Train_image, Train_labels, Train_iterator = read_and_decode(['tf_recoders/Train_data.tfrecords'], batch_size=4000,
                                                                epochs=1)

    with tf.Session() as sess:
        sess.run(Train_iterator.initializer)
        _batch_x1, _batch_y1 = sess.run([Train_image, Train_labels])

    with tf.Session(graph=embedding_graph) as sess:
        emb = sess.run(embed, feed_dict={
            img_placeholder: _batch_x1
        })

    np.save('model/emb_data.npy', {'emb_data': emb, 'labels': _batch_y1}, allow_pickle=True)
    print("\n---------------------------------------------------")
    print("embeding data 生成完毕。 存储于:  'model/emb_data.npy'")

    from alg.embedding_map import image_2_Value
    class_num = 4
    Img_2_V = image_2_Value()
    emb_means = []

    for cla in range(class_num):
        Train_image, Train_labels, Train_iterator = read_and_decode(['tf_recoders/' + str(cla) + '.tfrecords'],
                                                                    batch_size=1, epochs=1)

        imgs = []
        with tf.Session(config=config) as sess:
            sess.run(Train_iterator.initializer)
            while True:
                try:
                    _batch_x1, _batch_y1 = sess.run([Train_image, Train_labels])
                    imgs.append(np.squeeze(_batch_x1))
                except (tf.errors.OutOfRangeError, AttributeError):
                    imgs = np.array(imgs)
                    break
        emb = []
        i = 0
        while True:
            if i + 500 < imgs.shape[0]:

                with tf.compat.v1.Session(graph=embedding_graph, config=config) as sess:
                    em = sess.run(embed, feed_dict={
                        img_placeholder: imgs[i:i + 500]
                    })
                emb.append(np.squeeze( em))
                i = i + 500
            else:
                emb = np.array(emb).reshape(-1, 2)
                with tf.compat.v1.Session(graph=embedding_graph, config=config) as sess:
                    em = sess.run(embed, feed_dict={
                        img_placeholder: imgs[i:]
                    })
                emb = np.vstack((emb,  em))
                emb_means.append(np.mean(emb, axis=0))
                break

    np.save('model/close_to_obj_emb.npy', np.array(emb_means))
    print(np.array(emb_means))
    print("\n---------------------------------------------------")
    print("'close_to_obj' 平均嵌入向量 生成完毕。 存储于:  'model/close_to_obj_emb.npy'")
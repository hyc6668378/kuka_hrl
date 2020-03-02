#coding=utf-8

import numpy as np
from alg.toolkit import load_graph
import tensorflow as tf
import os
from PIL import Image

config=tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True))

class image_2_Value:
    def __init__(self):
        # self.img_mean = np.load('model/img_mean.npy')
        # self.img_std = np.load('model/img_std.npy')
        self.ave_emb = np.load('model/close_to_obj_emb.npy')

        self._graph = load_graph()
        # get input and output
        self.img_placeholder = self._graph.get_tensor_by_name(self._graph.get_operations()[0].values()[0].name)
        self.embed = self._graph.get_tensor_by_name(self._graph.get_operations()[-1].values()[-1].name)

    def __call__(self, imgs, **kwargs):
        # 图像预处理
        if imgs.ndim < 4: imgs = imgs[np.newaxis, ]

        # imgs = (imgs - self.img_mean) / self.img_std
        imgs = imgs.astype(np.float32)/ 255.0

        with tf.compat.v1.Session(graph=self._graph, config=config) as sess:
            emb = sess.run(self.embed, feed_dict={
                self.img_placeholder: imgs
            })
        return emb

class classifier:
    def __init__(self, shape, model_name='None', ):
        frozen_graph_filename = 'model/' + model_name + '.pd'
        self.shape = shape
        if not os.path.isfile(frozen_graph_filename):
            print('----------------------------------------------')
            print("'{}' can't find!\n \n".format(frozen_graph_filename))
            model_h5_name = 'model/'+ model_name +'.h5'
            model_pd_name = model_name+'.pd'
            print('\n----------------------------------------------')
            print("Try to converge from '{}'!\n \n".format(model_h5_name))
            from alg.toolkit import keras_model_to_frozen_graph

            keras_model_to_frozen_graph(model_h5_name=model_h5_name,
                                    model_pd_name=model_pd_name)

        self._frozen_graph = load_graph(frozen_graph_filename=frozen_graph_filename, graph_name="model_name")

        self.img_placeholder = self._frozen_graph.get_tensor_by_name(self._frozen_graph.get_operations()[0].values()[0].name)
        self.softmax_layer = self._frozen_graph.get_tensor_by_name(self._frozen_graph.get_operations()[-1].values()[-1].name)

    def _check_shape(self, imgs):
        if imgs.shape == self.shape:
            return imgs
        else:
            assert imgs.ndim == 3, '占时不支持 预测一批图像. imgs必须为单张图像。'
            imgs = Image.fromarray(imgs)
            imgs = imgs.resize(self.shape[:-1])
            return np.array(imgs)

    def __call__(self, imgs, **kwargs):

        imgs = self._check_shape(imgs)
        imgs = imgs[np.newaxis, ]
        if np.max(imgs)<1.:
            print('The image has been normalized and will not be normalized within the self.__call__():')
        else:
            # print('Normalize image in the self.__call__():')
            imgs = imgs.astype(np.float32)/ 255.0

        with tf.compat.v1.Session(graph=self._frozen_graph, config=config) as sess:
            prop = sess.run(self.softmax_layer, feed_dict={
                self.img_placeholder: imgs
            })
        return np.argmax(prop, axis=-1)[0] # choice class

class grasp_success_or_not(classifier):
    def __init__(self):
        super(grasp_success_or_not, self).__init__(shape=(32, 32, 3), model_name='grasp_success_or_not')

    def __call__(self, imgs, **kwargs):
        """
        返回 bool
        """
        return bool( super(grasp_success_or_not, self).__call__(imgs) )

class whether_can_grasp_or_not(classifier):
    def __init__(self):
        super(whether_can_grasp_or_not, self).__init__(shape=(32, 32, 3),model_name='whether_can_grasp_or_not')

    def __call__(self, imgs, **kwargs):
        """
        返回 bool
        """
        return bool( super(whether_can_grasp_or_not, self).__call__(imgs)-1 ) # TODO: 制作数据集时候 T F 搞反了.

class _close_to_obj(classifier):
    def __init__(self):
        super(_close_to_obj, self).__init__(shape=(128, 128, 3),model_name='_close_to_obj')


class phase_policy:
    def __init__(self, shape, model_name='None', ):
        frozen_graph_filename = 'model/' + model_name + '.pd'
        self.shape = shape

        assert os.path.isfile(frozen_graph_filename), "Can't find {}".format(frozen_graph_filename)

        self._frozen_graph = load_graph(frozen_graph_filename=frozen_graph_filename, graph_name=model_name)

        self.img_placeholder = self._frozen_graph.get_tensor_by_name(self._frozen_graph.get_operations()[0].values()[0].name)
        self.low_dim_ph = self._frozen_graph.get_tensor_by_name(self._frozen_graph.get_operations()[1].values()[0].name)
        self.pi = self._frozen_graph.get_tensor_by_name(self._frozen_graph.get_operations()[-1].values()[-1].name)

    def _check_shape(self, imgs):
        if imgs.shape == self.shape:
            return imgs
        else:
            assert imgs.ndim == 3, '占时不支持 预测一批图像. imgs必须为单张图像。'
            imgs = Image.fromarray(imgs)
            imgs = imgs.resize(self.shape[:-1])
            return np.array(imgs)

    def __call__(self, o, **kwargs):

        imgs = self._check_shape(o[0])

        with tf.compat.v1.Session(graph=self._frozen_graph, config=config) as sess:
            pi = sess.run(self.pi, feed_dict={
                self.img_placeholder: imgs[np.newaxis, ],
                self.low_dim_ph: o[1][np.newaxis, ]
            })
        return pi[0]


class _phase_2_policy(phase_policy):
    def __init__(self):
        super(_phase_2_policy, self).__init__(shape=(128, 128, 3), model_name='phase_2_policy')


class _phase_1_policy(phase_policy):
    def __init__(self):
        super(_phase_1_policy, self).__init__(shape=(128, 128, 3),model_name='phase_1_policy')

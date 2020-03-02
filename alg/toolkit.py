#coding=utf-8
import os
from os import listdir
from os.path import join
import tensorflow as tf
from PIL import Image
import matplotlib.pyplot as plt
import random
from tqdm import tqdm
import numpy as np

def normal(img):
    return img / 255.

def denormal(img):
    return img * 255.

def test_number():
    test_data_size = 0
    for root,dirs,files in os.walk('test_image/'):
        for _ in files:
            test_data_size += 1
    return test_data_size

def normalize(x, stats):
    if stats is None:
        return x
    return (x - stats.mean) / stats.std

def mkdir( rank, experiment_name):
    """传进来一个实验名， 生成 ./logs ./result ./model 生成对应文件夹"""
    folder = os.path.exists("logs/"+experiment_name )

    if not folder:
        os.makedirs("logs/" + experiment_name)
    if not os.path.exists("logs/"+experiment_name + "/DDPG_" + str(rank)):
        os.makedirs("logs/" + experiment_name + "/DDPG_" + str(rank))

    folder = os.path.exists("result/" + experiment_name)
    if not folder:
        os.makedirs("result/" + experiment_name)

    folder = os.path.exists("model/" + experiment_name)
    if not folder:
        os.makedirs("model/" + experiment_name)

def image(batch_size, epochs=2, shuffle=True, path='none'):

    # check 传进来的是一个 还是 一个list 的文件夹s
    if path == 'none':
        raise ValueError('Wrong Dataset Path!')
    elif type(path) == list:
        filenames=[]
        for Path in path:
            filenames.extend( [join(Path,f) for f in listdir(Path)] )
    else:
        filenames = [join(path,f) for f in listdir(path)]

    filename_queue = tf.train.string_input_producer(filenames, shuffle=shuffle, num_epochs=epochs)
    reader = tf.WholeFileReader()
    _, img_bytes = reader.read(filename_queue)
    images = tf.image.decode_jpeg(img_bytes, channels=3)

    return tf.train.batch([images], batch_size, dynamic_pad=True, num_threads=3, capacity=256)

# 制作TFRecords数据
def create_record(record_path="close_to_obj.tfrecords", name=0):

    writer = tf.python_io.TFRecordWriter(record_path)
    num_tf_example = 0

    class_path = "image/" + str(name) + "/"
    for img_name in tqdm(os.listdir(class_path)):
        img_path = class_path + img_name
        img = Image.open(img_path)
        img_raw = img.tobytes()  # 将图片转化为原生bytes
        example = tf.train.Example(
            features=tf.train.Features(feature={
                "label": tf.train.Feature(int64_list=tf.train.Int64List(value=[name])),
                'img_raw': tf.train.Feature(bytes_list=tf.train.BytesList(value=[img_raw]))
            }))
        writer.write(example.SerializeToString())
        num_tf_example += 1
    writer.close()
    print("\n---------------------------------------------------")
    print("{} tf_examples has been created successfully, which are saved in {}".format(num_tf_example, record_path))

def create_test_record(record_path='tf_recoders/test.tfrecords', test_class_num=8):
    # file_name list
    img_path = []
    for class_name in range(test_class_num):
        class_path = "test_image/" + str(class_name) + "/"
        for img_name in os.listdir(class_path):
            img_path.append(class_path + img_name)
    random.shuffle(img_path)

    writer = tf.python_io.TFRecordWriter(record_path)
    num_tf_example = 0

    for path in tqdm(img_path):
        img = Image.open(path)
        img_raw = img.tobytes()
        example = tf.train.Example(
            features=tf.train.Features(feature={
                "label": tf.train.Feature(int64_list=tf.train.Int64List(value=[int(path.split('/')[1])])), # path[11] -> class_label
                'img_raw': tf.train.Feature(bytes_list=tf.train.BytesList(value=[img_raw]))
            }))
        writer.write(example.SerializeToString())
        num_tf_example += 1
    writer.close()
    print("\n---------------------------------------------------")
    print("{} tf_examples has been created successfully, which are saved in {}".format(num_tf_example, record_path))

def _create_grasp_record(record_path, img_lab_pair, size=(32, 32)):
    writer = tf.python_io.TFRecordWriter(record_path)
    num_tf_example = 0

    for path, lab in tqdm(img_lab_pair):
        img = Image.open(path)

        if img.size != size:
            img = img.resize( size )
        assert img.size == size, 'img.size: {} != {}'.format(img.size, size)

        img_raw = img.tobytes()
        example = tf.train.Example(
            features=tf.train.Features(feature={
                "label": tf.train.Feature(int64_list=tf.train.Int64List(value=[int(lab)])),
                'img_raw': tf.train.Feature(bytes_list=tf.train.BytesList(value=[img_raw]))
            }))
        writer.write(example.SerializeToString())
        num_tf_example += 1
    writer.close()
    print("\n---------------------------------------------------")
    print("{} tf_examples has been created successfully, which are saved in {}\n \n".format(num_tf_example, record_path))

def create_grasp_records():
    """
    这个函数写的很low,大家忍受一下,能用就好.
    总之要干的事情就是 制作  1，抓到东西没抓到东西 的二分类 训练集和测试集合
                         2, 没抓到但抓一下可以抓到，和抓了也抓不到 二分类 训练集和测试集合
                         标签都是 0，失败  1，成功
                         测试集合大小 2048
                         'tf_recoders/grasp_success_or_not_train.tfrecords'
                         'tf_recoders/grasp_success_or_not_test.tfrecords'

                         'tf_recoders/whether_can_grasp_or_not_train.tfrecords'
                         'tf_recoders/grasp_success_or_not_test.tfrecords'
    """

    img_lab_pair = []

    """Success"""
    class_path = 'image/success_grasp_freature/'
    for img_name in os.listdir( class_path ):
        img_lab_pair.append(((class_path + img_name), 1))
    success_imgs = len(img_lab_pair)
    print('抓取中的图像：{}张'.format( success_imgs ))

    """Fail"""
    class_path = 'image/can_success_grasp/'
    for img_name in os.listdir( class_path ):
        img_lab_pair.append(((class_path + img_name), 0))

    class_path = 'image/fail_grasp_freature/'
    for img_name in os.listdir( class_path ):
        img_lab_pair.append(((class_path + img_name), 0))
    print('未抓取的图像：{}张'.format(len(img_lab_pair) - success_imgs))

    random.shuffle(img_lab_pair)
    print('\n训练集列表打乱完毕.')

    record_path = 'tf_recoders/grasp_success_or_not'
    test_data_size = int( len(img_lab_pair)/5 )
    _create_grasp_record(record_path+ '_train.tfrecords', img_lab_pair[test_data_size:])
    _create_grasp_record(record_path+ '_test.tfrecords', img_lab_pair[:test_data_size])
    img_lab_pair = []

    """can_success_grasp"""
    class_path = 'image/can_success_grasp/'
    for img_name in os.listdir( class_path ):
        img_lab_pair.append(((class_path + img_name), 1))
    success_imgs = len(img_lab_pair)
    print('可抓取状态的图像：{}张'.format( success_imgs ))

    """can not success_grasp"""
    class_path = 'image/fail_grasp_freature/'
    for index, img_name in enumerate( os.listdir( class_path )):
        img_lab_pair.append(((class_path + img_name), 0))
        if index >= 4095: break  # fail 的图像太多了，挑4096张
    print('不可抓取状态的图像：{}张'.format(len(img_lab_pair) - success_imgs))

    random.shuffle(img_lab_pair)
    print('\n训练集列表打乱完毕.')

    record_path = 'tf_recoders/whether_can_grasp_or_not'
    test_data_size = int( len(img_lab_pair)/5 ) # 5分之一做测试集
    _create_grasp_record(record_path + '_train.tfrecords', img_lab_pair[test_data_size:])
    _create_grasp_record(record_path + '_test.tfrecords', img_lab_pair[:test_data_size])

def create_close_to_obj_records():
    img_lab_pair = []

    for i in range(4):
        class_path = 'image/' + str(i) + '/'
        for img_name in os.listdir( class_path ):
            img_lab_pair.append(((class_path + img_name), i))
        print('类别{}：{}张'.format(i, len(os.listdir( class_path )) ))

    record_path = 'tf_recoders/_close_to_obj'
    test_data_size = int( len(img_lab_pair)/5 )
    _create_grasp_record(record_path+ '_train.tfrecords', img_lab_pair[test_data_size:], size=(128, 128))
    _create_grasp_record(record_path+ '_test.tfrecords', img_lab_pair[:test_data_size], size=(128, 128))
# =======================================================================================
def read_and_decode(filenames, batch_size=32, epochs=2, img_size='big', dataset_as_output=False):
    # img_mean = np.load('model/img_mean.npy')
    # img_std = np.load('model/img_std.npy')

    def _128_parse_function(example_proto):
        features = {"img_raw": tf.io.FixedLenFeature((), tf.string, default_value=""),
                    "label": tf.io.FixedLenFeature((), tf.int64, default_value=0)}
        features = tf.io.parse_single_example(example_proto, features)
        img = features['img_raw']
        img = tf.decode_raw(img, tf.uint8)
        img = tf.reshape(img, [128, 128, 3])
        img = tf.cast(img, tf.float32)/ 255.0
        label = tf.cast(features["label"], tf.int32)
        return img, label

    def _32_parse_function(example_proto):
        features = {"img_raw": tf.io.FixedLenFeature((), tf.string, default_value=""),
                    "label": tf.io.FixedLenFeature((), tf.int64, default_value=0)}
        features = tf.io.parse_single_example(example_proto, features)
        img = features['img_raw']
        img = tf.decode_raw(img, tf.uint8)
        img = tf.reshape(img, [32, 32, 3])
        img = tf.cast(img, tf.float32)/ 255.0
        label = tf.cast(features["label"], tf.int32)
        return img, label
    # filenames = [filename]
    dataset = tf.data.TFRecordDataset(filenames)

    _parse_function = _128_parse_function if img_size=='big' else _32_parse_function

    dataset = dataset.map(_parse_function)
    dataset = dataset.shuffle(buffer_size=5000)
    dataset = dataset.batch( batch_size )
    if dataset_as_output: return dataset
    print("TFRecord has been loaded successfully, which are saved in {}".format(filenames))


    dataset = dataset.prefetch(buffer_size=10 * batch_size)
    dataset = dataset.repeat( epochs )
    iterator = dataset.make_initializable_iterator()
    images, labels = iterator.get_next()

    return images, labels, iterator

# =======================================================================================
def show_embed(embed, label, step, class_num):
    plt.figure(figsize=(7, 5))
    for ite in range(class_num):
        m = '^' if ite>5 else 'o'
        plt.scatter(embed[label == ite][:, 0], embed[label == ite][:, 1], marker=m, alpha=0.2, label='step_'+str(ite))
        # embed [label == ite][:, 1]
        # ite*np.ones_like(embed [label == ite][:, 0]),
    plt.title('scatter plot ')
    plt.xlabel('variables x')
    plt.ylabel('variables y')
    plt.legend()
    plt.savefig('result/'+str(step)+'.png')
    plt.close()
# =======================================================================================

def load_graph(frozen_graph_filename='model/frozen_model.pb', graph_name="Embedding"):
    # We parse the graph_def file
    with tf.io.gfile.GFile(frozen_graph_filename, "rb") as f:
        graph_def = tf.compat.v1.GraphDef()
        graph_def.ParseFromString(f.read())
    # We load the graph_def in the default graph
    with tf.Graph().as_default() as graph:
        tf.import_graph_def(
            graph_def,
            input_map=None,
            return_elements=None,
            name=graph_name,
            op_dict=None,
            producer_op_list=None
        )
    return graph



def ppo_to_frozen_graph(exp_name='ppo_kuka_phase_2', model_pd_name='phase_2_policy.pd'):
    from env.KukaGymEnv import KukaDiverseObjectEnv
    from ppo import ppo

    test_env_fn = lambda : KukaDiverseObjectEnv(renders=True,
                         maxSteps=100,
                         blockRandom=0.25,
                         actionRepeat=300,
                         numObjects=1, dv=1.0,
                         isTest=False, phase = 2)
    model = ppo(test_env_fn, seed=0, steps_per_epoch=128,
            epochs=100, exp_name=exp_name, gamma=0.9,
           test_agent=False)
    model.load()
    with model.graph.as_default():
        graphdef_inf = tf.graph_util.remove_training_nodes(model.graph.as_graph_def())
        graphdef_frozen = tf.graph_util.convert_variables_to_constants(model.sess, graphdef_inf, ['pi/pi_out'])
        tf.io.write_graph(graphdef_frozen, "model", model_pd_name, as_text=False)
        print("Frozen_graph in 'model/"+model_pd_name)


def keras_model_to_frozen_graph(model_h5_name, model_pd_name):
    """ convert keras h5 model file to frozen graph(.pb file)

    model_h5_name = 'model/grasp_success_or_not.h5'
    model_pd_name= 'grasp_success_or_not.pd'   #  注意没有 'model/'
    keras_model_to_frozen_graph( model_h5_name= model_h5_name,
                                 model_pd_name= model_pd_name)
    """

    def freeze_graph(graph, session, output_node_names, model_pd_name):
        with graph.as_default():
            graphdef_inf = tf.graph_util.remove_training_nodes(graph.as_graph_def())
            graphdef_frozen = tf.graph_util.convert_variables_to_constants(session, graphdef_inf, output_node_names)
            tf.io.write_graph(graphdef_frozen, "model", model_pd_name, as_text=False)

    tf.keras.backend.set_learning_phase(False)  # this line most important

    assert os.path.isfile(model_h5_name),"Can't find {}".format(model_h5_name)

    model = tf.keras.models.load_model(model_h5_name)
    session = tf.keras.backend.get_session()
    freeze_graph(session.graph, session, [out.op.name for out in model.outputs], model_pd_name)
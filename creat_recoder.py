#coding=utf-8

# 收集完图像之后制作 tfrecoder


from alg.toolkit import create_grasp_records, create_close_to_obj_records
import os

_ = os.system("clear")

# class_num = len( os.listdir('image/'))
# class_num = 4
# create_test_record(record_path='tf_recoders/Train_data.tfrecords', test_class_num=class_num)
create_grasp_records()
create_close_to_obj_records()

"""
重新统计 训练集 均值 方差。
保存到 model/img_mean.npy'  'model/img_std.npy'
"""

"""
config = tf.ConfigProto()
config.gpu_options.allow_growth = True

obs_rms = RunningMeanStd(shape=(128, 128, 3))

img = image(1, epochs=1, path=['image/'+str(clas)+'/' for clas in range( class_num )])

with tf.Session(config=config) as sess:
    sess.run([tf.global_variables_initializer(), tf.local_variables_initializer()])
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord)
    step = 0
    try:
        print('\n开始统计:\n')
        while not coord.should_stop():
            batch_imgs = sess.run(img)
            obs_rms.update( batch_imgs )
            step +=1

    except tf.errors.OutOfRangeError:
        print('\n统计完毕! -- epoch limit reached\n')
    finally:
        coord.request_stop()
    coord.join(threads)
    print('\ntotal steps: ',step)
    img_mean = sess.run(obs_rms.mean)
    img_std = sess.run(obs_rms.std)

np.save('model/img_mean.npy', img_mean )
np.save('model/img_std.npy', img_std )

print("\n---------------------------------------------------")
print("新均值和方差已经存储于: 'model/img_mean.npy' 'model/img_std.npy'\n")
"""
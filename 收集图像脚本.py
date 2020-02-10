# coding=utf-8

from env.KukaGymEnv import KukaDiverseObjectEnv
import argparse
import numpy as np
from multiprocessing import Pool, Manager
import os
from PIL import Image
from tqdm import tqdm


def common_arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("-t", "--train", action="store_true", help="收集训练数据还是测试数据")
    parser.add_argument('--worker_step',    type=int, default=2019, help="每个worker循环次数.")
    return  parser


manager = Manager()
class_itm_dict = manager.dict()

parser = common_arg_parser()
args = parser.parse_args()


train_ = args.train

root_path = 'image/' if train_ else 'test_image/'
class_number = len(os.listdir( root_path)) - 3  # 物体特写3个目录 不算总类别


def collect_worker(worker_index):
    print (str(worker_index)+" start!")
    # worker_i = worker_index * 5000
    env = KukaDiverseObjectEnv(renders=False,
                               isDiscrete=False,
                               maxSteps=20,
                               blockRandom=0.3,
                               use_segmentation_Mask=False,
                               actionRepeat=200,
                               numObjects=1, dv=1.0,
                               isTest=not train_)
    for _ in tqdm(range(args.worker_step)):
        imgs=[]
        o = env.reset()
        imgs.extend(env._close_to_obj())
        before = env._obj_high()
        imgs.append( env._grasp() )
        imgs.append( env._pick_up(pick_up_iter = 100) )
        imgs.append( env._pick_up(pick_up_iter = 100) )
        grasp_correct = True if (env._obj_high() - before) > 0.03 else False
        imgs.extend( env._skill_move_to_box_above() )
        imgs.append(env._release())

        try :
            assert len(imgs) == class_number, 'len(imgs): {}  不等于  class_number: {}'.format(len(imgs), class_number)
        except (AssertionError):
            print('len(imgs): {}  不等于  class_number: {}'.format(len(imgs), class_number))
            quit()

        if grasp_correct:
            for i, img in enumerate(imgs):
                img_path = root_path+'%d/' % i
                img_path += str(class_itm_dict[str(i)]) + '.png'
                class_itm_dict[str(i)] += 1
                im = Image.fromarray(img[0])
                im.save(img_path)

                if i in {0,1,2,10}: # 没有抓住的样子
                    img_path = root_path + 'fail_grasp_freature/'
                    img_path += str(class_itm_dict['fail_grasp_freature']) + '.png'
                    class_itm_dict['fail_grasp_freature'] += 1

                    im = Image.fromarray(img[1])
                    im.save(img_path)
                elif i in {3}:
                    img_path = root_path + 'can_success_grasp/'
                    img_path += str(class_itm_dict['can_success_grasp']) + '.png'
                    class_itm_dict['can_success_grasp'] += 1

                    im = Image.fromarray(img[1])
                    im.save(img_path)
                else:  # 成功抓住的手部特写
                    img_path = root_path + 'success_grasp_freature/'
                    img_path += str(class_itm_dict['success_grasp_freature']) + '.png'
                    class_itm_dict['success_grasp_freature'] += 1

                    im = Image.fromarray(img[1])
                    im.save(img_path)
        else:   # 存下来所有 没有抓住的手部特写
            for i, img in enumerate(imgs):
                img_path = root_path+'fail_grasp_freature/'
                img_path += str(class_itm_dict['fail_grasp_freature']) + '.png'
                class_itm_dict['fail_grasp_freature'] += 1
                im = Image.fromarray(img[1])
                im.save(img_path)
    print(str(worker_index) + ': end')


if __name__ == '__main__':
    # 开12个进程 一起收集demo
    print('Parent process %s.' % os.getpid())

    worker = 4
    # 初始化 全局计数字典
    for i in os.listdir( root_path):
        class_itm_dict[i] = len( os.listdir(root_path + i))

    p = Pool(worker)

    for k in range(worker):
        p.apply_async(collect_worker, args=(k,))
    p.close()
    p.join()

    print('All subprocesses done.')
    print('----------------------------------------')
    for key in class_itm_dict:
        print('Folder "{}" has imges: {}'.format(key, class_itm_dict[key]))



# coding=utf-8

from env.kuka import Kuka
import os
from gym import spaces
import pybullet as p
from env import kuka
import numpy as np
import pybullet_data
import glob
from pkg_resources import parse_version
import gym
from gym.utils import seeding
import random
# from alg.embedding_map import _close_to_obj, whether_can_grasp_or_not
from PIL import Image
from copy import copy


class KukaDiverseObjectEnv(Kuka, gym.Env):
    """Class for Kuka environment with diverse objects.

    In each episode some objects are chosen from a set of 1000 diverse objects.
    These 1000 objects are split 90/10 into a train and test set.
    """

    def __init__(self,
                 urdfRoot=pybullet_data.getDataPath(),
                 actionRepeat=200,
                 renders=False,
                 maxSteps=8,
                 dv=0.06,
                 blockRandom=0.2,
                 cameraRandom=0,
                 width=128,
                 height=128,
                 numObjects=1,
                 isTest=False):

        # Environment Characters
        self._timeStep,     self._urdfRoot     =   1. / 240.     , urdfRoot
        self._actionRepeat, self._isTest       =   actionRepeat  , isTest
        self._renders,      self._maxSteps     =   renders       , maxSteps
        self.terminated,    self._dv           =   0             , dv
        self._blockRandom,  self._cameraRandom =   blockRandom   , cameraRandom
        self.finger_angle_max = 0.25
        self._width,        self._height       =   width         , height
        self._success,      self._numObjects   =   False         , numObjects

        # self._can_grasp_or_not = whether_can_grasp_or_not() # 抓一下能否抓到物体
        # self._close_to_obj     = _close_to_obj()  # 到技巧'_close_to_obj'的第几阶段了？
        self.observation_space = spaces.Box(low=0, high=255, shape=(self._width, self._height, 3), dtype=np.uint32)

        if self._renders:
            self.cid = p.connect(p.GUI,
                                 options = "--window_backend=2 --render_device=0")
            p.resetDebugVisualizerCamera(1.3, 180, -41, [0.52, -0.2, -0.33])
            # p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
        else:
            self.cid = p.connect(p.GUI,
                                 options = "--width={} --height={} --window_backend=2 --render_device=1".format(self._width, self._height))
            p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 0)
            p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
            p.configureDebugVisualizer(p.COV_ENABLE_WIREFRAME, 0)
            p.configureDebugVisualizer(p.COV_ENABLE_DEPTH_BUFFER_PREVIEW, 0)

        self.seed()
        self.action_space = spaces.Box(low=-1, high=1, shape=(5,))  # dx, dy, dz, da

    def _reset(self):
        look = [1.9, 0.5, 1]
        roll = -10
        pitch = -35
        yaw = 110

        look_2 = [-0.3, 0.5, 1.3]
        pitch_2 = -56
        yaw_2 = 245
        roll_2 = 0
        distance = 1.

        self._view_matrix_2 = p.computeViewMatrixFromYawPitchRoll(
            look_2, distance, yaw_2, pitch_2, roll_2, 2)

        self._view_matrix = p.computeViewMatrixFromYawPitchRoll(
            look, distance, yaw, pitch, roll, 2)

        fov = 20. + self._cameraRandom * np.random.uniform(-2, 2)
        aspect = self._width / self._height
        near = 0.01
        far = 10
        self._proj_matrix = p.computeProjectionMatrixFOV(
            fov, aspect, near, far)
        self.goal_rotation_angle = 0
        self._attempted_grasp = False
        self._env_step = 0
        self.terminated = 0
        self.finger_angle = 0.3
        self._success = False
        self._before = 0
        self.x, self.y = 0, 0
        self.out_of_range = False

        if self._isTest: _ = os.system("rm result/_*")

        p.resetSimulation()
        p.setPhysicsEngineParameter(numSolverIterations=300)
        p.setTimeStep(self._timeStep)

        p.loadURDF(os.path.join(self._urdfRoot, "plane.urdf"), [0, 0, -1])
        p.loadURDF(os.path.join(self._urdfRoot, "table/table.urdf"),
           [0.5000000, 0.00000, -.820000], p.getQuaternionFromEuler( np.radians([0,0,90.])))

        p.setGravity(0, 0, -10)
        self._kuka = kuka.Kuka(urdfRootPath=self._urdfRoot, timeStep=self._timeStep)
        self._envStepCounter = 0
        p.stepSimulation()

        # Choose the objects in the bin.
        urdfList = self._get_random_object(
            self._numObjects, self._isTest)
        self._objectUids = self._randomly_place_objects(urdfList)

        obs = self._get_observation()

        return obs

    def _low_dim_full_state(self):
        full_state = []

        for uid in self._objectUids:
            pos, ori = p.getBasePositionAndOrientation(uid)
            full_state.extend(pos)
        full_state.extend( self._kuka.getObservation())
        full_state = np.array(full_state).flatten()
        # 'obj_1.x', 'obj_1.y', 'obj_1.z', ... ,
        # 'gripper.x', 'gripper.y', 'gripper.z', 'gripper.r', 'gripper.p', 'gripper.y'
        return full_state

    def _obj_high(self):
        assert len(self._objectUids) == 1
        # obj.z
        return p.getBasePositionAndOrientation(self._objectUids[0])[0][2]

    def _close_to_obj_atomic_action(self, i=0):

        fs = self._low_dim_full_state()
        # 随便选一个物体
        current_End_EffectorPos = np.array( p.getLinkState(self._kuka.kukaUid,
                                                           self._kuka.kukaEndEffectorIndex)[0] )
        obj = fs[:3]
        high_offset = np.array([0.0, 0.0, 0.35], dtype=np.float32)# 比物体稍稍高一点点
        obj_offset = np.array([0.0, 0.02, 0.0], dtype=np.float32)# 物体的坐标和真实稍稍错位一点， 一点点调出来的。

        # move to up of object
        dis = obj - current_End_EffectorPos + high_offset + obj_offset

        if abs(dis[0])<5e-3 and abs(dis[1]) <5e-3:
            # 距离物体中心足够近, 垂直向下
            if i==0 or i==6 or i==15:
                return np.array([0.0, 0.0, -0.2, np.random.randn(), np.random.randn()], dtype=np.float32)
            else:
                return np.array([0.0, 0.0, -0.2, np.random.randn(), 0.1], dtype=np.float32)

        action = dis * 6
        action = np.append(action, np.random.randn()) # todo:爪子旋转占时随机
        action = np.append(action, np.random.randn()) # 随机爪子
        action = np.clip(action, self.action_space.low, self.action_space.high)

        return action

    def _skill_close_to_obj(self):
        img=[]
        self.finger_angle = self.finger_angle_max
        for i in range(25):
            atomic_a = self._close_to_obj_atomic_action(i)
            self._atomic_action(atomic_a, repeat_action=300)
            img.append(self._get_observation())
            r=self._reward()
        return [img[0], img[6], img[15], img[24]]

    def _Current_End_EffectorPos(self):
        state = p.getLinkState(self._kuka.kukaUid, self._kuka.kukaEndEffectorIndex)
        current_End_EffectorPos = np.array(state[0])
        return current_End_EffectorPos

    def _pick_up(self, pick_up_iter=300):
        # pick up
        current_End_EffectorPos = self._Current_End_EffectorPos()

        for _ in range(pick_up_iter):
            current_End_EffectorPos[2] = current_End_EffectorPos[2] + 0.001
            current_End_EffectorPos = np.clip(current_End_EffectorPos, a_min=np.array([0.2958, -0.4499, 0.0848]),
                                                                       a_max=np.array([0.70640, 0.3872, 0.56562]))

            self._kuka.applyAction(current_End_EffectorPos, da=self.goal_rotation_angle, fingerAngle=self.finger_angle)

            p.stepSimulation()

            self.finger_angle -= 0.3 / 100.
            self.finger_angle = np.clip(self.finger_angle, 0., 0.35)

        return self._get_observation()

    def _grasp(self):
        current_End_EffectorPos = self._Current_End_EffectorPos()
        # grasp
        for _ in range(500):
            self._kuka.applyAction( current_End_EffectorPos, self.goal_rotation_angle, fingerAngle=self.finger_angle)

            p.stepSimulation()

            self.finger_angle -= 0.3 / 100.
            self.finger_angle = np.clip(self.finger_angle, 0., 0.35)

        return self._get_observation()

    def _release(self):
        current_End_EffectorPos = self._Current_End_EffectorPos() + np.array([0., 0., 0.02]) # z轴高2cm 抵消重力。

        # release
        for _ in range(500):
            self._kuka.applyAction( current_End_EffectorPos, self.goal_rotation_angle, fingerAngle=self.finger_angle)

            p.stepSimulation()

            self.finger_angle += 0.15 / 100.
            self.finger_angle = np.clip(self.finger_angle, 0., 0.35)

        return self._get_observation()

    def _move_to_box_above_and_resease_atomic_action(self):
        box_pos = np.array( p.getBasePositionAndOrientation(self._kuka.trayUid)[0] )
        current_End_EffectorPos = self._Current_End_EffectorPos()
        dis = box_pos - current_End_EffectorPos

        action = dis * 4
        action = np.append(action, 0.0)  # todo:爪子不旋转
        action = np.append(action, -0.1)  # 闭合爪子
        action = np.clip(action, self.action_space.low, self.action_space.high)
        # 爪子最低高度 0.13 m
        if current_End_EffectorPos[2] < 0.13:
            action[2] = 1.
        else:
            action[2] = 0.

        return action

    def _skill_move_to_box_above(self):
        img = []
        for _ in range(18):
            atomic_a = self._move_to_box_above_and_resease_atomic_action()
            self._atomic_action(atomic_a, repeat_action=100)
            img.append(self._get_observation())
        return [img[4], img[9], img[17]]

    def _atomic_action(self, action, repeat_action=200):
        # 执行原子action

        # descale + gravity offset(z axis)
        act_descale = np.array([0.05, 0.05, 0.05, np.radians(90), 1.])
        action = action * act_descale + np.array([0., 0., 0.02, 0., 0.])

        state = p.getLinkState(self._kuka.kukaUid, self._kuka.kukaEndEffectorIndex)
        current_End_EffectorPos = np.array( state[0] )

        goal_pose = np.clip( current_End_EffectorPos + action[:3], a_min=np.array([0.2958, -0.4499, 0.0848]),
                                                                    a_max=np.array([0.70640, 0.3872 , 0.56562]) )
        out_of_range = np.sum( (goal_pose != current_End_EffectorPos + action[:3])[:2] )
        self.out_of_range = True if out_of_range else False

        # if out_of_range:
        #     print("goal_pose:\t", goal_pose )
        #     print("cur_pose:\t", current_End_EffectorPos + action[:3])
        #     print("action:\t", action[:3])

        self.goal_rotation_angle += action[-2]  # angel

        # execute
        for _ in range(repeat_action):
            self._kuka.applyAction( goal_pose, self.goal_rotation_angle, fingerAngle=self.finger_angle)
            p.stepSimulation()

            if action[-1]>0:
                self.finger_angle += 0.2 / 100.
            else:
                self.finger_angle -= 0.3 / 100.
            self.finger_angle = np.clip(self.finger_angle, 0., 0.3 )

    def _randomly_place_objects(self, urdfList):
        """Randomly places the objects in the bin.

        Args:
          urdfList: The list of urdf files to place in the bin.

        Returns:
          The list of object unique ID's.
        """

        # Randomize positions of each object urdf.
        objectUids = []
        for urdf_name in urdfList:
            xpos = 0.45 + self._blockRandom * random.random()
            ypos = 0.2 + self._blockRandom * (random.random() - .5)
            angle = np.pi / 2 + self._blockRandom * np.pi * random.random()
            orn = p.getQuaternionFromEuler([0, 0, angle])
            urdf_path = os.path.join(self._urdfRoot, urdf_name)
            uid = p.loadURDF(urdf_path, [xpos, ypos, .05],
                             [orn[0], orn[1], orn[2], orn[3]])
            objectUids.append(uid)
            # Let each object fall to the tray individual, to prevent object
            # intersection.
            for _ in range(300):
                p.stepSimulation()
        return objectUids

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def _get_observation(self):
        img_arr = p.getCameraImage(width=self._width,
                                   height=self._height,
                                   viewMatrix=self._view_matrix,
                                   projectionMatrix=self._proj_matrix,
                                   shadow=True,
                                   lightDirection=[1, 1, 1],
                                   renderer=p.ER_BULLET_HARDWARE_OPENGL
                                   )
        rgb = np.reshape(img_arr[2], (self._height, self._width, 4))[:, :, :3]
        # segmentation = img_arr[4]
        #
        # x, y = np.where(segmentation == 4) # 物体的掩码是4号
        #
        # # 手把物体挡住，找不到4号掩码，会返回空 x,y. 这时候就用上一次的 x,y
        # if x != np.array([]) and y != np.array([]):
        #     self.x, self.y = int(np.mean(x)), int(np.mean(y))
        #
        # # 大小不够 就resize图像。
        # imgs = Image.fromarray(copy(rgb[self.x-16:self.x+16, self.y-16:self.y+16, :]))
        # imgs = imgs.resize( size=(32,32) )

        return rgb

    def step_skill(self, skill_num):

        self._env_step +=1

        if skill_num == 0:
            _ =self._skill_close_to_obj()
        elif skill_num == 1:
            _ =self._grasp()
        elif skill_num == 2:
            _ =self._skill_move_to_box_above()
        else:
            _ =self._release()

        obs, obj_obs = self._get_observation()
        done = self._termination()
        reward = self._up_layer_reward()
        debug = {
            'is_success': self._success
        }

        return (obs, obj_obs), reward, done, debug

    def step(self, action, _step=0):
        self._env_step += 1
        self._atomic_action( action )

        obs= self._get_observation()

        reward = self._reward( _step )

        done = self._termination() or (reward==-1)
        debug = {
            'is_success': self._success
        }

        return obs, reward, done, debug

    def collect_img(self, grasp=False):
        self._env_step += 1
        self.finger_angle = 0. if grasp else random.random() * self.finger_angle_max
        a = self.action_space.sample()
        if grasp:
            a = a*1.5
        self._atomic_action( a )

        img, obj_obs = self._get_observation()
        done = self._termination()
        return img, done

    def _up_layer_reward(self):
        return 0

    def _reward(self, _step=0):
        # 如果机器人碰到框子。 直接惩罚
        if len(p.getContactPoints(bodyA=self._kuka.trayUid,
                                  bodyB=self._kuka.kukaUid)) != 0: return -1
        # 出界 有惩罚
        if self.out_of_range:
            return -1

        fs = self._low_dim_full_state()
        current_End_EffectorPos = np.array( p.getLinkState(self._kuka.kukaUid,
                                                           self._kuka.kukaEndEffectorIndex)[0] )
        obj = fs[:3]
        obj_offset = np.array([0.0, 0.02, 0.0], dtype=np.float32)# 物体的坐标和真实稍稍错位一点， 一点点调出来的。
        gripper_offset = np.array([0.0, 0.0, 0.25], dtype=np.float32)
        dis = obj - current_End_EffectorPos + obj_offset + gripper_offset
        dis = np.sqrt(np.sum(dis**2))

        # attach obj
        if sum( [len(p.getContactPoints(bodyA=uid, bodyB=self._kuka.kukaUid)) != 0 for uid in self._objectUids]):
            r = 7
            self._success = True
        else:
            if dis>0.57: r = 0
            elif 0.37< dis <=0.57: r = 1
            elif 0.27< dis <=0.37: r = 2
            elif 0.18< dis <=0.27: r = 3
            elif 0.14< dis <=0.18: r = 4
            elif 0.09 < dis <= 0.14: r = 5
            elif 0.05 < dis <= 0.09: r = 6
            else:
                r = 7
                self._success = True

# print("rank: {}".format(r))

        reward = r-self._before
        self._before = r
        return reward

    def _termination(self):
        return (self._env_step >= self._maxSteps) or self._success or self.out_of_range

    def _get_random_object(self, num_objects, test):
        """Randomly choose an object urdf from the random_urdfs directory.

        Args:
          num_objects:
            Number of graspable objects.

        Returns:
          A list of urdf filenames.
        """
        if test:
            urdf_pattern = os.path.join(self._urdfRoot, 'random_urdfs/*0/*.urdf')
        else:
            urdf_pattern = os.path.join(self._urdfRoot, 'random_urdfs/*[^0]/*.urdf')
        found_object_directories = glob.glob(urdf_pattern)
        total_num_objects = len(found_object_directories)
        selected_objects = np.random.choice(np.arange(total_num_objects),
                                            num_objects)
        selected_objects_filenames = []
        for object_index in selected_objects:
            selected_objects_filenames += [found_object_directories[object_index]]
        return selected_objects_filenames

    if parse_version(gym.__version__) >= parse_version('0.9.6'):
        reset = _reset
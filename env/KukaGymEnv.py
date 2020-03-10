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
import pyinter

# from alg.embedding_map import _close_to_obj, whether_can_grasp_or_not
# from PIL import Image
# from copy import copy


class KukaDiverseObjectEnv(Kuka, gym.Env):

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
                 isTest=False,
                 proce_num=0,
                 phase=1, rgb_only=True,
                 single_img=False,
                 verbose=True):

        # Environment Characters
        self._timeStep,     self._urdfRoot     =   1. / 240.     , urdfRoot
        self._actionRepeat, self._isTest       =   actionRepeat  , isTest
        self._renders,      self._maxSteps     =   renders       , maxSteps
        self.terminated,    self._dv           =   0             , dv
        self._blockRandom,  self._cameraRandom =   blockRandom   , cameraRandom
        self.finger_angle_max = 0.25
        self._width,        self._height       =   width         , height
        self._success,      self._numObjects   =   False         , numObjects
        self._proce_num, self._rgb_only, self._single_img   = proce_num, rgb_only, single_img

        self.phase_2, self._verbose, = (phase != 1), verbose

        # self._can_grasp_or_not = whether_can_grasp_or_not() # 抓一下能否抓到物体
        # self._close_to_obj     = _close_to_obj()  # 到技巧'_close_to_obj'的第几阶段了？
        if self._rgb_only:
            self.observation_space = spaces.Box(low=0, high=255, shape=(self._width*2, self._height, 3), dtype=np.uint32)
            if self._single_img:
                self.observation_space = spaces.Box(low=0, high=255, shape=(self._width, self._height, 3),
                                                    dtype=np.uint32)
        else:
            self.observation_space = spaces.Box(low=0, high=255, shape=(self._width, self._height, 3), dtype=np.uint32)


        if self._renders:
            self.cid = p.connect(p.GUI,
                                 options = "--window_backend=2 --render_device=0")
            p.resetDebugVisualizerCamera(1.3, 180, -41, [0.52, -0.2, -0.33])
            p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
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
        self._env_step = 0
        self.terminated = 0
        self.finger_angle = 0.3
        self._success = False

        self.out_of_range = False
        self._collision_box = False
        self._grasp_ok = False
        self.drop_down = False
        self._rank_before = 0
        self.inverse_rank = 0
        self._phase2_bigin = False


        if self._isTest: _ = os.system("rm result/_*")

        p.resetSimulation()
        p.setPhysicsEngineParameter(numSolverIterations=300)
        p.setTimeStep(self._timeStep)

        self._planeUid = p.loadURDF(os.path.join(self._urdfRoot, "plane.urdf"), [0, 0, -1])
        self._tableUid = p.loadURDF(os.path.join(self._urdfRoot, "table/table.urdf"),
           [0.5000000, 0.00000, -.820000], p.getQuaternionFromEuler( np.radians([0,0,90.])))

        p.setGravity(0, 0, -10)
        self._kuka = kuka.Kuka(urdfRootPath=self._urdfRoot, timeStep=self._timeStep)
        self._envStepCounter = 0
        p.stepSimulation()

        # Choose the objects in the bin.
        urdfList = self._get_random_object(
            self._numObjects, test=self._isTest)
        self._objectUids = self._randomly_place_objects(urdfList)
        self._init_obj_high = self._obj_high()

        # phase 2
        if self.phase_2:
            try_init_cout = 0
            while not self._grasp_ok:
                self._skill_close_to_obj()
                self._grasp_ok = self._grasp_pick_up()
                try_init_cout +=1
                if try_init_cout > 5:
                    if self._verbose: print("process: {}\tPhase2 Init Fail, Try again".format(self._proce_num))
                    return self._reset()
            self._rank_before = self._rank_2()
        else:
            self._rank_before = self._rank_1()

        self._domain_random()
        obs = self._get_observation()
        return obs

    def _domain_random(self):
        p.changeVisualShape(self._objectUids[0], -1,
                            rgbaColor=[random.random(), random.random(), random.random(), 1.])
        p.changeVisualShape(self._planeUid, -1,
                            rgbaColor=[random.random(), random.random(), random.random(), 1.])
        p.changeVisualShape(self._tableUid, -1,
                            rgbaColor=[random.random(), random.random(), random.random(), 1.])
        for jointIndex in range(self._kuka.numJoints):
            p.changeVisualShape(self._kuka.kukaUid, linkIndex=jointIndex,
                            rgbaColor=[random.random(), random.random(), random.random(), 1.])

        p.changeVisualShape(self._kuka.trayUid, -1,
                            rgbaColor=[random.random(), random.random(), random.random(), 1.])

    def _low_dim_full_state(self):
        full_state = []

        for uid in self._objectUids:
            pos, ori = p.getBasePositionAndOrientation(uid)
            full_state.extend(pos)
        full_state.extend( self._kuka.getObservation())
        full_state = np.array(full_state).flatten()
        # 'obj_1.latent_s', 'obj_1.y', 'obj_1.z', ... ,
        # 'gripper.latent_s', 'gripper.y', 'gripper.z', 'gripper.r', 'gripper.p', 'gripper.y'
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
        # img=[]
        self.finger_angle = self.finger_angle_max
        for i in range(25):
            atomic_a = self._close_to_obj_atomic_action(i)
            self._atomic_action(atomic_a, repeat_action=300)
            # img.append(self._get_observation())
            # r=self._reward()
        # return [img[0], img[6], img[15], img[24]]

    def _Current_End_Effector_State(self):
        _pos = np.array( p.getLinkState(self._kuka.kukaUid, self._kuka.kukaEndEffectorIndex) )[0]
        _end_effector = np.append(_pos, [self.goal_rotation_angle, self.finger_angle])
        return _end_effector

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

    def _grasp_pick_up(self):
        current_End_EffectorPos = self._Current_End_EffectorPos()

        before = self._obj_high()

        # grasp
        for _ in range(500):
            self._kuka.applyAction( current_End_EffectorPos, self.goal_rotation_angle, fingerAngle=self.finger_angle)

            p.stepSimulation()

            self.finger_angle -= 0.3 / 100.
            self.finger_angle = np.clip(self.finger_angle, 0., 0.35)

        # pick up
        current_End_EffectorPos = self._Current_End_EffectorPos()
        pick_up_iter = 300
        for _ in range(pick_up_iter):
            current_End_EffectorPos[2] = current_End_EffectorPos[2] + 0.001
            current_End_EffectorPos = np.clip(current_End_EffectorPos, a_min=np.array([0.2958, -0.4499, 0.0848]),
                                              a_max=np.array([0.70640, 0.3872, 0.56562]))
            self._kuka.applyAction(current_End_EffectorPos, da=self.goal_rotation_angle, fingerAngle=self.finger_angle)
            p.stepSimulation()
            self.finger_angle -= 0.3 / 100.
            self.finger_angle = np.clip(self.finger_angle, 0., 0.35)

        grasp_correct = True if (self._obj_high() - before) > 0.03 else False
        return grasp_correct

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
        # img = []
        for _ in range(18):
            atomic_a = self._move_to_box_above_and_resease_atomic_action()
            self._atomic_action(atomic_a, repeat_action=100)
            # img.append(self._get_observation())
        # return [img[4], img[9], img[17]]

    def _atomic_action(self, action, repeat_action=200):
        # 执行原子action

        # descale + gravity offset(z axis)
        act_descale = np.array([0.05, 0.05, 0.05, np.radians(90), 1.])
        action = action * act_descale + np.array([0., 0., 0.02, 0., 0.])

        state = p.getLinkState(self._kuka.kukaUid, self._kuka.kukaEndEffectorIndex)
        current_End_EffectorPos = np.array( state[0] )

        goal_pose = np.clip( current_End_EffectorPos + action[:3], a_min=np.array([0.2758, -0.4499, 0.0848]),
                                                                    a_max=np.array([0.79640, 0.4972 , 0.56562]) )
        out_of_range = np.sum( (goal_pose != current_End_EffectorPos + action[:3])[:2] )
        self.out_of_range = True if out_of_range else False

        if self._isTest:
            if out_of_range:
                print("goal_pose:\t", goal_pose )
                print("cur_pose:\t", current_End_EffectorPos + action[:3])
                print("action:\t", action[:3])

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
            xpos = 0.35 + self._blockRandom * random.random()
            ypos = 0.26 + self._blockRandom * (random.random() - .5)
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
        _, seed = seeding.np_random(seed)
        random.seed(seed)
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
        rgb_1 = np.reshape(img_arr[2], (self._height, self._width, 4))[:, :, :3]

        End_Effector_state = self._Current_End_Effector_State()

        if self._single_img:
            if self._rgb_only:
                return rgb_1
            else:
                return [rgb_1, End_Effector_state]

        img_arr_2 = p.getCameraImage(width=self._width,
                                   height=self._height,
                                   viewMatrix=self._view_matrix_2,
                                   projectionMatrix=self._proj_matrix,
                                   shadow=True,
                                   lightDirection=[1, 1, 1],
                                   renderer=p.ER_BULLET_HARDWARE_OPENGL
                                   )
        rgb_2 = np.reshape(img_arr_2[2], (self._height, self._width, 4))[:, :, :3]

        # (256,128,3)
        if self._rgb_only:
            return np.concatenate([rgb_1, rgb_2], axis=0)
        else:
            return [rgb_1, rgb_2, End_Effector_state]


    def full_state(self):
        End_Effector_state = self._Current_End_Effector_State()
        delta_obj_h = np.array([self._obj_high() - self._init_obj_high])
        collision_box = np.array([float(len(p.getContactPoints(bodyA=self._kuka.trayUid,
                                                               bodyB=self._kuka.kukaUid)) != 0)])
        collision_obj = np.array([float(len(p.getContactPoints(bodyA=self._objectUids[0],
                                                               bodyB=self._kuka.kukaUid)) != 0)])

        dis_gripper_2_obj = np.array([self._dis_gripper_2_obj()])
        gripper_close = np.array([float(p.getJointState(bodyUniqueId=self._kuka.kukaUid, jointIndex=11)[0] - \
                                        p.getJointState(bodyUniqueId=self._kuka.kukaUid, jointIndex=8)[0] < 1e-2)])
        obj_2_tray = np.array([self._dis_obj_2_tray()])
        obj = self._low_dim_full_state()[:3]
        full_state = np.concatenate([End_Effector_state,
                                     delta_obj_h,
                                     collision_box,
                                     collision_obj,
                                     dis_gripper_2_obj,
                                     gripper_close,
                                     obj_2_tray, obj], axis=-1)
        return full_state


    def step(self, action, _step=0):
        self._env_step += 1
        self._atomic_action( action )

        if not self._isTest: self._domain_random()

        obs= self._get_observation()

        reward = self._reward()

        done = self._termination()

        debug = { 'is_success': self._success }

        if self._success and self._verbose: print("process: {}\tsuccess!".format(self._proce_num))
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

    def _reward(self):
        # 如果机器人碰到框子。 直接惩罚
        if len(p.getContactPoints(bodyA=self._kuka.trayUid,
                                  bodyB=self._kuka.kukaUid)) != 0:
            self._collision_box = True
            if self._verbose: print("process: {}\txiangzi".format(self._proce_num))
            return -1

        # 出界 有惩罚
        if self.out_of_range:
            if self._verbose: print("process: {}\tout_of_range".format(self._proce_num))
            return -1

        # if self.phase_2:
        #     # [obj drown down on table]  or  [not grasping obj]
        #     if self._obj_high()< 0 or not (len(p.getContactPoints(bodyA=self._objectUids[0],
        #                                                       bodyB=self._kuka.kukaUid)) != 0):
        #         self.drop_down = True
        #         print("process: {}\tDrop down".format(self._proce_num))
        #         return -1

        rank = self._rank_1()

        reward = rank - self._rank_before
        self._rank_before = rank

        if reward < 0:
            self.inverse_rank +=1
        return reward

    def _rank_2(self):

        dis_to_box = self._dis_obj_2_tray()

        if dis_to_box > 0.57: rank = 14
        elif dis_to_box in pyinter.openclosed(0.5, 0.57): rank = 15
        elif dis_to_box in pyinter.openclosed(0.4, 0.5): rank = 16
        elif dis_to_box in pyinter.openclosed(0.3, 0.4): rank = 17
        elif dis_to_box in pyinter.openclosed(0.23, 0.3): rank = 18
        elif dis_to_box <= 0.23:
            self.drop_down = True
            rank = 19
        if self._verbose: print("process: {}\tPhase_2 !!\trank: {}\tdis_to_box:{:.3f}".format(self._proce_num, rank, dis_to_box))
        return rank

    def _rank_1(self):

        self._phase2_bigin = False

        # obj leave table
        if len(p.getContactPoints(bodyA=self._objectUids[0],
                                  bodyB=self._tableUid)) == 0 or self.drop_down:
            if self.drop_down:
                self._release()
                if len(p.getContactPoints(bodyA=self._objectUids[0],
                                      bodyB=self._kuka.trayUid)) != 0:
                    rank = 20
                    if self._verbose: print("process: {}\tAll_success !!".format(self._proce_num))
                else:
                    rank = 19
                    if self._verbose: print("process: {}\t release but not in frame..".format(self._proce_num))
                self._success = True
                return rank

            h = self._obj_high() - self._init_obj_high
            if h <= 0.01:
                rank = 9
                if self._verbose: print("process: {}\tobj rising up [0.5-1] cm !".format(self._proce_num))
            elif h in pyinter.openclosed(0.01, 0.04):
                rank = 10
                if self._verbose: print("process: {}\tobj rising up [1-4] cm !".format(self._proce_num))
            elif h in pyinter.openclosed(0.04, 0.07):
                rank = 11
                if self._verbose: print("process: {}\tobj rising up [4-7] cm !".format(self._proce_num))
            elif h in pyinter.openclosed(0.07, 0.1):
                rank = 12
                if self._verbose: print("process: {}\tobj rising up [7-10] cm !".format(self._proce_num))
            elif h in pyinter.openclosed(0.1, 0.15):
                rank = 13
                if self._verbose: print("process: {}\tobj rising up [10-15] cm !".format(self._proce_num))
            elif 0.15 < h:
                rank = self._rank_2()
                self._phase2_bigin = True
            return rank

        else:
            dis = self._dis_gripper_2_obj()

            if dis>0.57: rank = 0
            elif dis in pyinter.openclosed(0.37, 0.57): rank = 1
            elif dis in pyinter.openclosed(0.27, 0.37): rank = 2
            elif dis in pyinter.openclosed(0.18, 0.27): rank = 3
            elif dis in pyinter.openclosed(0.14, 0.18): rank = 4
            elif dis in pyinter.openclosed(0.09, 0.14): rank = 5
            elif dis in pyinter.openclosed(0.05, 0.09): rank = 6

            # joint Angle difference.
            else:
                gripper_joint_ = p.getJointState(bodyUniqueId=self._kuka.kukaUid, jointIndex=11)[0] - \
                                 p.getJointState(bodyUniqueId=self._kuka.kukaUid, jointIndex=8)[0]

                if gripper_joint_ not in pyinter.openclosed(0.02, 0.4):
                    rank = 7
                    if self._verbose: print("process: {}\tNot Grasping something.\tRank 7.\t{:.2f}".format(self._proce_num, gripper_joint_))
                else:
                    rank = 8
                    if self._verbose: print("process: {}\tGrasping something.\tRank 8.\t{:.2f}".format(self._proce_num, gripper_joint_))
            return rank

    def _dis_gripper_2_obj(self):
        obj = self._low_dim_full_state()[:3]
        current_End_EffectorPos = np.array(p.getLinkState(self._kuka.kukaUid,
                                                          self._kuka.kukaEndEffectorIndex)[0])
        obj_offset = np.array([0.0, 0.02, 0.0], dtype=np.float32)  # 物体的坐标和真实稍稍错位一点， 一点点调出来的。
        gripper_offset = np.array([0.0, 0.0, 0.25], dtype=np.float32)
        dis = obj - current_End_EffectorPos + obj_offset + gripper_offset
        dis = np.sqrt(np.sum(dis ** 2))
        return dis

    def _dis_obj_2_tray(self):
        obj_xy = self._low_dim_full_state()[:2]  # don't use z

        box_pos_xy = np.array(p.getBasePositionAndOrientation(self._kuka.trayUid)[0])[:2]
        dis_to_box = np.sqrt(np.sum((obj_xy - box_pos_xy) ** 2))
        return dis_to_box

    def _termination(self):
        if self._isTest: return (self._env_step >= self._maxSteps) or self._success

        if self.inverse_rank > 2 and self._verbose: print("process: {}\tinverse_rank>2".format(self._proce_num))

        return (self._env_step >= self._maxSteps) or \
               self._success or self.out_of_range or \
               self._collision_box or self.inverse_rank>2

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
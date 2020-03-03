# coding=utf-8
import numpy as np

import tensorflow as tf
config=tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True))
from sac import Count_Variables
import alg.ppo_core as ppo_core
from spinup.utils.mpi_tf import MpiAdamOptimizer, sync_all_params
from spinup.utils.mpi_tools import mpi_fork, mpi_avg, proc_id, mpi_statistics_scalar
from tqdm import tqdm

import random
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style="darkgrid", font_scale=1.2)

from sac import R_plot

class PPOBuffer:
    """
    A buffer for storing trajectories experienced by a PPO agent interacting
    with the environment, and using Generalized Advantage Estimation (GAE-Lambda)
    for calculating the advantages of state-action pairs.
    """

    def __init__(self, obs_dim, act_dim, size, gamma=0.99, lam=0.95):
        self.obs_buf = np.zeros(ppo_core.combined_shape(size, obs_dim), dtype=np.float32)
        self.obs_buf_2 = np.zeros(ppo_core.combined_shape(size, obs_dim), dtype=np.float32)
        self.obs_End_buf = np.zeros(ppo_core.combined_shape(size, 5), dtype=np.float32)

        self.f_s = np.zeros(ppo_core.combined_shape(size, 11), dtype=np.float32)

        self.act_buf = np.zeros(ppo_core.combined_shape(size, act_dim), dtype=np.float32)
        self.adv_buf = np.zeros(size, dtype=np.float32)
        self.rew_buf = np.zeros(size, dtype=np.float32)
        self.ret_buf = np.zeros(size, dtype=np.float32)
        self.val_buf = np.zeros(size, dtype=np.float32)
        self.logp_buf = np.zeros(size, dtype=np.float32)
        self.gamma, self.lam = gamma, lam
        self.ptr, self.path_start_idx, self.max_size = 0, 0, size

    def store(self, obs, f_s, act, rew, val, logp):
        """
        Append one timestep of agent-environment interaction to the buffer.
        """
        assert self.ptr < self.max_size  # buffer has to have room so you can store
        self.obs_buf[self.ptr] = obs[0]
        self.obs_buf_2[self.ptr] = obs[1]
        self.obs_End_buf[self.ptr] = obs[2]

        self.f_s[self.ptr] = f_s

        self.act_buf[self.ptr] = act
        self.rew_buf[self.ptr] = rew
        self.val_buf[self.ptr] = val
        self.logp_buf[self.ptr] = logp
        self.ptr += 1

    def finish_path(self, last_val=0):
        """
        Call this at the end of a trajectory, or when one gets cut off
        by an epoch ending. This looks back in the buffer to where the
        trajectory started, and uses rewards and value estimates from
        the whole trajectory to compute advantage estimates with GAE-Lambda,
        as well as compute the rewards-to-go for each state, to use as
        the targets for the value function.

        The "last_val" argument should be 0 if the trajectory ended
        because the agent reached a terminal state (died), and otherwise
        should be V(s_T), the value function estimated for the last state.
        This allows us to bootstrap the reward-to-go calculation to account
        for timesteps beyond the arbitrary episode horizon (or epoch cutoff).
        """

        path_slice = slice(self.path_start_idx, self.ptr)
        rews = np.append(self.rew_buf[path_slice], last_val)
        vals = np.append(self.val_buf[path_slice], last_val)

        # the next two lines implement GAE-Lambda advantage calculation
        deltas = rews[:-1] + self.gamma * vals[1:] - vals[:-1]
        self.adv_buf[path_slice] = ppo_core.discount_cumsum(deltas, self.gamma * self.lam)

        # the next line computes rewards-to-go, to be targets for the value function
        self.ret_buf[path_slice] = ppo_core.discount_cumsum(rews, self.gamma)[:-1]

        self.path_start_idx = self.ptr

    def get(self):
        """
        Call this at the end of an epoch to get all of the data from
        the buffer, with advantages appropriately normalized (shifted to have
        mean zero and std one). Also, resets some pointers in the buffer.
        """
        assert self.ptr == self.max_size  # buffer has to be full before you can get
        self.ptr, self.path_start_idx = 0, 0
        # the next two lines implement the advantage normalization trick
        adv_mean, adv_std = mpi_statistics_scalar(self.adv_buf)
        self.adv_buf = (self.adv_buf - adv_mean) / adv_std
        return [self.obs_buf, self.obs_buf_2, self.obs_End_buf, self.f_s, self.act_buf, self.adv_buf,
                self.ret_buf, self.logp_buf]


class ppo:
    def __init__(self, env_fn, seed=0,
        steps_per_epoch=4000, epochs=50, gamma=0.99, clip_ratio=0.2, pi_lr=3e-4,
        vf_lr=1e-3, train_pi_iters=80, train_v_iters=80, lam=0.97, max_ep_len=1000,
        target_kl=0.01, save_freq=10, exp_name='', test_agent=False, load=False):
        # seed
        self.seed = seed
        self.seed += 10000 * proc_id()
        tf.set_random_seed(self.seed)
        np.random.seed(self.seed)
        random.seed(self.seed)

        # Hyper-parameter
        self.env = env_fn()
        self.obs_dim = self.env.observation_space.shape
        self.act_dim = self.env.action_space.shape
        self.gamma, self.lam = gamma, lam
        self.steps_per_epoch, self.epochs = steps_per_epoch, epochs

        self.pi_lr, self.vf_lr = pi_lr, vf_lr
        self.clip_ratio, self.target_kl = clip_ratio, target_kl
        self.train_pi_iters, self.train_v_iters = train_pi_iters, train_v_iters
        self.exp_name, self.save_freq, self.max_ep_len = exp_name, save_freq, max_ep_len

        if test_agent:
            from visdom import Visdom
            self.viz = Visdom()
            assert self.viz.check_connection()
            self.win = self.viz.matplot(plt)

        # Experience buffer
        self.buf = PPOBuffer(self.obs_dim, self.act_dim, self.steps_per_epoch, self.gamma, self.lam)

        self.graph = tf.Graph()
        with self.graph.as_default():
            self.sess = tf.Session(config=config, graph=self.graph)

            self.__make_model()
            self.sess.run(tf.global_variables_initializer())

            # Sync params across processes
            self.sess.run(sync_all_params())
            Count_Variables()
            print('Trainable_variables:')
            for v in tf.compat.v1.trainable_variables():
                print('{}\t  {}'.format(v.name, str(v.shape)))

            var_list = tf.global_variables()
            self.saver = tf.compat.v1.train.Saver(var_list=var_list, max_to_keep=1)

            # summary
            self.writer = tf.compat.v1.summary.FileWriter("logs/" + exp_name)
            if load:
                self.load()
            self.sess.run(sync_all_params())

            self.ep_ret_ph = tf.placeholder(tf.float32, shape=(), name="ep_ret_ph")
            self.ep_Entropy_ph = tf.placeholder(tf.float32, shape=(), name="Entropy")
            self.clipfrac_ph = tf.placeholder(tf.float32, shape=(), name="clipfrac")
            self.ep_len_ph = tf.placeholder(tf.float32, shape=(), name="ep_len_ph")
            self.test_summary = tf.compat.v1.summary.merge(
                [tf.compat.v1.summary.scalar('EP_ret', self.ep_ret_ph, family='test'),
                 tf.compat.v1.summary.scalar('EP_len', self.ep_len_ph, family='test')])
            self.entropy_summary = tf.compat.v1.summary.merge(
                [tf.compat.v1.summary.scalar('Entropy', self.ep_Entropy_ph, family='test'),
                 tf.compat.v1.summary.scalar('clipfrac', self.clipfrac_ph, family='test')])

    def __make_model(self):

        self.o1_ph = tf.placeholder(tf.float32, [None, 128, 128, 3], name='o1_ph')
        self.o2_ph = tf.placeholder(tf.float32, [None, 128, 128, 3], name='o2_ph')
        self.o_low_dim_ph = tf.placeholder(tf.float32, [None, 5], name='o_low_dim_ph')
        self.f_s_ph = tf.placeholder(tf.float32, [None, 11], name='f_s_ph')

        a_ph, adv_ph, ret_ph, logp_old_ph = ppo_core.placeholders(self.act_dim, None, None, None)
        pi, logp, logp_pi, self.v = ppo_core.mlp_actor_critic(self.o1_ph, self.o2_ph, self.o_low_dim_ph, self.f_s_ph, a_ph,
                                                              action_space=self.env.action_space)

        self.all_phs = [self.o1_ph, self.o2_ph, self.o_low_dim_ph, self.f_s_ph, a_ph, adv_ph, ret_ph, logp_old_ph]

        self.get_action_ops = [pi, self.v, logp_pi]

        # PPO objectives
        ratio = tf.exp(logp - logp_old_ph)  # pi(a|s) / pi_old(a|s)
        min_adv = tf.where(adv_ph > 0, (1 + self.clip_ratio) * adv_ph, (1 - self.clip_ratio) * adv_ph)
        self.pi_loss = -tf.reduce_mean(tf.minimum(ratio * adv_ph, min_adv))
        self.v_loss = tf.reduce_mean((ret_ph - self.v) ** 2)

        # Info (useful to watch during learning)
        self.approx_kl = tf.reduce_mean(logp_old_ph - logp)  # a sample estimate for KL-divergence, easy to compute
        self.approx_ent = tf.reduce_mean(-logp)  # a sample estimate for entropy, also easy to compute
        clipped = tf.logical_or(ratio > (1 + self.clip_ratio), ratio < (1 - self.clip_ratio))
        self.clipfrac = tf.reduce_mean(tf.cast(clipped, tf.float32))

        # Optimizers
        self.train_pi = MpiAdamOptimizer(learning_rate=self.pi_lr).minimize(self.pi_loss)
        self.train_v = MpiAdamOptimizer(learning_rate=self.vf_lr).minimize(self.v_loss)

    def update(self):
        inputs = {k: v for k, v in zip(self.all_phs, self.buf.get())}

        # Training
        for i in range(self.train_pi_iters):
            _, kl = self.sess.run([self.train_pi, self.approx_kl], feed_dict=inputs)
            kl = mpi_avg(kl)

            if kl > 1.5 * self.target_kl:
                print('process %d: Early stopping at step %d due to reaching max kl.' % (proc_id(), i))
                break

        for _ in range(self.train_v_iters):
            self.sess.run(self.train_v, feed_dict=inputs)

    def rollout(self):
        o, r, d, ep_ret, ep_len = self.env.reset(), 0, False, 0, 0

        # Main loop: collect experience in env and update/log each epoch
        for epoch in tqdm(range(self.epochs)):
            for t in range(self.steps_per_epoch):
                f_s = self.env.full_state()
                a, v_t, logp_t = self.sess.run(self.get_action_ops, feed_dict={self.o1_ph: o[0][np.newaxis, ],
                                                                               self.o2_ph: o[1][np.newaxis, ],
                                                                               self.o_low_dim_ph: o[2][np.newaxis,],
                                                                               self.f_s_ph: f_s[np.newaxis,]})

                o2, r, d, _ = self.env.step(a[0])
                ep_ret += r
                ep_len += 1

                # save and log
                self.buf.store(o, f_s, a, r, v_t, logp_t)

                # Update obs (critical!)
                o = o2

                terminal = d or (ep_len == self.max_ep_len)
                if terminal or (t == self.steps_per_epoch - 1):
                    if not (terminal):
                        print('process %d: trajectory cut off by epoch at %d steps.' % (proc_id(), ep_len))
                    # if trajectory didn't reach terminal state, bootstrap value target
                    last_val = 0 if d else self.sess.run(self.v, feed_dict={self.f_s_ph: f_s[np.newaxis,]})
                    self.buf.finish_path(last_val)

                    if proc_id() == 0:
                        ep_s = self.sess.run(self.test_summary, {self.ep_ret_ph: ep_ret,
                                                                 self.ep_len_ph: ep_len})
                        self.writer.add_summary(ep_s, global_step=epoch)

                    o, ep_ret, ep_len = self.env.reset(), 0, 0

            # Save model
            if (epoch % self.save_freq == 0) or (epoch == self.epochs - 1):
                if proc_id()==0:
                    self.Save()

            # Perform PPO update!
            self.update()

    def Save(self):
        path = "model/" + self.exp_name + "/model.ckpt"
        print("process %d:  Save model to the: '{}'".format(path) % proc_id())
        self.saver.save(self.sess, save_path=path)

    def load(self):
        path = "model/" + self.exp_name + "/model.ckpt"
        print("\nLoad model From the: '{}'\n".format(path) )
        self.saver.restore(self.sess, save_path=path)

    def _choose_action(self, o):
        a, v_t, logp_t = self.sess.run(self.get_action_ops, feed_dict={self.o1_ph: o[0][np.newaxis,],
                                                                       self.o2_ph: o[1][np.newaxis,],
                                                                       self.o_low_dim_ph: o[2][np.newaxis,]})
        return a[0], v_t

    def test_agent(self, n=1):
        ep_r = []
        ep_l = []
        plt.figure(figsize=(9, 5))
        for j in range( n ):
            print('------------------Epoch: %d-------------------'%j)

            o, r, d, ep_ret, ep_len, level_plot = self.env.reset(), 0, False, 0, 0, []
            v_plot = []
            while not(d or (ep_len == self.max_ep_len)):

                # Take deterministic actions at test time
                a,v_t = self._choose_action(o)
                print('V: {}'.format(v_t[0]))
                v_plot.append( v_t )
                # _ = input()
                o, r, d, info = self.env.step(a, _step=ep_len)
                print("step: {} a: {}  r: {}".format(ep_len, np.around(a, decimals=2), r) )
                ep_ret += r
                ep_len += 1
                level_plot.append(self.env._rank_before)
                self.win = R_plot(level_plot, v_plot, self.viz, self.win)

            print('EP_ret: %d, \t EP_Len: %d' % (ep_ret, ep_len))

            if info['is_success']:
                print('success!')
            ep_r.append(ep_ret), ep_l.append(ep_len)
        plt.close()

if __name__ == '__main__':

    # import os
    # os.environ["CUDA_VISIBLE_DEVICES"] = str(proc_id()%2)

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--gamma', type=float, default=0.90)
    parser.add_argument('--seed', '-s', type=int, default=0)
    parser.add_argument('--cpu', type=int, default=1)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--exp_name', type=str, default='ppo')
    parser.add_argument('--load', action="store_true")
    args = parser.parse_args()

    import shutil
    try:
        shutil.rmtree("logs/" + str(args['exp_name']))
    except:
        pass

    mpi_fork(args.cpu)  # run parallel code with mpi

    from env.KukaGymEnv import KukaDiverseObjectEnv
    MAX_STEP = 64
    env_fn = lambda: KukaDiverseObjectEnv(renders=False,
                                          maxSteps=MAX_STEP,
                                          blockRandom=0.2,
                                          actionRepeat=200,
                                          numObjects=1, dv=1.0,
                                          isTest=False,
                                          proce_num=proc_id(), single_img=False,
                                          rgb_only=False)

    model = ppo(env_fn, gamma=args.gamma,
        seed=args.seed, steps_per_epoch=MAX_STEP, epochs=args.epochs,
                exp_name=args.exp_name, load=args.load)

    model.rollout()


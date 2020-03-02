# coding=utf-8
import numpy as np
import alg.core as core
from alg.core import get_vars
from env.KukaGymEnv import KukaDiverseObjectEnv
from tqdm import tqdm
from PIL import Image

import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style="darkgrid", font_scale=1.2)
import argparse

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import tensorflow as tf
config=tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True),
                      log_device_placement=True)

def common_arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment_name", type=str)
    parser.add_argument('--use_sample_dist', action="store_true")
    parser.add_argument('--use_half_dete', action="store_true")
    parser.add_argument('--use_adv', action="store_true")
    dict_args = vars(parser.parse_args())
    return dict_args

class ReplayBuffer:
    """
    A simple FIFO experience replay buffer for SAC agents.
    """

    def __init__(self, act_dim, size):
        self.obs1_buf = np.zeros([size, 128, 128, 3], dtype=np.float32)
        self.obs2_buf = np.zeros([size, 128, 128, 3], dtype=np.float32)

        self.acts_buf = np.zeros([size, act_dim], dtype=np.float32)
        self.rews_buf = np.zeros(size, dtype=np.float32)
        self.done_buf = np.zeros(size, dtype=np.float32)
        self.ptr, self.size, self.max_size = 0, 0, size

    def store(self, obs, act, rew, next_obs, done):
        self.obs1_buf[self.ptr] = obs[0]

        self.obs2_buf[self.ptr] = next_obs[0]
        self.acts_buf[self.ptr] = act
        self.rews_buf[self.ptr] = rew
        self.done_buf[self.ptr] = done
        self.ptr = (self.ptr+1) % self.max_size
        self.size = min(self.size+1, self.max_size)

    def sample_batch(self, batch_size=32):
        idxs = np.random.randint(0, self.size, size=batch_size)
        return dict(obs1=self.obs1_buf[idxs],
                    obs2=self.obs2_buf[idxs],
                    acts=self.acts_buf[idxs],
                    rews=self.rews_buf[idxs],
                    done=self.done_buf[idxs])

class sac_ES:
    def __init__(self, env_fn, seed=0,
        steps_per_epoch=5000, epochs=100, gamma=0.7,
        polyak=0.995, lr=3e-4, alpha=0.2, batch_size=1024, start_steps=1,
        max_ep_len=1000, memory_size=int(1e+4), max_kl=0.01, experiment_name='none', use_adv=False,
        use_half_dete=False, use_sample_dist=False, Test_agent=False):

        tf.set_random_seed(seed)
        np.random.seed(seed)
        self._success_obs_item = len( os.listdir('result/success_obs/'))
        self.env = env_fn()
        self.ac_per_state = 30
        self.memory_size = memory_size
        self.batch_size = batch_size
        self.gamma = gamma
        self.alpha = alpha
        self.lr = lr
        self.polyak = polyak
        self.start_steps = start_steps
        self.max_ep_len = max_ep_len
        self.epochs = epochs
        self.steps_per_epoch = steps_per_epoch
        self.max_kl = max_kl
        self.use_adv = use_adv
        self.entropy_too_low = False
        self.win = None
        if Test_agent:
            from visdom import Visdom
            self.viz = Visdom()
            assert self.viz.check_connection()
            self.win = self.viz.matplot(plt)

        self.act_dim = self.env.action_space.shape[0]
        self.replay_buffer = ReplayBuffer(act_dim=self.act_dim, size=self.memory_size)
        self.use_half_dete = use_half_dete
        self.use_sample_dist = use_sample_dist

        # Build computer graph
        self.graph = tf.Graph()

        with tf.device('/gpu:0'), self.graph.as_default():
            self.sess = tf.Session(config=config, graph=self.graph)

            self.__make_model()

            # Summary
            self.experiment_name = experiment_name
            self.writer = tf.compat.v1.summary.FileWriter("logs/" + self.experiment_name)
            self.ep_ret_ph = tf.placeholder(tf.float32, shape=(), name="ep_ret_ph")
            self.ep_len_ph = tf.placeholder(tf.float32, shape=(), name="ep_len_ph")
            self.test_summary = tf.compat.v1.summary.merge([tf.compat.v1.summary.scalar('EP_ret', self.ep_ret_ph, family='test'),
                                                  tf.compat.v1.summary.scalar('EP_len', self.ep_len_ph, family='test')])
            self.sess.run(tf.global_variables_initializer())
            self.sess.run(self.V_target_init)

            _ = os.system("clear")
            Count_Variables()
            print('Trainable_variables:')
            for v in tf.compat.v1.trainable_variables():
                print('{}\t  {}'.format(v.name, str(v.shape)))

    def __make_model(self):
        self.x1_ph = tf.placeholder(tf.float32, [None, 128, 128, 3], name='o1_ph')

        self.x2_ph = tf.placeholder(tf.float32, [None, 128, 128, 3], name='o2_ph')

        self.a_ph, self.r_ph, self.d_ph, self.q_ij_ph = core.placeholders(self.act_dim, None, None, self.ac_per_state)

        with tf.variable_scope('main'):
            self.dist, self.sample_dist, self.mean, self.pi, self.q1, self.q2, self.q1_pi, self.q2_pi, self.v, self.std = core.actor_critic(
                self.x1_ph, self.a_ph, action_space=self.env.action_space, hidden_size=(64, 32))

        vf_mlp = lambda x_: tf.squeeze( core.mlp( core._cnn(x_),
                                                         [64, 32, 1], tf.nn.relu, None), axis=1)

        with tf.variable_scope('target'):
            self.v_target = vf_mlp(self.x2_ph)

        # Policy Evaluate
        with tf.variable_scope('Policy_Evaluate'):
            # Min Double-Q:
            min_q_pi = tf.minimum( self.q1_pi, self.q2_pi )

            # Targets for Q and V regression
            q_target = tf.stop_gradient(self.r_ph + self.gamma * (1 - self.d_ph) * self.v_target)
            logp_pi =  tf.reduce_sum(self.dist.log_prob(self.pi), axis=-1, keep_dims=True) # sum the each dim_of_A
            v_target = tf.stop_gradient(min_q_pi - self.alpha * logp_pi)

            q1_loss = 0.5 * tf.reduce_mean((q_target - self.q1) ** 2)
            q2_loss = 0.5 * tf.reduce_mean((q_target - self.q2) ** 2)
            v_loss = 0.5 * tf.reduce_mean((v_target - self.v) ** 2)
            value_loss = q1_loss + q2_loss + v_loss
        # set_target_update
        self.V_target_init = tf.group([tf.assign(v_targ, v_main)
                                       for v_main, v_targ in zip(get_vars('main/v'), get_vars('target'))])

        # Policy Improvement
        with tf.variable_scope('Policy_Improvement'):
            # (samples_per_state, batch, self.act_dim) --> (-1, self.act_dim)
            self.a_flatten_s_dist =  self._samp_a( self.sample_dist )
            self.a_flatten_dist   =  self._samp_a( self.dist )

            if self.use_adv:
                q = self.q1 - self.v_target
            else:
                q = self.q1

            q_ij = tf.stop_gradient( tf.nn.softmax( tf.reshape( q, [self.ac_per_state, -1]), axis=0) )

            logp = tf.reduce_sum(self.dist.log_prob(self.a_ph), axis=-1, keep_dims=True)  # sum the each dim_of_A
            logp_ij = tf.reshape( logp, [self.ac_per_state, -1])

            self.entropy = tf.reduce_mean( self.dist.entropy() )

            likelihood_term = q_ij * logp_ij

        # Value train op
        value_optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=self.lr)
        value_params = get_vars('main/q') + get_vars('main/v')

        self.train_value_op = value_optimizer.minimize(value_loss, var_list=value_params)

        # value_learn -> update_target_value
        self.soft_update = tf.group([tf.assign(v_targ, self.polyak * v_targ + (1 - self.polyak) * v_main)
                                         for v_main, v_targ in zip(get_vars('main/v'), get_vars('target'))])

        # Maximum likelihood
        pi_loss = - tf.reduce_sum( likelihood_term )
        pi_optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=self.lr)
        self.train_pi_op = pi_optimizer.minimize(pi_loss, var_list=get_vars('main/pi'))

        var_list = tf.global_variables()
        self.saver = tf.compat.v1.train.Saver(var_list=var_list, max_to_keep=1)

        assert tf.get_collection(tf.GraphKeys.UPDATE_OPS) == []  # 确保没有 batch norml

        # summary
        self.a_summary = tf.compat.v1.summary.merge([tf.compat.v1.summary.scalar('pi_loss', pi_loss, family='actor')])
        self.c_summary = tf.compat.v1.summary.merge([tf.compat.v1.summary.scalar('value_loss', value_loss, family='critic'),
                                           tf.compat.v1.summary.scalar('q1_loss', q1_loss, family = 'critic'),
                                           tf.compat.v1.summary.scalar('q2_loss', q2_loss, family='critic'),
                                           tf.compat.v1.summary.scalar('v_loss', v_loss, family='critic'),
                                           tf.compat.v1.summary.scalar('pi_entropy', self.entropy, family='actor')])

    def _samp_a(self, dist):
        # (samples_per_state, batch, self.act_dim) --> (-1, self.act_dim)
        a = tf.reshape(dist.sample(self.ac_per_state), [-1, self.act_dim]),
        return tf.clip_by_value( a , self.env.action_space.low[0], self.env.action_space.high[0])

    def Save(self):
        path = "model/" + self.experiment_name + "/model.ckpt"
        # print("Save model to the: '{}'".format(path))
        self.saver.save(self.sess, save_path=path)

    def load(self):
        path = "model/" + self.experiment_name + "/model.ckpt"
        print("\nLoad model From the: '{}'\n".format(path) )
        self.saver.restore(self.sess, save_path=path)

    def _choose_action(self, o, deterministic=False):
        act_op = self.mean if deterministic else self.pi
        return self.sess.run(act_op, feed_dict={self.x1_ph: o[np.newaxis,]})[0]

    def learn(self):
        batch = self.replay_buffer.sample_batch(self.batch_size)

        a_flatten_opt = self.a_flatten_s_dist #if self.entropy_too_low else self.a_flatten_dist

        opt_step_1 = [self.train_value_op, a_flatten_opt, self.c_summary]
        feed_step_1 = { self.x1_ph: batch['obs1'],
                        self.x2_ph: batch['obs2'],
                        self.a_ph: batch['acts'],
                        self.r_ph: batch['rews'],
                        self.d_ph: batch['done']
                        }
        _, a_Flatten, c_s= self.sess.run(opt_step_1, feed_step_1 )
        self.writer.add_summary(c_s)

        feed_step_2 = {self.a_ph: np.squeeze(a_Flatten),
                       self.x1_ph: np.tile(batch['obs1'], [self.ac_per_state, 1, 1 , 1]),
                       self.x2_ph: np.tile(batch['obs2'], [self.ac_per_state, 1, 1, 1])}
        opt_step_2 = [self.entropy, self.train_pi_op, self.a_summary]

        # update policy by max_likelihood
        entropy, _, a_s = self.sess.run( opt_step_2, feed_step_2 )

        self.entropy_too_low = True  if entropy < -0.5 else False

        self.writer.add_summary(a_s)

        # update target_net and old pi
        self.sess.run( self.soft_update )

    def test_agent(self, deterministic=False, n=1):
        ep_r = []
        ep_l = []
        plt.figure(figsize=(9, 5))
        for j in range( n ):
            print('------------------Epoch: %d-------------------'%j)

            o, r, d, ep_ret, ep_len, level_plot = self.env.reset(), 0, False, 0, 0, []

            while not(d or (ep_len == self.max_ep_len)):

                # Take deterministic actions at test time
                a = self._choose_action(o,
                                    deterministic=deterministic)
                o, r, d, info = self.env.step(a, _step=ep_len)
                print("step: {} a: {}  r: {}".format(ep_len, np.around(a, decimals=2), r) )
                ep_ret += r
                ep_len += 1
                level_plot.append(self.env._before)
                self.win = R_plot(level_plot, self.viz, self.win)

            print('EP_ret: %d, \t EP_Len: %d' % (ep_ret, ep_len))

            if info['is_success']:
                print('success!')
            ep_r.append(ep_ret), ep_l.append(ep_len)
        plt.close()

    def rollout(self):
        total_steps = self.steps_per_epoch * self.epochs
        o, r, d, ep_ret, ep_len = self.env.reset(), 0, False, 0, 0

        deterministic = False
        log_t = []
        log_episode_ret = []

        for t in tqdm(range(total_steps)):

            if t > self.start_steps:
                if self.use_half_dete:
                    deterministic = True if np.random.randn() > 0 else False
                a = self._choose_action(o, deterministic)
            else:
                a = self.env.action_space.sample()

            o2, r, d, info = self.env.step(a)
            ep_ret += r
            ep_len += 1

            if info['is_success']:
                im = Image.fromarray(o2)
                im.save( 'result/success_obs/' + str(self._success_obs_item)+'.png' )
                self._success_obs_item +=1

            d = False if ep_len == self.max_ep_len else d

            self.replay_buffer.store(o, a, r, o2, d)

            o = o2

            if d or (ep_len == self.max_ep_len):

                if (t+1) >= 500: [self.learn() for _ in range(ep_len)] # todo::

                ep_s = self.sess.run(self.test_summary, {self.ep_ret_ph: ep_ret,
                                                           self.ep_len_ph: ep_len})
                self.writer.add_summary(ep_s, global_step=t)
                log_episode_ret.append(ep_ret)
                log_t.append(t)
                print("episode return: ", ep_ret)
                o, r, d, ep_ret, ep_len = self.env.reset(), 0, False, 0, 0

            if (t+1)%100 ==0:
                self.Save()

        np.save('result/'+self.experiment_name+'_reward.npy', (log_t, log_episode_ret), allow_pickle=True)

def Count_Variables():
    var_counts = tuple(core.count_vars(scope) for scope in
                       ['pi', 'v'])
    print(('\nNumber of parameters: \t pi: %d, \t ' +
           'v: %d\n') % var_counts)

def R_plot(r_plot,v_plot, viz, win):
    plt.cla()
    plt.plot(r_plot, color='r', label='Rank')
    plt.plot(v_plot, color='blue', label='V')

    # plt.ylim(0,15)
    # plt.xlim(0,25)
    plt.xlabel('Step')
    plt.ylabel('Rank')
    plt.title('Rank Trance')
    plt.legend(loc='upper right')
    return viz.matplot(plt, win=win)

if __name__ == '__main__':
    args = common_arg_parser()
    import shutil
    try:
        shutil.rmtree("logs/"+args['experiment_name'])
    except:
        pass


    MAX_STEP=150
    env_fn = lambda : KukaDiverseObjectEnv(renders=False,
                         maxSteps=MAX_STEP,
                         blockRandom=0.3,
                         actionRepeat=200,
                         numObjects=1, dv=1.0,
                         isTest=False)

    model = sac_ES(env_fn, steps_per_epoch=MAX_STEP, epochs=700, gamma=0.0,
                   polyak=0.995, memory_size=20000, lr=3e-4, alpha=0.2,
                   batch_size=32, start_steps=300, max_ep_len=500, **args)

    model.rollout()

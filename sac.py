# coding=utf-8
import numpy as np
import alg.core as core
from alg.core import get_vars
from env.KukaGymEnv import KukaDiverseObjectEnv
from tqdm import tqdm
import os
from PIL import Image

import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style="darkgrid", font_scale=1.2)

import tensorflow as tf
config=tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True))

class ReplayBuffer:
    """
    A simple FIFO experience replay buffer for SAC agents.
    """

    def __init__(self, act_dim, size):
        self.obs1_b_buf = np.zeros([size, 128, 128, 3], dtype=np.float32)
        self.obs1_l_buf = np.zeros([size, 32, 32, 3], dtype=np.float32)

        self.obs2_b_buf = np.zeros([size, 128, 128, 3], dtype=np.float32)
        self.obs2_l_buf = np.zeros([size, 32, 32, 3], dtype=np.float32)

        self.acts_buf = np.zeros([size, act_dim], dtype=np.float32)
        self.rews_buf = np.zeros(size, dtype=np.float32)
        self.done_buf = np.zeros(size, dtype=np.float32)
        self.ptr, self.size, self.max_size = 0, 0, size

    def store(self, obs, act, rew, next_obs, done):
        self.obs1_b_buf[self.ptr] = obs[0]
        self.obs1_l_buf[self.ptr] = obs[1]

        self.obs2_b_buf[self.ptr] = next_obs[0]
        self.obs2_l_buf[self.ptr] = next_obs[1]
        self.acts_buf[self.ptr] = act
        self.rews_buf[self.ptr] = rew
        self.done_buf[self.ptr] = done
        self.ptr = (self.ptr+1) % self.max_size
        self.size = min(self.size+1, self.max_size)

    def sample_batch(self, batch_size=32):
        idxs = np.random.randint(0, self.size, size=batch_size)
        return dict(obs1_b=self.obs1_b_buf[idxs],
                    obs1_l=self.obs1_l_buf[idxs],
                    obs2_b=self.obs2_b_buf[idxs],
                    obs2_l=self.obs2_l_buf[idxs],
                    acts=self.acts_buf[idxs],
                    rews=self.rews_buf[idxs],
                    done=self.done_buf[idxs])

class sac_ES:
    def __init__(self, env_fn, seed=0,
        steps_per_epoch=5000, epochs=100, gamma=0.7,
        polyak=0.995, lr=3e-4, alpha=0.2, batch_size=1024, start_steps=1,
        max_ep_len=1000, memory_size=int(1e+4), max_kl=0.01, experiment_name='none', use_adv=False,
        use_half_dete=False, use_sample_dist=False, save_freq=1, Test_agent=False):

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

        with self.graph.as_default():
            self.sess = tf.Session(config=config, graph=self.graph)

            self.__make_model()

            # Summary
            self.experiment_name = experiment_name
            self.writer = tf.summary.FileWriter("logs/" + self.experiment_name)
            self.ep_ret_ph = tf.placeholder(tf.float32, shape=(), name="ep_ret_ph")
            self.ep_len_ph = tf.placeholder(tf.float32, shape=(), name="ep_len_ph")
            self.test_summary = tf.summary.merge([tf.summary.scalar('EP_ret', self.ep_ret_ph, family='test'),
                                                  tf.summary.scalar('EP_len', self.ep_len_ph, family='test')])
            self.sess.run(tf.global_variables_initializer())
            self.sess.run(self.V_target_init)

            _ = os.system("clear")
            #
            # while input_var not in ['yes', 'no']:
            #     input_var = input("是否用 siamese_cnn 初始化卷积层参数. [yes/no]?")
            # if input_var == 'yes':
            #     siamese_cnn_dict = np.load('model/siamese_cnn.npy', allow_pickle=True).item()
            #     for para_Name in siamese_cnn_dict:
            #         for var in [v for v in tf.trainable_variables() if v.name in para_Name]:
            #             self.sess.run(tf.assign(var, siamese_cnn_dict[para_Name]))

            Count_Variables()

            print('Trainable_variables:')
            for v in tf.trainable_variables():
                print(v)

    def __make_model(self):
        self.x_b_ph = tf.placeholder(tf.float32, [None, 128, 128, 3], name='x_b_ph')
        self.x_l_ph = tf.placeholder(tf.float32, [None, 32, 32, 3], name='x_l_ph')

        self.x2_b_ph = tf.placeholder(tf.float32, [None, 128, 128, 3], name='x2_b_ph')
        self.x2_l_ph = tf.placeholder(tf.float32, [None, 32, 32, 3], name='x2_l_ph')

        self.a_ph, self.r_ph, self.d_ph, self.q_ij_ph = core.placeholders(self.act_dim, None, None, self.ac_per_state)

        with tf.variable_scope('main'):
            self.dist, self.sample_dist, self.mean, self.pi, self.q1, self.q2, self.q1_pi, self.q2_pi, self.v, self.std = core.actor_critic(
                self.x_b_ph, self.x_l_ph , self.a_ph, action_space=self.env.action_space, hidden_size=(64, 32))

        vf_mlp = lambda x_b, x_l, : tf.squeeze( core.mlp(tf.concat([core.cnn_b(x_b), core.cnn_l(x_l)], axis=1),
                                                         [64, 32, 1], tf.nn.relu, None), axis=1)

        with tf.variable_scope('target'):
            self.v_target = vf_mlp( self.x2_b_ph, self.x2_l_ph)

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
            if self.use_sample_dist:
                self.a_flatten = tf.clip_by_value(tf.reshape(self.sample_dist.sample(self.ac_per_state), [-1, self.act_dim]),
                                              self.env.action_space.low[0], self.env.action_space.high[0])
            else:
                self.a_flatten = tf.clip_by_value(tf.reshape(self.dist.sample(self.ac_per_state), [-1, self.act_dim]),
                                              self.env.action_space.low[0], self.env.action_space.high[0])

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
        value_optimizer = tf.train.AdamOptimizer(learning_rate=self.lr)
        value_params = get_vars('main/q') + get_vars('main/v')

        self.train_value_op = value_optimizer.minimize(value_loss, var_list=value_params)

        # value_learn -> update_target_value
        self.soft_update = tf.group([tf.assign(v_targ, self.polyak * v_targ + (1 - self.polyak) * v_main)
                                         for v_main, v_targ in zip(get_vars('main/v'), get_vars('target'))])

        # Maximum likelihood
        pi_loss = - tf.reduce_sum( likelihood_term )
        pi_optimizer = tf.train.AdamOptimizer(learning_rate=self.lr)
        self.train_pi_op = pi_optimizer.minimize(pi_loss, var_list=get_vars('main/pi'))

        var_list = tf.global_variables()
        self.saver = tf.train.Saver(var_list=var_list, max_to_keep=1)

        assert tf.get_collection(tf.GraphKeys.UPDATE_OPS) == []  # 确保没有 batch norml

        # summary
        self.a_summary = tf.summary.merge([tf.summary.scalar('pi_loss', pi_loss, family='actor')])
        self.c_summary = tf.summary.merge([tf.summary.scalar('value_loss', value_loss, family='critic'),
                                           tf.summary.scalar('q1_loss', q1_loss, family = 'critic'),
                                           tf.summary.scalar('q2_loss', q2_loss, family='critic'),
                                           tf.summary.scalar('v_loss', v_loss, family='critic'),
                                           tf.summary.scalar('pi_entropy', self.entropy, family='actor')])

    def Save(self):
        path = "model/" + self.experiment_name + "/model.ckpt"
        # print("Save model to the: '{}'".format(path))
        self.saver.save(self.sess, save_path=path)

    def load(self):
        path = "model/" + self.experiment_name + "/model.ckpt"
        print("Load model From the: '{}'".format(path) )
        self.saver.restore(self.sess, save_path=path)

    def _choose_action(self, o, deterministic=False):
        act_op = self.mean if deterministic else self.pi
        return self.sess.run( act_op, feed_dict={self.x_b_ph: o[0][np.newaxis, ],
                                                 self.x_l_ph: o[1][np.newaxis, ]})[0]

    def learn(self):
        batch = self.replay_buffer.sample_batch(self.batch_size)

        opt_step_1 = [self.train_value_op, self.a_flatten, self.c_summary]
        feed_step_1 = { self.x_b_ph: batch['obs1_b'],
                        self.x_l_ph: batch['obs1_l'],
                        self.x2_b_ph: batch['obs2_b'],
                        self.x2_l_ph: batch['obs2_l'],
                        self.a_ph: batch['acts'],
                        self.r_ph: batch['rews'],
                        self.d_ph: batch['done']
                        }
        _, a_flatten, c_s= self.sess.run(opt_step_1, feed_step_1 )
        self.writer.add_summary(c_s)

        feed_step_2 = {self.a_ph: a_flatten,
                       self.x_b_ph: np.tile( batch['obs1_b'], [self.ac_per_state, 1, 1 ,1]),
                       self.x_l_ph: np.tile(batch['obs1_l'], [self.ac_per_state, 1, 1, 1]),
                       self.x2_b_ph: np.tile(batch['obs2_b'], [self.ac_per_state, 1, 1, 1]),
                       self.x2_l_ph: np.tile( batch['obs2_l'], [self.ac_per_state, 1, 1, 1])}
        opt_step_2 = [self.train_pi_op, self.a_summary]

        # update policy by max_likelihood
        _, a_s = self.sess.run( opt_step_2, feed_step_2 )
        self.writer.add_summary(a_s)

        # update target_net and old pi
        self.sess.run( self.soft_update )

    def test_agent(self, deterministic=False, n=1):
        ep_r = []
        ep_l = []
        plt.figure(figsize=(7, 4))
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
        # logger
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
                im = Image.fromarray(o2[0])
                im.save(
                    'result/success_obs/' + str(self._success_obs_item)+'.png'
                )
                self._success_obs_item +=1

            d = False if ep_len == self.max_ep_len else d

            self.replay_buffer.store(o, a, r, o2, d)

            o = o2

            if d or (ep_len == self.max_ep_len):

                if (t+1) >= 516: [self.learn() for _ in range(ep_len)]

                ep_s = self.sess.run(self.test_summary, {self.ep_ret_ph: ep_ret,
                                                           self.ep_len_ph: ep_len})
                self.writer.add_summary(ep_s, global_step=t)
                log_episode_ret.append(ep_ret)
                log_t.append(t)
                o, r, d, ep_ret, ep_len = self.env.reset(), 0, False, 0, 0

            if (t+1)%300 ==0:
                self.Save()

        np.save('result/'+self.experiment_name+'_reward.npy', (log_t, log_episode_ret), allow_pickle=True)

def Count_Variables():

    var_counts = tuple(core.count_vars(scope) for scope in
                       ['main/pi', 'main/q1', 'main/q2', 'main/v', 'main'])
    print(('\nNumber of parameters: \t pi: %d, \t ' +
           'q1: %d, \t q2: %d, \t v: %d, \t total: %d\n') % var_counts)

def R_plot(r_plot, viz, win):
    plt.cla()
    plt.plot(r_plot, color='r', label='Rank')
    plt.ylim(0,5)
    plt.xlim(0,25)
    plt.xlabel('Step')
    plt.ylabel('Rank')
    plt.title('Rank Trance')
    plt.legend(loc='upper right')
    return viz.matplot(plt, win=win)

if __name__ == '__main__':
    import hyp

    if input('删除 '+"logs/"+hyp.EXPERIMENT_NAME+' ? ')!='no':
        import shutil
        shutil.rmtree("logs/"+hyp.EXPERIMENT_NAME)

    env_fn = lambda : KukaDiverseObjectEnv(renders=False,
                         maxSteps=15,
                         blockRandom=0.3,
                         actionRepeat=200,
                         numObjects=1, dv=1.0,
                         isTest=False)

    model = sac_ES(env_fn, steps_per_epoch=15, epochs=500, gamma=0.9,
                   polyak=0.995, memory_size=10000, lr=3e-4, alpha=0.2,
                   batch_size=64, start_steps=500, max_ep_len=500,
                   experiment_name=hyp.EXPERIMENT_NAME, use_adv=hyp.ADV,
                   use_half_dete=hyp.USE_HALF_DETE,
                   use_sample_dist=hyp.USE_SAMPLE_DIST)

    model.rollout()

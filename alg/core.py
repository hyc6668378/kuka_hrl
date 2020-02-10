# coding=utf-8
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
tfd = tfp.distributions
EPS = 1e-8

def placeholder(dim=None):
    return tf.placeholder(dtype=tf.float32, shape=(None,dim) if dim else (None,))

def placeholders(*args):
    return [placeholder(dim) for dim in args]

def mlp(x, hidden_sizes=(32,), activation=tf.tanh, output_activation=None):
    for h in hidden_sizes[:-1]:
        x = tf.layers.dense(x, units=h, activation=activation)
        x = tf.contrib.layers.layer_norm(x, center=True, scale=True)
    return tf.layers.dense(x, units=hidden_sizes[-1], activation=output_activation)

def get_vars(scope):
    return [x for x in tf.global_variables() if scope in x.name]

def count_vars(scope):
    v = get_vars(scope)
    return sum([np.prod(var.shape.as_list()) for var in v])

def gaussian_likelihood(x, mu, log_std):
    pre_sum = -0.5 * (((x-mu)/(tf.exp(log_std)+EPS))**2 + 2*log_std + np.log(2*np.pi))
    return tf.reduce_sum(pre_sum, axis=1)

def clip_but_pass_gradient(x, l=-1., u=1.):
    clip_up = tf.cast(x > u, tf.float32)
    clip_low = tf.cast(x < l, tf.float32)
    return x + tf.stop_gradient((u - x)*clip_up + (l - x)*clip_low)

def _cnn(x):
    x = x / 255.0
    net = tf.layers.conv2d(x, filters=64, kernel_size=3, activation=tf.nn.relu,
                           strides=2, padding='valid', name='conv2_1')

    net = tf.layers.max_pooling2d(net, pool_size=2, strides=2)

    net = tf.layers.conv2d(net, filters=128, kernel_size=4, activation=tf.nn.relu,
                           strides=1, padding='valid', name='conv2_2')

    net = tf.layers.max_pooling2d(net, pool_size=2, strides=2)

    net = tf.layers.conv2d(net, filters=64, kernel_size=3, name='conv2_3', activation=tf.nn.relu,
                           strides=2, padding='valid')

    net_max = tf.layers.max_pooling2d(net, pool_size=2, strides=2)

    net = tf.layers.conv2d(net_max, filters=96, kernel_size=2, name='conv2_4', activation=tf.nn.relu,
                           strides=2, padding='valid')

    net = tf.layers.flatten(net, name='cnn_flatten')

    # 合起来
    net = tf.concat([net, tf.layers.flatten(net_max)], axis=1)
    return tf.layers.flatten(net)

"""
Policies
"""

LOG_STD_MAX = 2
LOG_STD_MIN = -4
kernal_Num = 10

def make_Trun_Mix_Gauss_policy(x, a, hidden_sizes, activation, output_activation, action_sp):

    act_dim = a.shape.as_list()[-1]
    net = mlp(x, list(hidden_sizes) + [ 3*act_dim*kernal_Num ], activation, output_activation)

    center = ((action_sp.high + action_sp.low) / 2).reshape(-1, 1)
    divergence = ((action_sp.high - action_sp.low) / 2).reshape(-1, 1)
    a_range = list(zip(action_sp.low, action_sp.high))  # [(low, high),(low, high),...,(low, high)]

    net_w, net_mu, net_std = tf.split(net, [act_dim * kernal_Num, act_dim * kernal_Num, act_dim * kernal_Num], 1)

    net_w = tf.reshape(net_w, [-1, act_dim, kernal_Num])  # reshape
    net_mu = tf.reshape(net_mu, [-1, act_dim, kernal_Num])  # reshape
    net_std = tf.reshape(net_mu, [-1, act_dim, kernal_Num])

    weights_w = tf.nn.softmax(net_w)  # sum = 1.  every subpro weicht
    # weights_w = tf.ones_like(net)/kernal_Num
    mu = tf.tanh(net_mu, name='tanh')

    #loc = loc_ph * [tf.linspace(low, high, kernal_num) for low, high in a_range]  #
    loc = mu * divergence + center # 去中心化
    # std = (np.array([high - low for low, high in a_range]) / (2 * (kernal_Num - 1))).reshape(-1, 1)  # 标准差为相邻两个mu的长度的 1/4
    one_spin = tf.ones([act_dim, kernal_Num], dtype=tf.float32)

    std =  tf.tanh(tf.abs(net_std)) # 正数,归一化到（0,1）
    scale = one_spin * std
    low = one_spin * np.array([low for low, high in a_range]).reshape(-1, 1)  # [[low],[low],...,[low]]
    high = one_spin * np.array([high for low, high in a_range]).reshape(-1, 1)

    return tfd.MixtureSameFamily(tfd.Categorical(probs=weights_w), tfd.TruncatedNormal(loc, scale, low, high),
                                 validate_args=True, allow_nan_stats=False)

def make_Truncate_Gauss_policy(x, a, hidden_sizes, activation, output_activation, action_sp):
    act_dim = a.shape.as_list()[-1]

    net = mlp(_cnn(x), list(hidden_sizes), activation, activation)

    net_mu = tf.layers.dense(net, act_dim, activation=tf.tanh)
    # scale
    log_std = tf.layers.dense(net, act_dim, activation=tf.tanh)  # (-1,1)
    log_std = LOG_STD_MIN + 0.5 * (LOG_STD_MAX - LOG_STD_MIN) * (log_std + 1)
    scale = tf.exp(log_std)

    # loc, low, high
    loc =  net_mu * (0.5 * (action_sp.high - action_sp.low))
    low, high = action_sp.low, action_sp.high

    return tfd.TruncatedNormal(loc, scale, low, high), tfd.TruncatedNormal(loc, 1., low, high), loc, scale

def mlp_gaussian_policy(x, a, hidden_sizes, activation, output_activation):
    act_dim = a.shape.as_list()[-1]
    net = mlp(x, list(hidden_sizes), activation, output_activation=activation)
    mu = tf.layers.dense(net, act_dim, activation=output_activation)

    """
    Because algorithm maximizes trade-off of reward and entropy,
    entropy must be unique to state---and therefore log_stds need
    to be a_ neural network output instead of a_ shared-across-states
    learnable parameter vector. But for deep Relu and other nets,
    simply sticking an activationless dense layer at the end would
    be quite bad---at the beginning of training, a_ randomly initialized
    net could produce extremely large values for the log_stds, which
    would result in some actions being either entirely deterministic
    or too random to come back to earth. Either of these introduces
    numerical instability which could break the algorithm. To 
    protect against that, we'll constrain the output range of the 
    log_stds, to lie within [LOG_STD_MIN, LOG_STD_MAX]. This is 
    slightly different from the trick used by the original authors of
    SAC---they used tf.clip_by_value instead of squashing and rescaling.
    I prefer this approach because it allows gradient propagation
    through log_std where clipping wouldn't, but I don't know if
    it makes much of a_ difference.
    """
    log_std = tf.layers.dense(net, act_dim, activation=tf.tanh)
    log_std = LOG_STD_MIN + 0.5 * (LOG_STD_MAX - LOG_STD_MIN) * (log_std + 1)

    std = tf.exp(log_std)
    pi = mu + tf.random_normal(tf.shape(mu)) * std
    logp_pi = gaussian_likelihood(pi, mu, log_std)
    return mu, pi, logp_pi

def make_Multivariate_Gauss_Policy(x, a, hidden_sizes,
                                   activation, output_activation, action_sp):
    act_dim = a.shape.as_list()[-1]
    net_Architecture = list(hidden_sizes) + [ 3*act_dim+1 ] # shape the output
    net = mlp(x, list(net_Architecture), activation, output_activation=output_activation)

    # a_range = list(zip(action_sp.low, action_sp.high))  # [(low, high),(low, high),...,(low, high)]
    net_mu, net_d, net_U, net_m = tf.split(net, [act_dim, act_dim, act_dim, 1], 1)
    net_mu = tf.tanh(net_mu)  * ((action_sp.high - action_sp.low) / 2.0)  # scale
    net_U = tf.expand_dims(net_U, -1)

    return tfd.MultivariateNormalDiagPlusLowRank(loc=net_mu, scale_diag=net_d, scale_perturb_factor=net_U,
                                          scale_perturb_diag=net_m)

def apply_squashing_func(mu, pi, logp_pi):
    mu = tf.tanh(mu)
    pi = tf.tanh(pi)
    # To avoid evil machine precision error, strictly clip 1-pi**2 to [0,1] range.
    logp_pi -= tf.reduce_sum(tf.log(clip_but_pass_gradient(1 - pi**2, l=0, u=1) + 1e-6), axis=1)
    return mu, pi, logp_pi

"""
Actor-Critics
"""
def actor_critic(x, a, activation=tf.nn.relu,
                 output_activation=None, action_space=None, hidden_size=(32, 32)):
    # policy
    with tf.variable_scope('pi'):
        dist, sample_dist, mean, std = make_Truncate_Gauss_policy(x, a, hidden_size, activation, output_activation, action_space)
        pi = tf.clip_by_value( dist.sample(), action_space.low[0], action_space.high[0])

    # with tf.variable_scope('oldpi'):
    #     old_dist, old_mean = make_Truncate_Gauss_policy(x, a_, hidden_size, activation, output_activation, action_space)

    # vfs
    vf_mlp = lambda x_, a_ : tf.squeeze(mlp(tf.concat([_cnn(x_), a_], axis=-1),
                                            list(hidden_size) + [1], activation, None), axis=1)
    with tf.variable_scope('q1'):
        q1 = vf_mlp(x, a)
    with tf.variable_scope('q1', reuse=True):
        q1_pi = vf_mlp(x, pi)
    with tf.variable_scope('q2'):
        q2 = vf_mlp(x, a)
    with tf.variable_scope('q2', reuse=True):
        q2_pi = vf_mlp(x, pi)
    with tf.variable_scope('v'):
        v = tf.squeeze(mlp( _cnn(x),
                           list(hidden_size) + [1], activation, None), axis=1)
    return dist, sample_dist, mean, pi, q1, q2, q1_pi, q2_pi, v, std
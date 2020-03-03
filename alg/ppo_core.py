import numpy as np
import tensorflow as tf
import scipy.signal
from gym.spaces import Box, Discrete
from functools import partial

import tensorflow_probability as tfp
tfd = tfp.distributions

EPS = 1e-8

def combined_shape(length, shape=None):
    if shape is None:
        return (length,)
    return (length, shape) if np.isscalar(shape) else (length, *shape)

def placeholder(dim=None):
    return tf.placeholder(dtype=tf.float32, shape=combined_shape(None,dim))

def placeholders(*args):
    return [placeholder(dim) for dim in args]

def placeholder_from_space(space):
    if isinstance(space, Box):
        return placeholder(space.shape)
    elif isinstance(space, Discrete):
        return tf.placeholder(dtype=tf.int32, shape=(None,))
    raise NotImplementedError

def placeholders_from_spaces(*args):
    return [placeholder_from_space(space) for space in args]

def mlp(x, hidden_sizes=(32,), output_activation=None):
    for h in hidden_sizes[:-1]:
        x = tf.layers.dense(x, units=h, activation=tf.nn.relu,
                            kernel_initializer=tf.orthogonal_initializer(gain=np.sqrt(2)),
                            bias_initializer=tf.constant_initializer(0.0))

    x = tf.layers.dense(x, units=hidden_sizes[-1], activation=output_activation,
                           kernel_initializer=tf.orthogonal_initializer(gain=np.sqrt(2)),
                           bias_initializer=tf.constant_initializer(0.0) )
    return x

def get_vars(scope=''):
    return [x for x in tf.trainable_variables() if scope in x.name]

def count_vars(scope=''):
    v = get_vars(scope)
    return sum([np.prod(var.shape.as_list()) for var in v])

def gaussian_likelihood(x, mu, log_std):
    pre_sum = -0.5 * (((x-mu)/(tf.exp(log_std)+EPS))**2 + 2*log_std + np.log(2*np.pi))
    return tf.reduce_sum(pre_sum, axis=1)

def discount_cumsum(x, discount):
    """
    magic from rllab for computing discounted cumulative sums of vectors.

    input:
        vector x,
        [x0,
         x1,
         x2]

    output:
        [x0 + discount * x1 + discount^2 * x2,
         x1 + discount * x2,
         x2]
    """
    return scipy.signal.lfilter([1], [1, float(-discount)], x[::-1], axis=0)[::-1]


conv2_ = partial(tf.layers.conv2d, activation=None,
                           kernel_initializer=tf.orthogonal_initializer(gain=np.sqrt(2)),
                           bias_initializer=tf.constant_initializer(0.0), padding='valid')

def _cnn(x,  name_scope='none'):
    with tf.variable_scope( name_scope ):
        max_pool = lambda net: tf.layers.max_pooling2d(net, pool_size = 2, strides = 2)

        # relu_LN = lambda net: tf.nn.relu(tf.contrib.layers.layer_norm(net))
        x = x / 255.0
        net = conv2_(x, filters=64, kernel_size=3, strides=2,  name='conv2_1')
        net = max_pool(  tf.nn.relu( net ))

        net = conv2_(net, filters=128, kernel_size=4, strides=1,  name='conv2_2')
        net = max_pool(  tf.nn.relu( net ))

        net = conv2_(net, filters=64, kernel_size=3, strides=2, name='conv2_3')
        net_max = max_pool(  tf.nn.relu( net ))

        net = conv2_(net_max, filters=96, kernel_size=2, strides=2, name='conv2_4')
        net = tf.nn.relu( net )

        # 合起来
        net = tf.concat([tf.layers.flatten(net),
                         tf.layers.flatten(net_max)], axis=1)
    return net

LOG_STD_MAX = 2
LOG_STD_MIN = -4

def mlp_Truncate_Gauss_policy(latent_s, a, hidden_sizes, output_activation, action_sp):
    act_dim = a.shape.as_list()[-1]

    latent_s = mlp(latent_s, list(hidden_sizes) + [act_dim * 2], output_activation=tf.tanh)
    # The mean and the variance both calculat from the net.
    _mu, _log = tf.split( latent_s, [act_dim, act_dim], axis=1)

    log_std = LOG_STD_MIN + 0.5 * (LOG_STD_MAX - LOG_STD_MIN) * (_log + 1)  # (LOG_STD_MIN, LOG_STD_MAX)
    scale = tf.exp(log_std)

    # loc, low, high
    loc = tf.multiply( _mu , (0.5 * (action_sp.high - action_sp.low)), name='pi_deterministic')
    low, high = action_sp.low, action_sp.high

    Truncate_Gauss = tfd.TruncatedNormal(loc, scale, low, high)

    log_Prob = lambda a_batch: tf.reduce_sum(Truncate_Gauss.log_prob( a_batch ), axis=-1, keep_dims=True) # sum the each dim_of_A

    pi = Truncate_Gauss.sample(name="pi_sample")
    logp     = log_Prob( a )
    logp_pi  = log_Prob( pi )

    return pi, logp, logp_pi


def mlp_actor_critic(o1, o2, o_low_dim_ph, f_s, a, hidden_sizes=(128, 128, 128),
                     output_activation=tf.tanh, action_space=None):
    with tf.variable_scope('pi'):
        pi_latent = tf.concat([_cnn(o1, 'o1_cnn'), _cnn(o2, 'o2_cnn'), o_low_dim_ph], axis=-1)
        pi, logp, logp_pi = mlp_Truncate_Gauss_policy( pi_latent, a,
                                       hidden_sizes, output_activation, action_space)

    with tf.variable_scope('v'):
        v = tf.squeeze(mlp( f_s, list(hidden_sizes)+[1], None), axis=1)
    return pi, logp, logp_pi, v

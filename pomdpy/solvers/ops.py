import tensorflow as tf
import numpy as np


def linear(input_, output_size, stddev=0.02, bias_start=0.0, activation_fn=None, name='linear'):
    """
    Fully connected linear layer

    :param input_:
    :param output_size:
    :param stddev:
    :param bias_start:
    :param activation_fn:
    :param name:
    :return:
    """
    shape = input_.get_shape().as_list()

    with tf.variable_scope(name):
        w = tf.get_variable('Matrix', [shape[1], output_size], tf.float32,
                            tf.random_normal_initializer(stddev=stddev))
        b = tf.get_variable('bias', [output_size],
                            initializer=tf.constant_initializer(bias_start))

        out = tf.nn.bias_add(tf.matmul(input_, w), b)

        if activation_fn is not None:
            return activation_fn(out), w, b
        else:
            return out, w, b


def simple_linear(input_, initializer=tf.constant_initializer([1.]), bias_start=0.0,
                  activation_fn=None, name='simple_linear'):
    """
    simple element-wise linear layer

    :param input_:
    :param initializer
    :param bias_start
    :param activation_fn:
    :param name:
    :return:
    """
    with tf.variable_scope(name):
        w = tf.get_variable('Matrix', input_.get_shape(), tf.float32,
                            initializer)
        b = tf.get_variable('bias', [input_.get_shape()[1]],
                            initializer=tf.constant_initializer(bias_start))
        out = tf.nn.bias_add(tf.mul(input_, w), b)

        if activation_fn is not None:
            return activation_fn(out), w, b
        else:
            return out, w, b


def select_action_tf(belief, vector_set):
    """
    Compute optimal action given a belief distribution
    :param belief: dim(belief) == dim(AlphaVector)
    :param vector_set
    :return: optimal action, V(b)
    """
    assert not len(vector_set) == 0

    max_v = tf.constant([-np.inf], tf.float32)
    best_action = tf.constant([-1])
    for av in vector_set:
        with tf.name_scope('V_b'):
            v = tf.reduce_sum(tf.mul(av.v, belief))
            best_action = tf.cond(tf.greater(v, max_v)[0], lambda: tf.constant([av.action]),
                                  lambda: best_action)
            max_v = tf.maximum(v, max_v)

    return best_action, max_v


def clipped_error(x):
    # Huber loss
    try:
        return tf.select(tf.abs(x) < 1.0, 0.5 * tf.square(x), tf.abs(x) - 0.5)
    except:
        return tf.where(tf.abs(x) < 1.0, 0.5 * tf.square(x), tf.abs(x) - 0.5)

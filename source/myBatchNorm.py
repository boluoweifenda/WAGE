import tensorflow as tf
from tensorflow.python.training import moving_averages
from tensorflow.python.framework import ops
from tensorflow.python.ops import math_ops

@tf.RegisterGradient('L1BNGrad')
def L1BNGrad(op, *grad):
  x_norm = op.outputs[0]
  shape = x_norm.get_shape().as_list()
  reduce_axis = [0, 1, 2] if len(shape) == 4 else [0]
  scale = op.inputs[1]
  grad_y = grad[0]
  grad_y_mean = math_ops.reduce_mean(grad_y, reduce_axis)
  mean = math_ops.reduce_mean(grad_y * x_norm, reduce_axis)
  grad_x = scale*(grad_y-grad_y_mean - math_ops.sign(x_norm) * mean)
  return grad_x, None


def batch_normalization(x, mean, variance, offset, scale, epsilon, L1):
  def L1BN(x):
    with tf.name_scope('L1BN'):
      with tf.get_default_graph().gradient_override_map({'Mul': 'L1BNGrad'}):
        x_offset = x - tf.stop_gradient(mean)
        y = tf.multiply(x_offset,1/(variance + epsilon))
        return y
  if L1:
    x = L1BN(x)
  else:
    x = (x - mean) * tf.rsqrt(variance + epsilon)
  if scale is not None:
    x = x * scale
  if offset is not None:
    x = x + offset
  return x

def myBatchNorm(x, center=True, scale=True, is_training=True, decay=0.9, epsilon=1e-5, L1=False, data_format='NCHW'):
  with tf.variable_scope('myBatchNorm'):
    shape = x.get_shape().as_list()
    if data_format=='NCHW' and len(shape)==4:
      x = tf.transpose(x,[0,2,3,1]) # to NHWC
    reduce_axis = [0,1,2] if len(shape) == 4 else [0]
    channel = x.get_shape().as_list()[-1]

    beta = tf.get_variable(
        'beta', channel, tf.float32,
        initializer=tf.constant_initializer(0.0, tf.float32)) if center else None
    gamma = tf.get_variable(
        'gamma', channel, tf.float32,
        initializer=tf.constant_initializer(1.0, tf.float32)) if scale else None
    moving_mean = tf.get_variable(
      'moving_mean', channel, tf.float32,
      initializer=tf.constant_initializer(0.0, tf.float32),
      trainable=False)
    moving_variance = tf.get_variable(
      'moving_std', channel, tf.float32,
      initializer=tf.constant_initializer(1.0, tf.float32),
      trainable=False)

    if is_training:
      if L1:
        mean = tf.reduce_mean(x, reduce_axis)
        variance = tf.reduce_mean(tf.abs(x-mean), reduce_axis)
      else:
        mean, variance = tf.nn.moments(x, reduce_axis, name='moments')

      update_mean = moving_averages.assign_moving_average(moving_mean, mean, decay, False)
      update_variance = moving_averages.assign_moving_average(moving_variance, variance, decay, False)
      ops.add_to_collections(ops.GraphKeys.UPDATE_OPS, update_mean)
      ops.add_to_collections(ops.GraphKeys.UPDATE_OPS, update_variance)

      # beta = beta + tf.random_normal(beta.get_shape(), mean=0, stddev=0.1)
      # gamma = gamma * tf.random_normal(gamma.get_shape(), mean=1, stddev=0.1)

      x = batch_normalization(x, mean, variance, beta, gamma, epsilon, L1)
      # x = batch_normalization(x, moving_mean, moving_variance, beta, gamma, epsilon, L1)
    else:
      x = batch_normalization(x, moving_mean, moving_variance, beta, gamma, epsilon, L1)

    if data_format == 'NCHW' and len(shape)==4:
      x = tf.transpose(x,[0,3,1,2])

  return x
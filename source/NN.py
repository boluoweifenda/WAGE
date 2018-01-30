import tensorflow as tf
import Option
from tensorflow.contrib.layers import batch_norm
import Quantize
import myInitializer


class NN(object):
  def __init__(self, X, Y, training=True, global_step=None):

    self.shapeX = X.get_shape().as_list()
    self.shapeY = Y.get_shape().as_list()

    # if data dype is not float32, we assume that there is no preprocess
    if X.dtype != tf.float32:
      X = tf.cast(X, tf.float32)
      print 'Input data dype is not float32, perform simple preprocess [0,255]->[-1,1]'
      X = X / 127.5 - 1
    else:
      print 'Input data dype is float32, we assume they are preprocessed already'

    # quantize inputs
    self.H = [X]
    self._QA(X)

    self.Y = Y

    self.lossFunc = Option.lossFunc
    self.L2 = Option.L2

    self.initializer = myInitializer.variance_scaling_initializer(
      factor=1.0, mode='FAN_IN', uniform=True)

    self.is_training = training
    self.GPU = Option.GPU

    self.W = []
    self.W_q = []
    self.W_clip_op = []
    self.W_q_op = []

  def build_graph(self):
    if Option.dataSet == 'CIFAR10':
      out = self._VGG7()
    else:
      assert False, 'None network model is defined!'

    self.out = out
    return self._loss(out, self.Y)


  def _VGG7(self):

    x = self.H[-1]

    with tf.variable_scope('U0'):
      with tf.variable_scope('C0'):
        x = self._conv(x, 3, 128)
        x = self._activation(x)
      with tf.variable_scope('C1'):
        x = self._conv(x, 3, 128)
        x = self._pool(x, 'MAX', 2, 2)
        x = self._activation(x)

    with tf.variable_scope('U1'):
      with tf.variable_scope('C0'):
        x = self._conv(x, 3, 256)
        x = self._activation(x)
      with tf.variable_scope('C1'):
        x = self._conv(x, 3, 256)
        x = self._pool(x, 'MAX', 2, 2)
        x = self._activation(x)

    with tf.variable_scope('U2'):
      with tf.variable_scope('C0'):
        x = self._conv(x, 3, 512)
        x = self._activation(x)
      with tf.variable_scope('C1'):
        x = self._conv(x, 3, 512)
        x = self._pool(x, 'MAX', 2, 2)
        x = self._activation(x)

    x = self._reshape(x)
    with tf.variable_scope('FC'):
      x = self._fc(x, 1024, name='fc0')
      x = self._activation(x)
      x = self._fc(x, self.shapeY[1], name='fc1')

    # for last layer(first layer in backpro) error input quantization
    with tf.variable_scope('last'):
      x = self._QE(x)

    return x


  def _loss(self, out, labels):
    labels = tf.cast(labels,tf.float32)

    with tf.name_scope('loss'):
      if self.lossFunc == 'SSE':
        loss = 0.5 * tf.reduce_sum(tf.square(labels - out))
      else:
        loss = self.lossFunc(labels, out)
      if self.L2 > 0:
        loss += self.L2 * self._L2()

    # error calculation
    with tf.name_scope('error'):
      # classification labels
      label = tf.argmax(labels, axis=1)

      in_top_k = 1
      error = tf.reduce_mean(tf.cast(tf.logical_not(tf.nn.in_top_k(out, label, in_top_k)), tf.float32))

    if self.is_training:
      self._totalParameters()
      with tf.name_scope('debug'):
        self.gradsH = tf.gradients(loss, self.H)
        self.gradsW = tf.gradients(loss, self.W)

    return loss, error

  def _arr(self, stride_or_ksize):
    # data format NCHW
    return [1, 1, stride_or_ksize, stride_or_ksize]

  def _QA(self, x):
    if Option.bitsA <= 16:
      x = Quantize.A(x)
      self.H.append(x)
    return x

  def _QE(self, x):
    if Option.bitsE <= 16:
      x = Quantize.E(x)
      self.H.append(x)
    return x

  def _activation(self, x):
    x = tf.nn.relu(x)
    x = self._QE(x)
    x = self._QA(x)
    return x

  def _get_variable(self, shape, name):
    with tf.name_scope(name) as scope:
      self.W.append(tf.get_variable(name=name, shape=shape, initializer=self.initializer))

      print 'W:', self.W[-1].device, scope, shape,
      if Quantize.bitsW <= 16:
        # manually clip and quantize W if needed
        self.W_q_op.append(tf.assign(self.W[-1], Quantize.Q(self.W[-1], Quantize.bitsW)))
        self.W_clip_op.append(tf.assign(self.W[-1],Quantize.C(self.W[-1],Quantize.bitsW)))

        scale = Option.W_scale[len(self.W)-1]
        print 'Scale:%d' % scale
        self.W_q.append(Quantize.W(self.W[-1], scale))
        return self.W_q[-1]
      else:
        print ''
        return self.W[-1]

  def _conv(self, x, ksize, c_out, stride=1, padding='SAME', name='conv'):
    c_in = x.get_shape().as_list()[1]
    W = self._get_variable([ksize, ksize, c_in, c_out], name)
    x = tf.nn.conv2d(x, W, self._arr(stride), padding=padding, data_format='NCHW', name=name)
    self.H.append(x)
    return x

  def _fc(self, x, c_out, name='fc'):
    c_in = x.get_shape().as_list()[1]
    W = self._get_variable([c_in, c_out], name)
    x = tf.matmul(x, W)
    self.H.append(x)
    return x

  def _pool(self, x, type, ksize, stride=1, padding='SAME'):
    if type == 'MAX':
      x = tf.nn.max_pool(x, self._arr(ksize), self._arr(stride), padding=padding, data_format='NCHW')
    elif type == 'AVG':
      x = tf.nn.avg_pool(x, self._arr(ksize), self._arr(stride), padding=padding, data_format='NCHW')
    else:
      assert False, ('Invalid pooling type:' + type)
    self.H.append(x)
    return x

  def _batch_norm(self, x, data_format='NCHW'):
    x = batch_norm(x, center=True, scale=True, is_training=self.is_training, decay=0.9, epsilon=1e-5, fused=True, data_format=data_format)
    self.H.append(x)
    return x

  def _reshape(self, x, shape=None):
    if shape == None:
      shape = reduce(lambda x, y: x * y, x.get_shape().as_list()[1:])
    x = tf.reshape(x, [-1, shape])
    self.H.append(x)
    return x

  def _totalParameters(self):
    total_parameters_fc = 0
    total_parameters_conv = 0
    for var in tf.trainable_variables():
      name_lowcase = var.op.name.lower()
      if name_lowcase.find('fc') > -1:
        total_parameters_fc += reduce(lambda x, y: x * y, var.get_shape().as_list())
      elif name_lowcase.find('conv') > -1:
        total_parameters_conv += reduce(lambda x, y: x * y, var.get_shape().as_list())
    total_parameters = total_parameters_fc + total_parameters_conv
    print 'CONV: %d FC: %d Total: %d' % (total_parameters_conv,total_parameters_fc,total_parameters)
    return total_parameters

  def _L2(self):
    decay = []
    for var in tf.trainable_variables():
      name_lowcase = var.op.name.lower()
      if name_lowcase.find('fc') > -1 or name_lowcase.find('conv') > -1:
        if Option.bitsW == 32:
          decay.append(tf.nn.l2_loss(var))
    return tf.add_n(decay)



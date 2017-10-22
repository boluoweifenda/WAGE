import numpy as np
import time
import tensorflow as tf
import NN
import Option
import Log
import getData
import Quantize
import scipy.io as sio
from tqdm import tqdm

# for single GPU quanzation
def quantizeGrads(Grad_and_vars):
  if Quantize.bitsG <= 16:
    grads = []
    for grad_and_vars in Grad_and_vars:
      grads.append( [ Quantize.G(grad_and_vars[0]) , grad_and_vars[1] ])
    return grads
  return Grad_and_vars

def main():
  # get Option
  GPU = Option.GPU
  batchSize = Option.batchSize
  pathLog = '../log/' + Option.Time + '(' + Option.Notes + ')' + '.txt'
  Log.Log(pathLog, 'w+', 1) # set log file
  print time.strftime('%Y-%m-%d %X', time.localtime()), '\n'
  print open('Option.py').read()

  # get data
  numThread = 4*len(GPU)
  assert batchSize % len(GPU) == 0, ('batchSize must be divisible by number of GPUs')

  with tf.device('/cpu:0'):
    batchTrainX,batchTrainY,batchTestX,batchTestY,numTrain,numTest,label =\
        getData.loadData(Option.dataSet,batchSize,numThread,Option.validNum)

  batchNumTrain = numTrain / batchSize
  batchNumTest = numTest / 100

  optimizer = Option.optimizer
  global_step = tf.get_variable('global_step', [], dtype=tf.int32, initializer=tf.constant_initializer(0), trainable=False)
  Net = []
  # for single GPU
  with tf.device('/gpu:%d' % GPU[0]):
    Net.append(NN.NN(batchTrainX, batchTrainY, training=True, global_step=global_step))
    lossTrainBatch, errorTrainBatch = Net[-1].build_graph()
    update_op = tf.get_collection(tf.GraphKeys.UPDATE_OPS)  # batchnorm moving average update ops
    update_op += Net[0].W_clip_op

    gradTrainBatch = quantizeGrads(optimizer.compute_gradients(lossTrainBatch))
    with tf.control_dependencies(update_op):
      train_op = optimizer.apply_gradients(gradTrainBatch, global_step=global_step)
    tf.get_variable_scope().reuse_variables()
    Net.append(NN.NN(batchTestX, batchTestY, training=False))
    _, errorTestBatch = Net[-1].build_graph()


  # Build an initialization operation to run below.
  config = tf.ConfigProto()
  config.gpu_options.allow_growth = True
  config.allow_soft_placement = True
  config.log_device_placement = False
  sess = Option.sess = tf.InteractiveSession(config=config)
  sess.run(tf.global_variables_initializer())
  saver = tf.train.Saver(max_to_keep=None)
  # Start the queue runners.
  tf.train.start_queue_runners(sess=sess)



  def getErrorTest():
    errorTest = 0.
    for i in tqdm(xrange(batchNumTest),desc = 'Test', leave=False):
      errorTest += sess.run([errorTestBatch])[0]
    errorTest /= batchNumTest
    return errorTest


  sess.run([Net[0].W_q_op])
  print "\nOptimization Start!\n"
  for epoch in xrange(1000):

    # check lr_schedule
    if len(Option.lr_schedule) / 2:
      if epoch == Option.lr_schedule[0]:
        Option.lr_schedule.pop(0)
        lr_new = Option.lr_schedule.pop(0)
        if lr_new == 0:
          print 'Optimization Ended!'
          exit(0)
        lr_old = sess.run(Option.lr)
        sess.run(Option.lr.assign(lr_new))
        print 'lr: %f -> %f' % (lr_old, lr_new)

    print 'Epoch: %03d ' % (epoch),

    lossTotal = 0.
    errorTotal = 0
    t0 = time.time()
    for batchNum in tqdm(xrange(batchNumTrain),desc = 'Epoch: %03d'%epoch, leave=False, smoothing=0.1):
      _, loss_delta, error_delta = sess.run([train_op, lossTrainBatch, errorTrainBatch])

      lossTotal += loss_delta
      errorTotal += error_delta

    lossTotal /= batchNumTrain
    errorTotal /= batchNumTrain

    print 'Loss: %.6f Train: %.4f' % (lossTotal, errorTotal),

    # get test error
    errorTest = getErrorTest()
    print 'Test: %.4f FPS: %d' % (errorTest,numTrain / (time.time() - t0)),


    if epoch == 0:
      errorTestBest = errorTest

    if errorTest < errorTestBest:
      if Option.saveModel is not None:
        saver.save(sess, Option.saveModel)
        print 'S',
    if errorTest < errorTestBest:
      errorTestBest = errorTest
      print 'BEST',

    print ''


if __name__ == '__main__':
  main()


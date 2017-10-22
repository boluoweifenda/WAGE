import time
import tensorflow as tf

Time = time.strftime('%Y-%m-%d', time.localtime())
# Notes = 'vgg7 2888'
Notes = 'lenet5 2888'

GPU = [1]
batchSize = 128
validNum = 0

dataSet = 'MNIST'  # 'MNIST','SVHN','CIFAR10', 'ILSVRC2012'

loadModel = None
# loadModel = '../model/' + '2017-10-27' + '(' + 'vgg7 2888' + ')' + '.tf'
saveModel = None
saveModel = '../model/' + Time + '(' + Notes + ')' + '.tf'

bitsW = 2  # bit width of weights
bitsA = 8  # bit width of activations
bitsG = 8  # bit width of gradients
bitsE = 8  # bit width of errors

bitsR = 16  # bit width of randomizer

use_batch_norm = False
lr = tf.Variable(initial_value=0., trainable=False, name='lr', dtype=tf.float32)
lr_schedule = [0, 8, 200, 1,250,1./8,300,0]
L2 = 0

lossFunc = 'SSE'
optimizer = tf.train.GradientDescentOptimizer(1)  # lr is controlled in Quantize.G

# shared variables, defined by other files
seed = None
sess = None
W_scale = []

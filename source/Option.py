import time
import tensorflow as tf

debug = False
Time = time.strftime('%Y-%m-%d', time.localtime())
# Notes = 'vgg7 2888'
Notes = 'temp'

GPU = [0]
batchSize = 128

dataSet = 'CIFAR10'

loadModel = None
# loadModel = '../model/' + '2017-12-06' + '(' + 'vgg7 2888' + ')' + '.tf'
saveModel = None
# saveModel = '../model/' + Time + '(' + Notes + ')' + '.tf'

bitsW = 2  # bit width of we ights
bitsA = 8  # bit width of activations
bitsG = 8  # bit width of gradients
bitsE = 8  # bit width of errors

bitsR = 16  # bit width of randomizer

lr = tf.Variable(initial_value=0., trainable=False, name='lr', dtype=tf.float32)
lr_schedule = [0, 8, 200, 1,250,1./8,300,0]

L2 = 0

lossFunc = 'SSE'
# lossFunc = tf.losses.softmax_cross_entropy
optimizer = tf.train.GradientDescentOptimizer(1)  # lr is controlled in Quantize.G
# optimizer = tf.train.MomentumOptimizer(lr, 0.9, use_nesterov=True)

# shared variables, defined by other files
seed = None
sess = None
W_scale = []

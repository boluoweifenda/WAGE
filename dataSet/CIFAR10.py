import cPickle
import numpy as np

trainX = np.zeros([50000,32,32,3], dtype=np.uint8)
trainY = np.zeros([50000,10], dtype=np.uint8)
testX  = np.zeros([10000,32,32,3], dtype=np.uint8)
testY  = np.zeros([10000,10], dtype=np.uint8)
label = ['airplane', 'automoblie','bird','cat','deer', 'dog','frog','horse','ship','truck']

trainFileName = ['cifar-10-batches-py/data_batch_1',
                 'cifar-10-batches-py/data_batch_2',
                 'cifar-10-batches-py/data_batch_3',
                 'cifar-10-batches-py/data_batch_4',
                 'cifar-10-batches-py/data_batch_5']

testFileName = ['cifar-10-batches-py/test_batch']

index = 0
for name in trainFileName:
    f = open(name,'rb')
    dict = cPickle.load(f)
    f.close()
    trainX[index:index + 10000, ...] = dict['data'].reshape([10000, 3, 32, 32]).transpose([0, 2, 3, 1])
    trainY[np.arange(index,index+10000), dict['labels']] = 1
    index += 10000

index = 0
for name in testFileName:
    f = open(name, 'rb')
    dict = cPickle.load(f)
    f.close()
    testX[index:index + 10000, ...] = dict['data'].reshape([10000, 3, 32, 32]).transpose([0, 2, 3, 1])
    testY[np.arange(index,index+10000), dict['labels']] = 1
    index += 10000

np.savez('CIFAR10.npz',trainX=trainX,trainY=trainY,testX=testX,testY=testY,label=label)


import scipy.io as sio
import numpy as np

label = ['0', '1','2','3','4', '5','6','7','8','9']

train = sio.loadmat('train_32x32.mat')
test = sio.loadmat('test_32x32.mat')
extra = sio.loadmat('extra_32x32.mat')

trainX = np.transpose(train['X'],(3,0,1,2))
trainY = np.zeros([trainX.shape[0],10], dtype=np.uint8)
for i in xrange(trainX.shape[0]):
    trainY[i,train['y'][i]%10] = 1

testX = np.transpose(test['X'],(3,0,1,2))
testY = np.zeros([testX.shape[0],10], dtype=np.uint8)
for i in xrange(testX.shape[0]):
    testY[i,test['y'][i]%10] = 1

extraX = np.transpose(extra['X'],(3,0,1,2))
extraY = np.zeros([extraX.shape[0],10], dtype=np.uint8)
for i in xrange(extraX.shape[0]):
    extraY[i,extra['y'][i]%10] = 1

trainX = np.concatenate((trainX,extraX),axis=0)
trainY = np.concatenate((trainY,extraY),axis=0)

np.savez('SVHN.npz',trainX=trainX,trainY=trainY,testX=testX,testY=testY,label=label)
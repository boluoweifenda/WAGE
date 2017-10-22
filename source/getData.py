import tensorflow as tf
import numpy as np
import Option



def preprocess(x, train=False):
  dataSet = Option.dataSet
  if dataSet == 'MNIST':
    pass
  elif dataSet == 'SVHN':
    if train:
      x = tf.image.resize_image_with_crop_or_pad(x, 40, 40)
      x = tf.random_crop(x, [32, 32, 3])
  elif dataSet == 'CIFAR10':
    if train:
      x = tf.image.resize_image_with_crop_or_pad(x, 40, 40)
      x = tf.random_crop(x, [32, 32, 3])
      x = tf.image.random_flip_left_right(x)
  else:
    print 'Unkown dataset',dataSet,'no preprocess'
  x = tf.transpose(x, [2, 0, 1])# from HWC to CHW
  return x



def loadData(dataSet,batchSize,numThread,validNum=0):
  if dataSet != 'ILSVRC2012':
    pathNPZ = '../dataSet/' + dataSet + '.npz'
    numpyTrainX, numpyTrainY, numpyTestX, numpyTestY, label = loadNPZ(pathNPZ, validNum)
    numTrain = numpyTrainX.shape[0]
    numTest = numpyTestX.shape[0]

    trainX, trainY = data2Queue(numpyTrainX, numpyTrainY, batchSize,numThread, True,True)
    testX, testY = data2Queue(numpyTestX, numpyTestY, 100, 1, False,False)

  else:
    # pathData = '/mnt/HDD1/dataSet/' + dataSet + '/' + 'TFRecord(resized)/'
    pathData = '/home/wushuang/Desktop/TFRecord(resized)/'
    synsetFile = '../dataSet/imagenet_lsvrc_2015_synsets.txt'
    metaFile = '../dataSet/imagenet_metadata.txt'
    label = getImageNetLabel(synsetFile, metaFile)

    trainX, trainY = loadTFRecord(pathData+'train-*',batchSize,True)
    testX, testY = loadTFRecord(pathData+'validation-*', 100,False)

    numTrain = 1281167
    numTest = 50000

  return trainX,trainY,testX,testY,numTrain,numTest,label

def getImageNetLabel(synsetFile,metaFile):
  label = []
  synset = tf.gfile.FastGFile(synsetFile, 'r').readlines()
  meta = tf.gfile.FastGFile(metaFile, 'r').readlines()
  synset_to_human = {}
  for i in meta:
    if i:
      parts = i.strip().split('\t')
      assert len(parts) == 2
      synset_to_human[parts[0]] = parts[1]

  for i in synset:
    meta = synset_to_human[i.strip('\n')]
    label.append(meta)

  assert len(label) == 1000
  return label


# get dataset from NPZ files
def loadNPZ(pathNPZ, validNum=0):
  data = np.load(pathNPZ)

  trainX = data['trainX']
  trainY = data['trainY']

  if validNum > 100:
    testX = trainX[-validNum:]
    testY = trainY[-validNum:]
    trainX = trainX[0:-validNum]
    trainY = trainY[0:-validNum]
  else:
    testX = data['testX']
    testY = data['testY']

  label = data['label']
  return trainX, trainY, testX, testY, label

# get ImageNet tensorflow record files
def loadTFRecord(pattern,batch_size,train=False):

  # ImageNet preprocess
  def image_preprocessing(image_buffer, train):
    image = tf.image.decode_jpeg(image_buffer, channels=3)
    if train:
      image = tf.random_crop(image, [224, 224, 3])
      image = tf.image.random_flip_left_right(image)
    else:
      image = tf.image.central_crop(image, 0.875)
    image = tf.transpose(image, [2, 0, 1])
    return image

  def parse_example_proto(example_serialized):
    feature_map = {
      'image/encoded': tf.FixedLenFeature([], dtype=tf.string,
                                          default_value=''),
      'image/class/label': tf.FixedLenFeature([1], dtype=tf.int64,
                                              default_value=-1),
      'image/class/text': tf.FixedLenFeature([], dtype=tf.string,
                                             default_value=''),
    }
    sparse_float32 = tf.VarLenFeature(dtype=tf.float32)
    # Sparse features in Example proto.
    feature_map.update(
      {k: sparse_float32 for k in ['image/object/bbox/xmin',
                                   'image/object/bbox/ymin',
                                   'image/object/bbox/xmax',
                                   'image/object/bbox/ymax']})

    features = tf.parse_single_example(example_serialized, feature_map)
    label = tf.cast(features['image/class/label'], dtype=tf.int32)

    return features['image/encoded'], label

  data_files = tf.gfile.Glob(pattern)
  if not data_files:
    print 'No files found'
  # Create filename_queue
  if train:
    filename_queue = tf.train.string_input_producer(data_files, shuffle=True, capacity=16)
  else:
    filename_queue = tf.train.string_input_producer(data_files, shuffle=False, capacity=1)


  num_readers = 4
  num_preprocess_threads = 8 * len(Option.GPU)
  examples_per_shard = 1024
  min_queue_examples = examples_per_shard * 16

  examples_queue = tf.FIFOQueue(capacity=examples_per_shard + 3 * batch_size, dtypes=[tf.string])
  # Create multiple readers to populate the queue of examples.
  if num_readers > 1:
    enqueue_ops = []
    for _ in range(num_readers):
      reader = tf.TFRecordReader()
      _, value = reader.read(filename_queue)
      enqueue_ops.append(examples_queue.enqueue([value]))

    tf.train.queue_runner.add_queue_runner(
        tf.train.queue_runner.QueueRunner(examples_queue, enqueue_ops))
    example_serialized = examples_queue.dequeue()
  else:
    reader = tf.TFRecordReader()
    _, example_serialized = reader.read(filename_queue)

  image_buffer, label_index = parse_example_proto(example_serialized)
  image = image_preprocessing(image_buffer,True)
  label_index = tf.one_hot(label_index[0],1000,dtype=tf.int32)

  if train:
    images, label_index_batch = tf.train.shuffle_batch(
        [image,label_index],
        batch_size=batch_size,
        num_threads= num_preprocess_threads,
        capacity = 2*min_queue_examples,
        min_after_dequeue= min_queue_examples)
  else:
    images, label_index_batch = tf.train.batch(
      [image, label_index],
      batch_size=batch_size,
      num_threads=num_preprocess_threads,
      capacity=2* min_queue_examples
    )

  return images, label_index_batch



def data2Queue(dataX, dataY, batchSize, numThreads, shuffle=False, isTraining=True, seed=None):
  q = tf.FIFOQueue(capacity=dataX.shape[0], dtypes=[dataX.dtype, dataY.dtype],shapes=[dataX.shape[1:],dataY.shape[1:]])
  enqueue_op = q.enqueue_many([dataX, dataY])
  sampleX, sampleY = q.dequeue()
  qRunner = tf.train.QueueRunner(q, [enqueue_op])
  tf.train.add_queue_runner(qRunner)

  sampleX_ = preprocess(sampleX, isTraining)

  if shuffle:
    batchX, batchY = tf.train.shuffle_batch([sampleX_, sampleY],
                                            batch_size=batchSize,
                                            num_threads=numThreads, capacity=dataX.shape[0],
                                            min_after_dequeue=dataX.shape[0] / 2,
                                            seed=seed)
  else:
    batchX, batchY = tf.train.batch([sampleX_, sampleY],
                                    batch_size=batchSize,
                                    num_threads=numThreads,
                                    capacity=dataX.shape[0])

  return batchX, batchY



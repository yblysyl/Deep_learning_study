import tensorflow as tf
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import numpy as np
import os
import pickle
import matplotlib.pyplot as plt  
from matplotlib.pyplot import imshow ##%matplotlib inline


CIFARDIR ='../cifar-10-python/cifar-10-batches-py/'
print(os.listdir(CIFARDIR))

def load_data(file):##read data from data file
    with open(file, 'rb') as fo:
        data = pickle.load(fo, encoding='bytes')
    return data[b'data'], data[b'labels']

class CifarData:
    def __init__ (self, filenames, need_shuffle):
        all_data =[]
        all_labels =[]
        for filename in filenames:
            data, labels = load_data(filename)
            for item, label in zip(data, labels):
                if label in [0,1]:
                    all_data.append(item)
                    all_labels.append(label)
        self._data = np.vstack(all_data)
        self._labels = np.hstack(all_labels)
        print(self._data.shape)
        print(self._labels .shape)
        self._num_examples = self._data.shape[0]
        self._need_shuffle = need_shuffle
        self._indicator = 0
        if self._need_shuffle:
            self._shuffle_data()
    def _shuffle_data(self):
        #[0,1,2,3,4,5] -> [5,3,2,4,0,11
        p = np.random.permutation(self._num_examples)
        self._data = self._data[p]
        self._labels = self._labels[p]

    def next_batch(self, batch_size):
        end_indicator = self._indicator + batch_size
        if end_indicator > self._num_examples:
            if self._need_shuffle:
                self._shuffle_data()
                self._indicator = 0
                end_indicator = batch_size
            else:
                raise Exception("have no more examples")
        if end_indicator > self._num_examples:
            raise Exception("batch size is larger than all examples")
        batch_data = self._data[self._indicator: end_indicator]
        batch_labels = self._labels[self._indicator: end_indicator]
        self._indicator = end_indicator
        return batch_data, batch_labels
train_filenames=[os.path.join(CIFARDIR, 'data_batch_%d' % i) for i in range(1,6)]
test_filenames = [os .path.join(CIFARDIR,'test_batch')]
train_data = CifarData(train_filenames, True)

'''    以下为tf1的代码,在现在的tf2中被抛弃'''
x=tf.placeholder(tf.float32,[None,3072]) 

#[None]
y=tf.placeholder(tf.int64,[None])

## (3072*1)
w = tf.get_variable('w',[x.get_shape()[-1], 1],initializer=tf.random_normal_initializer(0, 1))

## (1,)
b = tf.get_variable('b',[1],initializer=tf.constant_initializer(0.0))

##[None，3072] * [3072， 1] = [None， 1]
y_=tf.matmul(x,w) + b

##[None， 1]
p_y_1 = tf.nn.sigmoid(y_)

## [Ndne， 1] 
y_reshaped = tf.reshape(y, (-1, 1))  ##转换shape

y_reshaped_float = tf.cast(y_reshaped,tf.float32)



##计算模型和实际得差值 
loss = tf.reduce_mean(tf.square(y_reshaped_float - p_y_1))



###计算准确率
#boo1
predict = p_y_1>0.5
#[1.0.1.1.1.0.0.0]
correct_prediction = tf.equal(tf.cast(predict, tf.int64), y_reshaped)
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float64))

with tf.name_scope('train_op'):
    train_op = tf.train.AdamOptimizer(1e-3).minimize(loss) ##梯度下降算法



init = tf.global_variable_initializer()
with tf.Session() as sess:
    sess.run([loss, accuracy, train_op], feed_dict={x,y}) ##train_op 有表示训练

import tensorflow as tf
import numpy as np
import pickle
import json
import os

#加载和预处理图像数据
def randomize(dataset, labels):
    # Randomly permute a sequence
    permutation=np.random.permutation(labels.shape[0])
    shuffled_dataset= dataset[permutation,:,:]
    shuffled_labels= labels[permutation]
    return shuffled_dataset, shuffled_labels

def one_hot_encode(np_array):
    #数据预处理之独热编码（One-Hot Encoding）
    #独热编码即 One-Hot 编码，又称一位有效编码，其方法是使用N位状态寄存器来对N个状态进行编码，每个状态都由他独立的寄存器位，并且在任意时候，其中只有一位有效。
    #解决了分类器不好处理属性数据的问题 在一定程度上也起到了扩充特征的作用
    return (np.arange(10) == np_array[:, None]).astype(np.float32)#Return evenly spaced values within a given interval.

def reformat_data(dataset, labels, image_width, image_height, image_depth):
    np_dataset_ = np.array([np.array(image_data).reshape(image_width,image_height,image_depth) for image_data in dataset])
    np_labels_ = one_hot_encode(np.array(labels, dtype=np.float32))
    np_dataset, np_labels = randomize(np_dataset_, np_labels_)
    return np_dataset, np_labels

def flatten_tf_array(array):
    shape=array.get_shape().as_list()
    return tf.reshape(array, [shape[0], shape[1]*shape[2]*shape[3]])

def accuracy(predictions, labels):
    #Returns the indices of the maximum values along an axis
    return (100.0 * np.sum(np.argmax(predictions, 1)==np.argmax(labels, 1))/predictions.shape[0])


cifar10_folder = './data/cifar10/'
train_datasets = ['data_batch_1', 'data_batch_2', 'data_batch_3', 'data_batch_4', 'data_batch_5', ]
test_dataset = ['test_batch']
c10_image_height = 32
c10_image_width = 32
c10_image_depth = 3
c10_num_labels = 10
c10_image_size = 32  # Ahmet Taspinar的代码缺少了这一语句

with open(cifar10_folder + test_dataset[0], 'rb') as f0:
    c10_test_dict = pickle.load(f0, encoding='bytes')

c10_test_dataset, c10_test_labels = c10_test_dict[b'data'], c10_test_dict[b'labels']
test_dataset_cifar10, test_labels_cifar10 = reformat_data(c10_test_dataset, c10_test_labels, c10_image_size,
                                                          c10_image_size, c10_image_depth)

c10_train_dataset, c10_train_labels = [], []
for train_dataset in train_datasets:
    with open(cifar10_folder + train_dataset, 'rb') as f0:
        c10_train_dict = pickle.load(f0, encoding='bytes')
        c10_train_dataset_, c10_train_labels_ = c10_train_dict[b'data'], c10_train_dict[b'labels']

        c10_train_dataset.append(c10_train_dataset_)
        c10_train_labels += c10_train_labels_

c10_train_dataset = np.concatenate(c10_train_dataset, axis=0)
train_dataset_cifar10, train_labels_cifar10 = reformat_data(c10_train_dataset, c10_train_labels, c10_image_size,
                                                            c10_image_size, c10_image_depth)
del c10_train_dataset
del c10_train_labels

print("训练集包含以下标签: {}".format(np.unique(c10_train_dict[b'labels'])))
print('训练集维度', train_dataset_cifar10.shape, train_labels_cifar10.shape)
print('测试集维度', test_dataset_cifar10.shape, test_labels_cifar10.shape)

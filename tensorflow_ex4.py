import tensorflow as tf
import numpy as np

#加载和预处理图像数据
def randomize(dataset, labels):
    permutation=np.random.permutation(labels.shape[0])#置换
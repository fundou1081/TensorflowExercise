import tenssorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

#mnist数据集相关的常数
INPUT_NODE = 784
OUTPUT_NODE = 10

#神经网络参数
LAYER1_NODE = 500  #隐藏层节点数，只有一个隐藏层，有500节点
BATCH_SIZE = 100  # 小 随机梯度下降 大 梯度下降
LEARNING_RATE_BASE = 0.8  #基础学习率
LEARNING_RATE_DECAY = 0.99  #学习率的衰减率
REGULARIZATION_RATE = 0.0001  #描述模型复杂度的正则化项在损失函数中的系数
TRAINING_STEPS = 30000  #训练论数
MOVING_AVERAGE_DECAY = 0.99  #滑动平均衰减率


#计算前向传播结果
def inference(input_tensor, avg_class, weights1, biases1, weights2, biases2):
    #当没有提供滑动平均类时，直接使用参数当前的取值
    if avg_class == None:
        layer1 = tf.nn.relu(tf.matmul(input_tensor, weights1) + biases1)
        return tf.matmul(layer1, weights2) + biases2
    else:
        layer1=tf.nn.relu(tf.matmul(input_tensor,avg_class.average(weights1))+avg_class.average(biases1))
        return tf.matmul(layer1,avg_class.average(weights2))+avg_class.average(biases2)

def train(mnist):
    x=tf.placeholder(tf.float32,[None,INPUT_NODE], name='x-input')
    y_ =tf.placeholder(tf.float32,[None,OUTPUT_NODE], name='y-input')

    weights1=tf.Variable(tf.truncated_normal([INPUT_NODE,LAYER1_NODE],stddev=0.1))
    biases1=tf.Variable(tf.constant(0.1,shape=[LAYER1_NODE]))

    weights2=tf.Variable(tf.truncated_normal([LAYER1_NODE,OUTPUT_NODE],stddev=0.1))
    biases2=tf.Variable(tf.constant(0.1,shape=[OUTPUT_NODE]))

    y= inference(x,None,weights1,biases1,weights2,biases2)
    global_step=tf.Variable(0,trainable=False)

    variable_averages=tf.train.ExponentialMovingAve
    
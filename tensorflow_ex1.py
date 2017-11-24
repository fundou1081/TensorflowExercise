#follow https://www.jiqizhixin.com/articles/2017-08-29-14

import tensorflow as tf
hello = tf.constant('Hello tf')
sess=tf.Session()
print(sess.run(hello))

##############
import numpy as np
a = tf.constant(2, tf.int16)
b = tf.constant(4, tf.float32)
c = tf.constant(8, tf.float32)

d = tf.Variable(2, tf.int16)
e = tf.Variable(4, tf.float32)
f = tf.Variable(8, tf.float32)

g = tf.constant(np.zeros(shape=(2,2), dtype=np.float32)) #可以正常声明变量

h = tf.zeros([11], tf.int16)
i = tf.ones([2,2], tf.float32)
j = tf.zeros([1000,4,3], tf.float64)

k = tf.Variable(tf.zeros([2,2], tf.float32))
l = tf.Variable(tf.zeros([5,6,5], tf.float32))


################
a=tf.constant(2,tf.int16)
b=tf.constant(4,tf.float32)

graph= tf.Graph()
with graph.as_default():
    a=tf.Variable(8,tf.float32)
    b=tf.Variable(tf.zeros([2,2],tf.float32))

with tf.Session(graph=graph) as session:
    tf.global_variables_initializer().run()
    print(f)
    print(session.run(a))
    print(session.run(b))


w1=tf.Variable(tf.random_normal([2,3],stddev=1,seed=1))
#tf.truncated_normal() 函数，即截断正态分布随机数，它只保留 [mean-2*stddev,mean+2*stddev] 范围内
w2=tf.Variable(tf.truncated_normal([2,3],stddev=1,seed=1))

weights = tf.Variable(tf.truncated_normal([256*256,10]))
biases=tf.Variable(tf.zeros([10]))
print(weights.get_shape().as_list())
print(biases.get_shape().as_list())

w1=tf.Variable(tf.random_normal([1,2],stddev=1,seed=1))
#因为需要重复输入x，而每建一个x就会生成一个结点，计算图的效率会低。所以使用占位符
x=tf.placeholder(tf.float32,shape=(1,2))
x1=tf.constant([0.7, 0.9])
a=x+w1
b=x1+w1
sess=tf.Session()
sess.run(tf.global_variables_initializer())
#运行y时将占位符填上，feed_dict为字典，变量名不可变
y_1=sess.run(a, feed_dict={x:[[0.7, 0.9]]})
y_2=sess.run(b)
print(y_1)
print(y_2)
sess.close
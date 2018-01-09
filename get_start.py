# -*- coding: utf-8 -*-
"""
Created on Tue Jan  9 16:47:27 2018

@author: 126453

http://www.tensorfly.cn/tfdoc/get_started/introduction.html

"""
import tensorflow as tf
import numpy as np

# 使用 NumPy 生成假数据(phony data), 总共 100 个点.
x_data = np.float32(np.random.rand(2, 100))  # 创建随机 2x100 矩阵 2行 100列
y_data = np.dot([0.1, 0.2], x_data) + 0.3  # 矩阵乘法

# 构造一个线性模型
b = tf.Variable(tf.zeros([1]))
W = tf.Variable(tf.random_uniform([1, 2], -1.0, 1.0))
y = tf.matmul(W, x_data) + b  # 矩阵a 乘 矩阵b

# 最小化方差
loss = tf.reduce_mean(tf.square(y - y_data))
optimizer = tf.train.GradientDescentOptimizer(0.5)  # 梯度下降算法优化器 学习率
train = optimizer.minimize(loss)

# 初始化变量
init = tf.initialize_all_variables()

#启动图(graph)
sess=tf.Session()
sess.run(init)

#拟合平面
for step in range(0,201):
    sess.run(train)
    if step%20==0:
        print(step,sess.run(W),sess.run(b))
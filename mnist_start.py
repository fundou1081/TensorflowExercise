import input_data
import tensorflow as tf

mnist = input_data.read_data_sets("mnist/", one_hot=True)

x = tf.placeholder("float", [None, 784])

W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros(10))
y=tf.nn.softmax(tf.matmul(x,W)+b)


#计算交叉熵
y_=tf.placeholder("float", [None, 10])
cross_entropy = -tf.reduce_sum(y_*tf.log(y))

train_step=tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)

init = tf.global_variables_initializer()

sess=tf.Session()
sess.run(init)

for i in range(1000):
    batch_xs , batch_ys = mnist.train.next_batch(100)
    sess.run(train_step, feed_dict={x: batch_xs, y_:batch_ys})

# tf.argmax 给出某个tensor对象在某一维上的其数据最大值所在的索引值
# 由于标签向量是由0,1组成，因此最大值1所在的索引位置就是类别标签
correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))

#布尔值转换成浮点数，然后取平均值
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

print(sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels}))
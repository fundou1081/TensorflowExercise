import tensorflow as tf

x= tf.random_normal([10,10])
y=tf.random_normal([10,10])
z=tf.matmul(x,y)

sess=tf.Session()
z_val=sess.run(z)

print(z_val)
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 19 01:00:46 2017

@author: 方豆
"""

import tensorflow as tf

a=tf.constant([1.0, 2.0 ], name="a")
b=tf.constant([2.0, 3.0], name="b")

result = a+b

sess = tf.Session()
print( sess.run(result) )
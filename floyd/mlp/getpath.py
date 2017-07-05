import os
import tensorflow as tf
x=tf.constant("The current path is",os.getcwd())
init = tf.initialize_all_variables()
sess = tf.Session()
sess.run(init)
result = sess.run(x)
print result
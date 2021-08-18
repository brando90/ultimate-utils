import tensorflow as tf

#print hello world
hello = tf.constant('Hello, TensorFlow!')
sess = tf.Session()
print(sess.run(hello))

# do a matrix addition in tensorflow
a = tf.constant(10)
b = tf.constant(32)
print(sess.run(a + b))

import tensorflow as tf

sess= tf.Session()

a = tf.constant([[1, 2, 3, 5, 4],
                 [4, 5, 6, 3, 7],
                 [7, 8, 9, 2, 1],
                 [1, 6, 7, 9, 8],
                 [9, 3, 5, 7, 4]], name='a')
b = tf.constant( 4, name='b')
print (sess.run(tf.shape(b)))
add_op = a + b

"adding scalar"
print(sess.run(add_op))


b = tf.constant([4, 5, 21, 25, 21], name='b')
print (sess.run(tf.shape(b)))
add_op = a + b

"adding array"
print(sess.run(add_op))


b = tf.constant([[5, 21, 4, 28, 25], [28, 25, 21, 4, 5], [31, 3, 2017, 2013, 1988],[9, 3, 5, 7,9], [1982, 1979, 2013, 1988, 2017]], name='b')
print (sess.run(tf.shape(b)))
add_op = a + b

"adding matrix"
print(sess.run(add_op))

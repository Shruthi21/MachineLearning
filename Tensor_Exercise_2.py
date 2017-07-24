import tensorflow as tf

sess= tf.Session()

a = tf.constant([[1, 2, 3, 5],
                 [4, 5, 6, 3],
                 [7, 8, 9, 2],
                 [1, 6, 7, 9]], name='a')
b = tf.constant( 4, name='b')
print (sess.run(tf.shape(b)))
add_op = a + b

"adding scalar"
print(sess.run(add_op))


b = tf.constant([4, 5, 21, 25], name='b')
print (sess.run(tf.shape(b)))
add_op = a + b

"adding array"
print(sess.run(add_op))


b = tf.constant([[5, 21, 4, 28], [28, 25, 21, 4], [31, 3, 2017, 5],[9, 3, 5, 7]], name='b')
print (sess.run(tf.shape(b)))
add_op = a + b

"adding matrix"
print(sess.run(add_op))

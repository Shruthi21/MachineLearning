import tensorflow as tf

a = tf.constant([2])
b = tf.constant([3])

c = tf.add(a,b)
session=tf.Session()

print(session.run(c))
session.close()

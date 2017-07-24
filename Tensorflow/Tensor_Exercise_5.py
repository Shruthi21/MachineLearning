import tensorflow as tf
import numpy as np

data = np.random.randint(1000, size=10000)
print(data)
x = tf.placeholder(tf.float32)
y = (5 * tf.square(x)) - (3 * x) + 15

with tf.Session() as session:
	print(session.run(y, {x:data}))
	print(session.run(y, {x:[1,2,3,4]}))

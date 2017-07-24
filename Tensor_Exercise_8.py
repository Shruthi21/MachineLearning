import tensorflow as tf

sess = tf.Session()
W=tf.Variable([.3],tf.float32)
b=tf.Variable([-.3],tf.float32)
x = tf.placeholder(tf.float32)
linear_model = W * x + b

init = tf.global_variables_initializer()
sess.run(init)
print ('Linear model for x value = [1,2,3,4]:')

print(sess.run(linear_model, {x:[1,2,3,4]}))


y = tf.placeholder(tf.float32)
squared_deltas = tf.square(linear_model - y)


loss = tf.reduce_sum(squared_deltas)


print ('Loss for Linear model for x  = [1,2,3,4]: and y = [0,-1,-2,-3]')
print(sess.run(loss, {x:[1,2,3,4], y:[0,-1,-2,-3]}))



fixW = tf.assign(W,[-1.])
fixb = tf.assign(b,[1.])
sess.run([fixW,fixb])
print ('Loss for Linear model for x  = [1,2,3,4]: and y = [0,-1,-2,-3] for different W & b')
print(sess.run(loss,{x:[1,2,3,4],y:[0,-1,-2,-3]}))



optimizer = tf.train.GradientDescentOptimizer(0.01)
train = optimizer.minimize(loss)


sess.run(init) # reset values to incorrect defaults.
for i in range(1000):
  sess.run(train, {x:[1,2,3,4], y:[0,-1,-2,-3]})

print(sess.run([W, b]))

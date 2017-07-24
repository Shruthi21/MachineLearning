import tensorflow as tf

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

print(mnist)

x = tf.placeholder(tf.float32, [None, 784]) # input image
y = tf.placeholder(tf.float32, [None, 10])  # input label

W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))
print(x,y,W,b)

prediction = tf.nn.softmax(tf.matmul(x, W) + b)
print('Prediction')
print(prediction)

cross_entropy = tf.reduce_mean(-tf.reduce_sum(y * tf.log(prediction), reduction_indices=[1]))
print('cross entropy')
print(cross_entropy)

train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)
print('trsin step')
print(train_step)

correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(prediction, 1))
print ('correct prediction')
print(correct_prediction)

accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
print('accuracy')
print(accuracy)

for i in range(1000):
    with tf.Session() as sess:
        tf.global_variables_initializer().run()
        batch_xs, batch_ys = mnist.train.next_batch(100)
        print(batch_xs,batch_ys)
        sess.run(train_step, feed_dict={x: batch_xs, y: batch_ys})
        print(sess.run(accuracy, feed_dict={x: mnist.test.images, y: mnist.test.labels}))

import tensorflow as tf

a = tf.placeholder(tf.float32)
b = tf.placeholder(tf.float32)
adder_node = a + b

print(a,b)
print(adder_node)

sess= tf.Session()
print(sess.run(adder_node, {a:4.0, b:5.0}))
print(sess.run(adder_node,{a:[1,3],b:[2,4]}))
add_and_triple = adder_node * 3

print(add_and_triple )
print(sess.run(add_and_triple, {a:3, b:4.5}))

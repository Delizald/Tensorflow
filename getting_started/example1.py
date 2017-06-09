#!/usr/python
import tensorflow as tf

#node1 = tf.constant(3.0, tf.float32)
#node2 = tf.constant(4.0) #implicit float32
#print(node1,node2)

##Sessions
sess = tf.Session()
#print(sess.run([node1, node2]))

##Placeholder
#A placeholder is a promise to provide a value later
a = tf.placeholder(tf.float32)
b = tf.placeholder(tf.float32)
adder_node = a + b #tf.add(a,b)
print(sess.run(adder_node, {a:3,b:4.5}))
print(sess.run(adder_node, {a: [1,3], b: [2, 4]}))



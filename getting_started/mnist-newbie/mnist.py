Import tensorflow as tf

#placeholder
x = tf.placeholder(tf.float32, [None, 784])
#weight
W = tf.Variable(tf.zeros([784, 10]))
#bias
b = tf.Variable(tf.zeros([10]))

#let's define our model
y = tf.nn.softmax(tf.matmul(x, W) + b)

#implementing cross-entropy
y_ = tf.placeholder(tf.float32, [None, 10])
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))

#training using back-propagation
train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

#launch the model in an interactive session
sess = tf.InteractiveSession()
tf.global_variables_initializer().run()

#let's train 1000
for _ in range(1000):
  batch_xs, batch_ys = mnist.train.next_batch(100)
  sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})

#Evaluating our model

correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
print(sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels}))


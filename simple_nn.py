import tensorflow as tf 
import glob
import os

# making data (data in ./data_thchs30/dev)
os.chdir(os.path.abspath("./data_thchs30/dev"))
sounds = glob.glob(os.path.abspath("*.wav"))
labels = []
for s in sounds:
  s = s + ".trn"
  with open(s, encoding = "utf8") as f:
    name = f.readline()
    label = name[:-5] + ".zh"
    labels.append(os.path.abspath(label))

sounds_queue = tf.train.string_input_producer(sounds)
labels_queue = tf.train.string_input_producer(labels)

dataset = (sounds_queue, labels_queue)
# Parameters
learning_rate = 0.1
num_steps = 500
batch_size = 128
display_step = 100

n_hidden_1 = 2048 # 1st layer number of neurons
n_hidden_2 = 256 # 2nd layer number of neurons
num_input =  10545 
num_output = 49#

X = tf.placeholder("float", shape=(1, num_input))
Y = tf.placeholder("float", shape=(1, num_output))


# Store layers weight & bias
weights = {
    'h1': tf.Variable(tf.random_normal([num_input, n_hidden_1])),
    'h2': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2])),
    'out': tf.Variable(tf.random_normal([n_hidden_2, num_output]))
}
biases = {
    'b1': tf.Variable(tf.random_normal([n_hidden_1])),
    'b2': tf.Variable(tf.random_normal([n_hidden_2])),
    'out': tf.Variable(tf.random_normal([num_output]))
}


# Create model
def neural_net(x):
    # Hidden fully connected layer with 256 neurons
    layer_1 = tf.add(tf.matmul(x, weights['h1']), biases['b1'])
    # Hidden fully connected layer with 256 neurons
    layer_2 = tf.add(tf.matmul(layer_1, weights['h2']), biases['b2'])
    # Output fully connected layer with a neuron for each class
    out_layer = tf.matmul(layer_2, weights['out']) + biases['out']
    return out_layer

# Construct model
logits = neural_net(X)

# Define loss and optimizer
loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
    logits=logits, labels=Y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
train_op = optimizer.minimize(loss_op)

# Evaluate model (with test logits, for dropout to be disabled)
correct_pred = tf.equal(tf.argmax(logits, 1), tf.argmax(Y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

# Initialize the variables (i.e. assign their default value)
init = tf.global_variables_initializer()

# Start training
with tf.Session() as sess:

    # Run the initializer
    sess.run(init)

    for step in range(1, num_steps+1):
        batch_x, batch_y = dataset.train.next_batch(batch_size)
        # Run optimization op (backprop)
        sess.run(train_op, feed_dict={X: batch_x, Y: batch_y})
        if step % display_step == 0 or step == 1:
            # Calculate batch loss and accuracy
            loss, acc = sess.run([loss_op, accuracy], feed_dict={X: batch_x,
                                                                 Y: batch_y})
            print("Step " + str(step) + ", Minibatch Loss= " + \
                  "{:.4f}".format(loss) + ", Training Accuracy= " + \
                  "{:.3f}".format(acc))

    print("Optimization Finished!")

    # Calculate accuracy
    print("Testing Accuracy:", \
        sess.run(accuracy, feed_dict={X: sounds_queue,
                                      Y: labels_queue}))






from tensorflow.examples.tutorials.mnist import input_data
mnist_fashion = input_data.read_data_sets('data/fashion', one_hot=True)

import tensorflow as tf

import matplotlib
import matplotlib.pyplot as plt

import numpy as np
import csv
csvData = []
csvTest = []


# Parameters
learning_rate = 0.000003
training_iters = 200000
batch_size = 64
display_step = 20

# Network Parameters
n_input = 784 # mnist_fashion data input (img shape: 28*28)
n_classes = 10 # mnist_fashion total classes (0-9 digits)
dropout = 0.8 # Dropout, probability to keep units

# tf Graph input
x = tf.placeholder(tf.float32, [None, n_input])
y = tf.placeholder(tf.float32, [None, n_classes])
keep_prob = tf.placeholder(tf.float32) # dropout (keep probability)

# Create AlexNet model
def conv2d(name, l_input, w, b):
    return tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(l_input, w, strides=[1, 1, 1, 1], padding='SAME'),b), name=name)

def max_pool(name, l_input, k):
    return tf.nn.max_pool(l_input, ksize=[1, k, k, 1], strides=[1, k, k, 1], padding='SAME', name=name)

def norm(name, l_input, lsize=4):
    return tf.nn.lrn(l_input, lsize, bias=1.0, alpha=0.001 / 9.0, beta=0.75, name=name)

def alex_net(_X, _weights, _biases, _dropout):
    # Reshape input picture
    _X = tf.reshape(_X, shape=[-1, 28, 28, 1])

    # Convolution Layer
    conv1 = conv2d('conv1', _X, _weights['wc1'], _biases['bc1'])
    # Max Pooling (down-sampling)
    pool1 = max_pool('pool1', conv1, k=2)
    # Apply Normalization
    norm1 = norm('norm1', pool1, lsize=4)
    # Apply Dropout
    norm1 = tf.nn.dropout(norm1, _dropout)

    # Convolution Layer
    conv2 = conv2d('conv2', norm1, _weights['wc2'], _biases['bc2'])
    # Max Pooling (down-sampling)
    pool2 = max_pool('pool2', conv2, k=2)
    # Apply Normalization
    norm2 = norm('norm2', pool2, lsize=4)
    # Apply Dropout
    norm2 = tf.nn.dropout(norm2, _dropout)

    # Convolution Layer
    conv3 = conv2d('conv3', norm2, _weights['wc3'], _biases['bc3'])
    # Max Pooling (down-sampling)
    pool3 = max_pool('pool3', conv3, k=2)
    # Apply Normalization
    norm3 = norm('norm3', pool3, lsize=4)
    # Apply Dropout
    norm3 = tf.nn.dropout(norm3, _dropout)

    # Fully connected layer
    dense1 = tf.reshape(norm3, [-1, _weights['wd1'].get_shape().as_list()[0]]) # Reshape conv3 output to fit dense layer input
    dense1 = tf.nn.relu(tf.matmul(dense1, _weights['wd1']) + _biases['bd1'], name='fc1') # Relu activation

    dense2 = tf.nn.relu(tf.matmul(dense1, _weights['wd2']) + _biases['bd2'], name='fc2') # Relu activation

    # Output, class prediction
    out = tf.matmul(dense2, _weights['out']) + _biases['out']
    return out

# Store layers weight & bias
weights = {
    'wc1': tf.Variable(tf.random_normal([3, 3, 1, 64])),
    'wc2': tf.Variable(tf.random_normal([3, 3, 64, 128])),
    'wc3': tf.Variable(tf.random_normal([3, 3, 128, 256])),
    'wd1': tf.Variable(tf.random_normal([4*4*256, 1024])),
    'wd2': tf.Variable(tf.random_normal([1024, 1024])),
    'out': tf.Variable(tf.random_normal([1024, 10]))
}
biases = {
    'bc1': tf.Variable(tf.random_normal([64])),
    'bc2': tf.Variable(tf.random_normal([128])),
    'bc3': tf.Variable(tf.random_normal([256])),
    'bd1': tf.Variable(tf.random_normal([1024])),
    'bd2': tf.Variable(tf.random_normal([1024])),
    'out': tf.Variable(tf.random_normal([n_classes]))
}



plot_y_acc = []
plot_y_test = []

fig_acc, ax_acc = plt.subplots()
#fig_loss, ax_loss = plt.subplots()
# Data for plotting
ax_acc.set(xlabel='step (s)', ylabel='Accuracy', title='Accuracy at each step')
ax_acc.grid() 

#ax_loss.set(xlabel='step (s)', ylabel='Loss', title='Loss at each step')
#ax_loss.grid()

LR_list = [0.0000005, 0.0000007, 0.000001, 0.000003]
for learning_rate in LR_list:
    plot_y_acc.clear()
    #plot_y_test.clear()

    

    # Construct model
    pred = alex_net(x, weights, biases, keep_prob)

    # Define loss and optimizer
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))
    optimizer = tf.train.MomentumOptimizer(learning_rate=learning_rate, momentum=0.9).minimize(cost)#AdamOptimizer

    # Evaluate model
    correct_pred = tf.equal(tf.argmax(pred,1), tf.argmax(y,1))
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

    # Initializing the variables
    init = tf.initialize_all_variables()
    
    csvData.append(['step', 'lr='+str(learning_rate)])
    # Launch the graph
    with tf.Session() as sess:
        sess.run(init)
        step = 1
        # Keep training until reach max iterations
        while step * batch_size < training_iters:
            batch_xs, batch_ys = mnist_fashion.train.next_batch(batch_size)
            # Fit training using batch data
            sess.run(optimizer, feed_dict={x: batch_xs, y: batch_ys, keep_prob: dropout})
            
            if step % display_step == 0:
                # Calculate batch accuracy
                acc = sess.run(accuracy, feed_dict={x: batch_xs, y: batch_ys, keep_prob: 1.})
                # Calculate batch loss
                loss = sess.run(cost, feed_dict={x: batch_xs, y: batch_ys, keep_prob: 1.})
                plot_y_acc.append(acc)
     
                csvData.append([step, acc])
                
                #plot_y_loss.append(loss)
                print('(lr='+str(learning_rate)+')' + "Iter " + str(step*batch_size) + ", Minibatch Loss= " + "{:.6f}".format(loss) + ", Training Accuracy= " + "{:.5f}".format(acc))
            step += 1
        print ("Optimization Finished!")
        # Calculate accuracy for 256 mnist_fashion test images
        acc_test = sess.run(accuracy, feed_dict={x: mnist_fashion.test.images[:256], y: mnist_fashion.test.labels[:256], keep_prob: 1.})
        plot_y_test.append(acc_test)
        
        print ("Testing Accuracy:", acc_test)
       # csvTest = csvTest + plot_y_test
        ax_acc.plot(range(int(step/display_step)), plot_y_acc, label='lr='+str(learning_rate))#/display_step
        #ax_loss.plot(range(int(step/display_step)), plot_y_loss, label='lr='+str(learning_rate))#/display_step

legend = ax_acc.legend(loc='lower right', shadow=True, fontsize='x-large')    
#legend = ax_loss.legend(loc='upper right', shadow=True, fontsize='x-large')
fig_acc.savefig("compare_lr_acc.png")
#fig_loss.savefig("compare_lr_loss.png")
plt.show()


"""
with open('lr_test.csv', 'w') as csvFile:
    writer = csv.writer(csvFile)
    writer.writerows(csvTest)

csvFile.close()
"""


#csvData = np.reshape(4,-1).T
csvData = np.reshape(np.ravel(csvData, order='F'), (-1, 2*len(LR_list)), order='F')

with open('lr_train.csv', 'w') as csvFile:
    writer = csv.writer(csvFile)
    writer.writerows(csvData)

csvFile.close()


import seaborn as sns
color_order = ['xkcd:cerulean','xkcd:ocean','xkcd:lightish blue','xkcd:powder blue']
sns.barplot(x=LR_list, y=plot_y_test, palette=color_order).set_title("Test accuracy with different learning rate")
plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
plt.show()



























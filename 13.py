########################################################
# Load  libraries
########################################################
from PIL import ImageFilter, ImageStat, Image, ImageDraw
from multiprocessing import Pool, cpu_count
from sklearn.preprocessing import LabelEncoder
import pandas as pd
import numpy as np
import glob
import cv2
import time
from keras.utils import np_utils
import os
import tensorflow as tf
from sklearn.utils import shuffle

tf.compat.v1.disable_v2_behavior()

##########################################################
# Read the input images and then resize the image to 64 x 64 x 3 size
###########################################################


def get_im_cv2(path):
    img = cv2.imread(path)
    resized = cv2.resize(img, (64, 64), cv2.INTER_LINEAR_EXACT)
    return resized

###########################################################
# Each of the folders corresponds to a different class
# Load the images into array and then define their output classes based on
# the folder number
###########################################################


def load_train():
    X_train = []
    X_train_id = []
    y_train = []
    start_time = time.time()

    print("Read train images")
    folders = ['Type_1', 'Type_2', 'Type_3']
    for fld in folders:
        index = folders.index(fld)
        print("load folder {} (Index:{})".format(fld, index))
        path = os.path.join(".", "Downloads", "Intel", "train", fld, "*.jpg")
        files = glob.glob(path)

        for fl in files:
            flbase = os.path.basename(fl)
            img = get_im_cv2(fl)
            X_train.append(img)
            X_train_id.append(flbase)
            y_train.append(index)

    for fld in folders:
        index = folders.index(fld)
        print('load folder {} (Index: {})'.format(fld, index))
        path = os.path.join('.', 'Downloads', 'Intel',
                            'Additional', fld, '*.jpg')
        files = glob.glob(path)

        for fl in files:
            flbase = os.path.basename(fl)
            img = get_im_cv2(fl)
            X_train.append(img)
            X_train_id.append(flbase)
            y_train.append(index)

    print('Read train data time: {} seconds'.format(
        round(time.time() - start_time, 2)))
    return X_train, y_train, X_train_id

###################################################################
# Load the test images
###################################################################


def load_test():
    path = os.path.join(".", "Downloads", "Intel", "test", "*.jpg")
    files = sorted(glob.glob(path))

    X_test = []
    X_test_id = []
    for fl in files:
        flbase = os.path.basename(fl)
        img = get_im_cv2(fl)
        X_test.append(img)
        X_test_id.append(flbase)

    path = os.path.join(".", "Downloads", "Intel", "test_stg2", "*.jpg")
    files = sorted(glob.glob(path))
    for fl in files:
        flbase = os.path.basename(fl)
        img = get_im_cv2(fl)
        X_test.append(img)
        X_test_id.append(flbase)

    return X_test, X_test_id

##################################################
# Normalize the image data to have values between 0 and 1
# by diving the pixel intensity values by 255.
# Also convert the class label into vectors of length 3 corresponding to
# the 3 classes
# Class 1 - [1 0 0]
# Class 2 - [0 1 0]
# Class 3 - [0 0 1]
##################################################


def read_and_normalize_train_data():
    train_data, train_target, train_id = load_train()

    print("convert to numpy...")
    train_data = np.array(train_data, dtype=np.uint8)
    train_target = np.array(train_target, dtype=np.uint8)

    print("reshape...")
    train_data = train_data.transpose((0, 2, 3, 1))
    train_data = train_data.transpose((0, 1, 3, 2))

    print('Convert to float...')
    train_data = train_data.astype("float32")
    train_data = train_data/255
    train_target = np_utils.to_categorical(train_target, 3)

    print("train shape: ", train_data.shape)
    print(train_data.shape[0], ' train samples')

    return train_data, train_target, train_id

###############################################################
# Normalize test-image data
###############################################################


def read_and_normalize_test_data():
    start_time = time.time()
    test_data, test_id = load_test()

    test_data = np.array(test_data, dtype=np.uint8)
    test_data = test_data.transpose((0, 2, 3, 1))
    train_data = test_data.transpose((0, 1, 3, 2))

    test_data = test_data.astype("float32")
    test_data = test_data/255

    print('Test shape:', test_data.shape)
    print(test_data.shape[0], 'test samples')
    print('Read and process test data time: {} seconds'.format(
        round(time.time() - start_time, 2)))

    return test_data, test_id

##########################################################
# Read and normalize the train data
##########################################################


train_data, train_target, train_id = read_and_normalize_train_data()

##########################################################
# Shuffle the input training data to aid stochastic gradient descent
##########################################################

list1_shuf = []
list2_shuf = []
index_shuf = range(len(train_data))
index_shuf = shuffle(index_shuf)

for i in index_shuf:
    list1_shuf.append(train_data[i, :, :, :])
    list2_shuf.append(train_target[i, ])

list1_shuf = np.array(list1_shuf, dtype=np.uint8)
list2_shuf = np.array(list2_shuf, dtype=np.uint8)

##########################################################
# TensorFlow activities for Network Definition and Training
##########################################################
# Create the different layers

channel_in = 3
channel_out = 64
channel_out1 = 128

'''C O N V O L U T I O N L A Y E R'''


def conv2(x, W, b, strides=1):
    x = tf.nn.conv2d(x, W, strides=[1, strides, strides, 1], padding="SAME")
    x = tf.nn.bias_add(x, b)
    return tf.nn.relu(x)


''' P O O L I N G L A Y E R'''


def maxpool2d(x, stride=2):
    return tf.nn.max_pool(x, ksize=[1, stride, stride, 1], padding="same", strides=[1, stride, stride, 1])

# Create the feed-forward model


def conv_net(x, weights, biases, dropout):

    # CNN 1

    conv1 = conv2(x, weights["wc1"], biases["bc1"])
    conv1 = maxpool2d(conv1, stride=2)

    # CNN 2

    conv2_1 = conv2(conv1, weights["wc2"], biases["bc2"])
    conv2_1 = maxpool2d(conv2_1, stride=2)

    conv2_2 = conv2(conv2_1, weights["wc3"], biases["bc3"])
    conv2_2 = maxpool2d(conv2_2, stride=2)

    # Dense Layer

    fc1 = tf.reshape(conv2_2, [-1, weights["wd1"].get_shape().as_list()[0]])
    fc1 = tf.add(tf.matmul(fc1, weights["wd1"]), biases["bd1"])

    # Apply Dropout

    fc1 = tf.nn.dropout(fc1, dropout)

    # Another Dense Layer

    fc2 = tf.add(tf.matmul(fc1, weights["wd2"]), biases["bd2"])
    fc2 = tf.nn.relu(fc2)

    # Apply Dropout

    fc2 = tf.nn.dropout(fc2, dropout)

    # Output class prediction

    out = tf.add(tf.matmul(fc2, weights["out"]), biases["out"])
    return out

######################################################
# Define several parameters for the network and learning
#######################################################


start_time = time.time()
learning_rate = 0.01
epochs = 200
batch_size = 128
num_batches = list1_shuf.shape[0]/128
input_height = 64
input_width = 64
n_classes = 3
dropout = 0.5
display_step = 1
filter_height = 3
filter_width = 3
depth_in = 3
depth_out1 = 64
depth_out2 = 128
depth_out3 = 256


#######################################################
# input–output definition
#######################################################

x = tf.compat.v1.placeholder(
    tf.float32, [None, input_height, input_width, depth_in])
y = tf.compat.v1.placeholder(tf.float32, [None, n_classes])
keep_prop = tf.compat.v1.placeholder(tf.float32)

########################################################
# Define the weights and biases
########################################################

weights = {
    "wc1": tf.Variable(tf.random.normal([filter_height, filter_width, depth_in, depth_out1])),
    "wc2": tf.Variable(tf.random.normal([filter_height, filter_width, depth_out1, depth_out2])),
    "wc3": tf.Variable(tf.random.normal([filter_height, filter_width, depth_out2, depth_out3])),
    "wd1": tf.Variable(tf.random.normal([(input_height/8)*(input_height/8)*256, 512])),
    "wd2": tf.Variable(tf.random.normal([512, 512])),
    "out": tf.Variable(tf.random.normal([512, n_classes]))
}


biases = {
    "bc1": tf.Variable(tf.random.normal([64])),
    "bc2": tf.Variable(tf.random.normal([128])),
    "bc3": tf.Variable(tf.random.normal([256])),
    "bd1": tf.Variable(tf.random.normal([512])),
    "bd2": tf.Variable(tf.random.normal([512])),
    "out": tf.Variable(tf.random.normal([n_classes]))
}

######################################################
# Define the TensorFlow ops for training
######################################################

pred = conv_net(x, weights, biases, keep_prop)

# Define loss function and optimizer

cost = tf.reduce_mean(
    tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))
optimizer = tf.compat.v1.train.AdamOptimizer(
    learning_rate=learning_rate).minimize(cost)

# Evaluate model

correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, dtype=tf.float32))

# Define the initialization op

init = tf.compat.v1.global_variables_initializer()

######################################################
# Launch the execution graph and invoke the training
######################################################

star_time = time.time()

with tf.compat.v1.Session() as sess:
    sess.run(init)

    for i in range(epochs):
        for j in range(num_batches):
            batch_x, batch_y = list1_shuf[i*(batch_size):(i+1)*(
                batch_size)], list2_shuf[i*(batch_size):(i+1)*(batch_size)]
            sess.run(optimizer, feed_dict={
                     x: batch_x, y: batch_y, keep_prop: dropout})
            loss, acc = sess.run([cost, accuracy], feed_dict={
                                 x: batch_x, y: batch_y, keep_prop: 1.})
        if(epochs % display_step == 0):
            print("Epoch:", '%04d' % (i+1), "cost=",
                  "{:.9f}".format(loss), "Training accuracy", "{:.5f}".format(acc))
    
    print('Optimization Completed')

#%%

import os
import cv2
import urllib
import numpy as np
from PIL import Image
import tensorflow as tf
from sklearn.utils import shuffle
from urllib.request import urlopen
from matplotlib import pyplot as plt
from sklearn.cross_validation import train_test_split

#%%
def vid_input:
    url = 'http://192.168.43.1:8080/shot.jpg'

    while True:
        img_resp = urllib.request.urlopen(url)
        img_np = np.array(bytearray(img_resp.read()),dtype=np.uint8)
        img_clr = cv2.imdecode(img_np,-1)
        img_gray = cv2.cvtColor(img_clr, cv2.COLOR_BGR2GRAY)

        v = np.median(img_gray)
        sigma = 0.6

        lower = int(max(0, (1.0 - sigma) * v))
        upper = int(min(255, (1.0 + sigma) * v))

        filter = cv2.Canny(img_gray,lower,upper)
    
        cv2.imshow('Original',img_clr)
        cv2.imshow('Canny Filter',filter)

        if ord('q')==cv2.waitKey(10):
            exit(0)
        
#%%

img_rows = 200 
img_cols = 200
img_channels = 1
batch_size = 32
nb_classes = 5
nb_epoch = 15
nb_filters = 32
pool_shape = 2
filter_shape = 3
learning_rate = 0.0001
output = ["OK", "NOTHING","PEACE", "PUNCH", "STOP"]

#%%
#surf sift orb
x = tf.placeholder(tf.float32, [None, img_rows*img_cols])
x_shaped = tf.reshape(x, [1, img_rows, img_cols, 1])
y = tf.placeholder(tf.float32, [None, 10])

#%%

def conv_layer(input_data, img_channels, nb_filters, filter_shape, pool_shape, name):
    conv_filter_shape = [filter_shape, filter_shape, img_channels, nb_filters]
    weights = tf.Variable(tf.truncated_normal(conv_filter_shape, stddev = 0.03), name = name+'_W')
    bias = tf.Variable(tf.truncated_normal([nb_filters]), name=name+'_b')
    out_layer = tf.nn.conv2d(input_data, weights, [1, 1, 1, 1], padding='SAME')
    out_layer += bias
    out_layer = tf.nn.relu(out_layer)
    ksize = [1, pool_shape, pool_shape, 1]
    strides = [1, 2, 2, 1]
    out_layer = tf.nn.max_pool(out_layer, ksize=ksize, strides=strides, padding='SAME')
    return out_layer

#%%

layer1 = conv_layer(x_shaped, img_channels, nb_filters, filter_shape, pool_shape, name = 'layer1')
layer2 = conv_layer(layer1, 32, nb_filters, filter_shape, pool_shape, name = 'layer2')

#%%

flattened = tf.reshape(layer2, [-1, 80000])

wd1 = tf.Variable(tf.truncated_normal([80000, 128], stddev=0.03), name='wd1')
bd1 = tf.Variable(tf.truncated_normal([128], stddev=0.01), name='bd1')
dense_layer1 = tf.matmul(flattened, wd1) + bd1
dense_layer1 = tf.nn.relu(dense_layer1)

wd2 = tf.Variable(tf.truncated_normal([128, 5], stddev=0.03), name='wd2')
bd2 = tf.Variable(tf.truncated_normal([5], stddev=0.01), name='bd2')
dense_layer2 = tf.matmul(dense_layer1, wd2) + bd2
y_ = tf.nn.softmax(dense_layer2)

#%%

cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits = dense_layer2, labels = y))
os.chdir("D:\Coding\CNN-hand-gesture-detection")
print (os.getcwd())


#%%

def modlistdir(path):
    listing = os.listdir(path)
    retlist = []
    for name in listing:
        if name.startswith('.'):
            continue
        retlist.append(name)
    return retlist
    


path2 = './leapmotion_dataset'
imlist = modlistdir(path2)

image1 = np.array(Image.open(path2 +'/' + imlist[0])) 

m,n = image1.shape[0:2] 
total_images = len(imlist) 

immatrix = np.array([np.array(Image.open(path2+ '/' + images).convert('L')).flatten()
                        for images in imlist], dtype = 'f')



print (immatrix.shape)

label=np.ones((total_images,),dtype = int)

samples_per_class = int(total_images / nb_classes)
print ("samples_per_class - ",samples_per_class)
s = 0
r = int(samples_per_class)
for classIndex in range(nb_classes):
    label[s:r] = classIndex
    s = r
    r = s + samples_per_class


data,Label = shuffle(immatrix,label, random_state=2)
train_data = [data,Label]

(X, Y) = (train_data[0],train_data[1])


# Spltting

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=4)

X_train = X_train.reshape(X_train.shape[0], img_channels, img_rows, img_cols)
X_test = X_test.reshape(X_test.shape[0], img_channels, img_rows, img_cols)

X_train = X_train.astype('float32')
X_test = X_test.astype('float32')

# normalize
X_train /= 255
X_test /= 255

# convert class vectors to binary class matrices
# Y_train = np_utils.to_categorical(y_train, nb_classes)
# Y_test = np_utils.to_categorical(y_test, nb_classes)
# return X_train, X_test, Y_train, Y_test



#%%

optimiser = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cross_entropy)

correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

init_op = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init_op)
    total_batch = int(total_images / batch_size)
    for epoch in range(nb_epoch):
        avg_cost = 0
        for i in range(total_batch):
            batch_x = tf.train.batch(list(X_train), batch_size=batch_size)
            batch_y = tf.train.batch(list(Y_train), batch_size=batch_size)
            _, c = sess.run([optimiser, cross_entropy], feed_dict={x: batch_x, y: batch_y})
            avg_cost += c / total_batch
        test_acc = sess.run(accuracy, feed_dict={x: X_test, y: Y_test})
        print("Epoch:", (epoch + 1), "cost =", "{:.3f}".format(avg_cost), "test accuracy: {:.3f}".format(test_acc))

    print("\nTraining complete!")
    print(sess.run(accuracy, feed_dict={x: X_test, y: Y_test}))

















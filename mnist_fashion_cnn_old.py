from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# import TensorFlow and tf.keras 
import tensorflow as tf
from tensorflow import keras 

# import cv2 
import keras
import numpy as np
import matplotlib.pyplot as plt

# import network model 
from inception_resnet_v1 import create_inception_resnet_v1
# 
print('The version of tensorflow:',tf.__version__)
print('The version of keras:', keras.__version__)

# access the Fashion MNIST directly from TensorFlow,just import and load the data. 
# rename for tf.keras.datasets.fashion_mnist
mnist = tf.keras.datasets.fashion_mnist

# we start to load the Fashion-MNIST data with load_data() method.
# Returns: Tuple of Numpy arrays.
#          '(x_train,y_train),(x_test,y_test)'.
#          '(x_train,label_train),(x_test,label_test)'
#          '(image_train,label_train),(image_test,label_test)'

(training_images, training_labels), (test_images, test_labels) = mnist.load_data()

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
#  
print('\n The type of training_images is',type(training_images))
print('\n The shape of training_images is',training_images.shape)
# 
print('\n The type of test_labels is',type(test_labels))
print('\n The shape of test_labels is %s...' %test_labels.shape)
print('\n The length of test_labels is %s...'%len(test_labels))
#
print('\n **************************************************************')
#
print('\n The shape of training_labels is %s...'%training_labels.shape)
print('\n The length of training_labels is %s...'%len(training_labels))
#
print('\n training_labels looks like',training_labels)
#
#
plt.figure()
plt.imshow(training_images[0])
plt.colorbar()
plt.grid(True)
plt.show()
#
#
# This is numpy.ndarray.reshape
'''
Method

ndarray.reshape(shape,order='C')

Unlike the free function numpy.reshape, this method on ndarray allows the elements of the shape parameter to be passed in as separate arguments.For example, a.reshape(10,11) is equivalent to a.reshape((10,11))

'''
plt.figure(figsize=(10,10)) # the area size not the number of row and column 
for i in range(25): 
    plt.subplot(5,5,i+1)
   # plt.xticks([5,10,15,20,25])
   # plt.yticks([5,10,15,20,25])
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(training_images[i], cmap=plt.cm.binary)
    plt.xlabel(class_names[training_labels[i]])
    #training_labels should be a value from 0-9.
    #class_names[9] is corresponding to ankle boot. 
plt.show()

#training_images=tf.ones(shape=[90,160,160,3])
 
training_images=training_images.reshape(60000, 28, 28, 1)
# 
print('\n The type of training_images is',type(training_images))
print('\n The shape of training_images is',training_images.shape)
# 
training_images=training_images / 255.0
test_images = test_images.reshape(10000, 28, 28, 1)
test_images=test_images/255.0
# scale these values to a range of 0 to 1 before feeding to the neural network model.
# It is important to preprocess the train and test set in the same way Divide the values by 255. 


# Define a function to resize image and convert img to RGB

def process_image(npimgs):
    print('By tf.shape,The shape of npimgs is',tf.shape(npimgs))
    print('The rank of npimgs is',tf.rank(npimgs))
    tfimgs=tf.convert_to_tensor(npimgs,dtype=tf.float32)
    print('The rank of converted tensor tfimgs is',tf.rank(tfimgs))
    imgsrgb=tf.image.grayscale_to_rgb(tfimgs,name='Convert_to_rgb')
    while true:
        batch=imgsrgb[0]
      
    #imgrgb= tf.cond(tf.rank(tfimgis<4,)
    resrgbimgs=tf.image.resize_images(imgsrgb,size=[299,299],method=tf.image.ResizeMethod.AREA,align_corners=False,preserve_aspect_ratio=False)
    #sess=tf.InteractiveSession()
    #print('\n', type(resrgbimgs),'\n')
    
    sess=tf.Session()
    with sess.as_default():
        t=resrgbimgs.eval()
        # numpy to tensor
        # tensor =tf.constant(np_array)
        # print(tensor)
        # Convert tensor back to numpy array
        # npresrgbimgs=resrgbimgs.eval()
        # resrgbimgs=resrgbimgs.reshape(60000,299,299,3)
    return t

procedimgs = process_image(training_images)

print('The type of imgsrgb', type(procedimgs))
print('The shape of imgsrgb', procedimgs.shape)


# scale these values to a range of 0 to 1 before feeding to the neural network model.
# It is important to preprocess the train and test set in the same way Divide the values by 255. 
# Dividing by 25 does not determine the generation of gray image. 
test_images = test_images.reshape(10000, 28, 28, 1)
test_images=test_images/255.0
test_images=process_image(test_images)
#
#
# Bulid the model 
# 
from inception_resnet_v1 import create_inception_resnet_v1
# from inception_resnet_v1_keras import InceptionResNetV1 
model = create_inception_resnet_v1(nb_classes=10,scale=True)
#model = InceptionResNetV1(input_shape=(28,28,1),classes=10,dropout_keep_prob=0.8,weights_path=None)

#model = tf.keras.models.Sequential([
#  tf.keras.layers.Conv2D(64, (3,3), activation='relu', input_shape=(28, 28, 1)),
#  tf.keras.layers.MaxPooling2D(2, 2),
#  tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
#  tf.keras.layers.MaxPooling2D(2,2),
#  tf.keras.layers.Flatten(),
  # Flatten the input. Does not affect the batch size. 
  # (None,28*28*the dimensionality of the output space i.e, the number of output filters in the convolution)
  # The input for dense: [60000,8*8*156][60000,9984]
#  tf.keras.layers.Dense(128, activation='relu'),
  # tf.keras.layers.Dense(128,activation='relu',input_shape=(60000,9984))
  # The output should be (60000,128)
#  tf.keras.layers.Dense(10, activation='softmax')
  #Dense: Just your regular densely-connected NN layer. 
  #Dense implements the operation:
  # output= activation(dot(input,kernel)+bias)
  # Kernel is a weights matrix created by the layer.
#])

# 
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.summary()
#model.fit(training_images, training_labels, epochs=5)
model.fit(procedimgs,training_labels,epochs=5,batch_size=16)
test_loss = model.evaluate(test_images, test_labels)

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from tensorflow import keras 
import keras
import numpy as np
import cv2
import matplotlib.pyplot as plt

from inception_resnet_v1 import create_inception_resnet_v1
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
#
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
(training_images, training_labels), (test_images, test_labels) = mnist.load_data()

#tensor_training_labels = tf.convert_to_tensor(training_labels,dtype=tf.float32)
#tensor_training_labels_2000 = tf.slice(tensor_training_labels,[0],[2000],name=None)

#with tf.Session() as sess:
#    print(sess.run(tensor_training_labels_2000))

# return four Numpy arrays
#
# class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
#
print('\n The type of training_images is',type(training_images))
print('\n The shape of training_images is',training_images.shape)
# 
print('\n The type of test_labels is',type(test_labels))
print('\n The shape of test_labels is %s...' %test_labels.shape)
print('\n The length of test_labels is %s...'%len(test_labels))
#
#
print('**************************************************************')
print('\n The shape of training_labels is %s...'%training_labels.shape)

print('\n The length of training_labels is %s...'%len(training_labels))
#
#

print('\n training_labels looks like',training_labels)




#
#
plt.figure()
plt.imshow(training_images[0])
#plt.colorbar()
#plt.grid(True)
plt.show()





#
#
# The usage of numpy.reshape
'''

Give a new shape to an array without changing its data

numpy.reshape(a,newshape,order='C')

a: array_like Array to be reshaped.

newshape: int or tuple of ints. 


'''
# But here is numpy.ndarray.reshape
'''
Method

ndarray.reshape(shape,order='C')

Unlike the free function numpy.reshape, this method on ndarray allows the elements of the shape parameter to be passed in as separate arguments.For example, a.reshape(10,11) is equivalent to a.reshape((10,11))

'''
plt.figure(figsize=(10,10)) # the area size not the number of  row and columns 
for i in range(25): # i=0 to 24
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
# training_images=np.array(60000,160,160,1)
#training_images=tf.ones(shape=[90,160,160,3])
# 
training_images=training_images.reshape(60000, 28, 28, 1)
print('\n The type of training_images is',type(training_images))

print('\n The shape of training_images is',training_images.shape)
# 
training_images=training_images / 255.0
#
def process_image_cv2(npimgs):
    stacked_rgbrs =[]
    for i range(60000),
        perimgrgb = cv2.merge((npimgs[i]))
    print("The shape of perimgrgb",perimgrgb.shape)
        resize_img= cv2.resize(perimgrgb,(299,299),interpolation=cv2.INTER_AREA)
    print("The shape of resize_img", resize_img.shape)
        stacked_rgbrs=


    #print('By tf.shape,The shape of npimgs is',tf.shape(npimgs))
    #print('The rank of npimgs is',tf.rank(npimgs))
    tfimgs=tf.convert_to_tensor(npimgs,dtype=tf.float32)
    print('The rank of converted tensor tfimgs is',tf.rank(tfimgs))
    imgsrgb=tf.image.grayscale_to_rgb(tfimgs,name='Convert_to_rgb')
    # With tf.slice to slove the problem
    print('The shape of imgsrgb:',imgsrgb.shape)
    resrgbimgs = tf.image.resize_images(imgsrgb, size=[299, 299], method=tf.image.ResizeMethod.BILINEAR,
                                                   align_corners=False, preserve_aspect_ratio=False)


def process_image(npimgs):
    #print('By tf.shape,The shape of npimgs is',tf.shape(npimgs))
    #print('The rank of npimgs is',tf.rank(npimgs))
    tfimgs=tf.convert_to_tensor(npimgs,dtype=tf.float32)
    print('The rank of converted tensor tfimgs is',tf.rank(tfimgs))
    imgsrgb=tf.image.grayscale_to_rgb(tfimgs,name='Convert_to_rgb')
    # With tf.slice to slove the problem
    print('The shape of imgsrgb:',imgsrgb.shape)
    resrgbimgs = tf.image.resize_images(imgsrgb, size=[299, 299], method=tf.image.ResizeMethod.BILINEAR,
                                                   align_corners=False, preserve_aspect_ratio=False)
    # tf.while_loop()

    #splited_120_multi = tf.split(imgsrgb,num_or_size_splits=120,axis=0)
    # define a list of tensor to save all the sub-tensor (500,299,299,3)
    '''
    #all_splits = []
    #for i in range(120):
        splited_name = 'split_' + str(i)
        splited_name= splited_120_multi[i]
        resrgbimgs_500_single = tf.image.resize_images(splited_name, size=[299, 299], method=tf.image.ResizeMethod.BILINEAR, align_corners=False, preserve_aspect_ratio=False)
        # Append
        all_splits.append(resrgbimgs_500_single)
    print('The length of all_splits:', len(all_splits))
    # concat
    resrgbimgs_60000 = tf.concat(all_splits, axis=0)
    print('The type of resrgbimgs_60000', type(resrgbimgs_60000))
    print('The shape of resrgbimgs_60000', resrgbimgs_60000.shape ''''''

            #print('The type of splited_name', type(splited_name))
    #for j in range(120):
        #resrgbimgs_500_single= tf.image.resize_images(splited_name[j],size=[299,299],method=tf.image.ResizeMethod.BILINEAR,align_corners=False,preserve_aspect_ratio=False)
    #print('The type of resrgbimgs_500_single', type(resrgbimgs_500_single))
    #print('The shape of resrgbimgs_500_single',resrgbimgs_500_single.shape)
     #       resrgbimgs_60000=tf.concat(,0)
     '''
    '''
    with sess.as_default():
        npresrgbimgs = resrgbimgs.eval()
    return  npresrgbimgs
    '''
    sess = tf.Session()
    with sess.as_default():
        res = sess.run(resrgbimgs)
    sess.reset(resrgbimgs)
    return res


    #print("name is ", name)
    #imgsrgb_1p = tf.slice(imgsrgb,[0,0,0,0],[500,-1,-1,-1],name=None)
    #imgsrgb_2p = tf.slice(imgsrgb,[10000,0,0,0],[10000,-1,-1,-1],name=None)
    #imgsrgb_3p = tf.slice(imgsrgb,[20000,0,0,0],[10000,-1,-1,-1],name=None)
    #imgsrgb_4p = tf.slice(imgsrgb,[30000,0,0,0],[10000,-1,-1,-1],name=None)
    #imgsrgb_4p = tf.slice(imgsrgb, [30000, 0, 0, 0], [10000, -1, -1, -1], name=None)
    #imgsrgb_4p = tf.slice(imgsrgb, [30000, 0, 0, 0], [10000, -1, -1, -1], name=None)
	#imgsrgb_1p = tf.slice(imgsrgb,[])
                    #print('The shape of imgsrgb_1p',imgsrgb_1p.shape)
    #print('The shape of imgsrgb_2p',imgsrgb_2p.shape)
    #print('The shape of imgsrgb_3p',imgsrgb_3p.shape)
    #print('The shape of imgsrgb_4p',imgsrgb_4p.shape)
	#imgrgb= tf.cond(tf.rank(tfimgis<4,)
                    # resrgbimgs_1p=tf.image.resize_images(imgsrgb_1p,size=[299,299],method=tf.image.ResizeMethod.BILINEAR,align_corners=False,preserve_aspect_ratio=False)
    #resrgbimgs_2p=tf.image.resize_images(imgsrgb_2p,size=[299,299],method=tf.image.ResizeMethod.BILINEAR,align_corners=False,preserve_aspect_ratio=False)
    
    #resrgbimgs_3p=tf.image.resize_images(imgsrgb_3p,size=[299,299],method=tf.image.ResizeMethod.BILINEAR,align_corners=False,preserve_aspect_ratio=False)
    #resrgbimgs_4p=tf.image.resize_images(imgsrgb_4p,size=[299,299],method=tf.image.ResizeMethod.BILINEAR,align_corners=False,preserve_aspect_ratio=False)
    #resrgbimgs=tf.concat([resrgbimgs_1p,resrgbimgs_2p,resrgbimgs_3p,resrgbimgs_4p],0)
    
    #print('The shape of resrgbimgs',resrgbimgs.shape)
                    # print('The shape of resrgbimgs_1p',resrgbimgs_1p.shape)


    #resrgbimgs.eval()

    # sess=tf.InteractiveSession()
    # print(type(resrgbimgs.eval()))
    # resrgbimgs=resrgbimgs.reshape(60000,299,299,3)
    # return t
hund_stream= []

procedimgs = [process_image (training_images[i:i+10]) for i in range(0,training_images.shape[0],10)]

#print('The shape of imgsrgb', tf.shape(procedimgs))
#print('The shape of imgsrgb', procedimgs.shape)


#plt.figure()
#plt.imshow(training_images[0],1,cmap=plt.cm.binary)
#plt.colorbar()
#plt.grid(True)
#plt.show()
#
# 
# scale these values to a range of 0 to 1 before feeding to the neural network model.
# It is important to preprocess the train and test set in the same way Divide the values by 255. 
# Dividing by 25 does not determine the generation of gray image. 
test_images = test_images.reshape(10000, 28, 28, 1)
test_images=test_images/255.0
#test_images=process_image(test_images)
#
# 
#plt.figure()
#plt.imshow(training_images[0])
#plt.colorbar()
#plt.grid(True)
#plt.show()
#
# Bulid the model 
# 
from inception_resnet_v1 import create_inception_resnet_v1
#from inception_resnet_v1_keras import InceptionResNetV1
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
model.fit(training_images, training_labels, epochs=5)
#model.fit(procedimgs,tensor_training_labels_2000,epochs=5,batch_size=16)
#model.fit(procedimgs,tensor_training_labels_2000,epochs=5)

#model.fit(procedimgs,tensor_training_labels_2000,epochs=1)
#test_loss = model.evaluate(test_images, test_labels)

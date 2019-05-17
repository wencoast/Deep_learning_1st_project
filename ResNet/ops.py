# utils for data augmentation 
import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim
import os
import random
from scipy import misc
# Performs random left right(horizontal flipping or not) per batch
# batch[i] random flip using random.getrandbits() method
def _random_flip_leftright(batch):

    print('The batch looks like',np.shape(batch))
    #print('The shape of batch[i]',np.shape(batch[0]))
    batch=np.reshape(batch,[-1,28,28,1])
    # batch=np.expand_dims(batch,axis=-1)
    # if batch_size= 64,i=0,1,2...,63
    for i in range(len(batch)):
        # python bulit in bool function.
        # Return a Boolean valuem True or False
        # random.getrandbits(1) is converted using the standard truth test procedure.
        # if x is false or omitted, this return False. Otherwise, return True.
        if bool(random.getrandbits(1)):
            # random.getrandbits return a python long int with k random bits.
            # Return 1 bit 0/1 this kind random python integer
            batch[i] = np.fliplr(batch[i])
        # Similar to the following script
        # def flip(image,random_flip)
        #     if random_flip and np.random.choice([True,False]):
        #         image=np.fliplr(image)
        #     return image
    return batch

def _random_crop(batch, crop_shape, padding=None):
    oshape = np.shape(batch[0])
    # print('\n The oshape is :', np.shape(batch[0]))
    if padding:
        oshape = (oshape[0] + 2 * padding, oshape[1] + 2 * padding)
    new_batch = []
    npad = ((padding, padding), (padding, padding), (0, 0))
    for i in range(len(batch)):
        new_batch.append(batch[i])
        if padding:
            new_batch[i] = np.lib.pad(batch[i], pad_width=npad,
                                      mode='constant', constant_values=0)
        nh = random.randint(0, oshape[0] - crop_shape[0])
        nw = random.randint(0, oshape[1] - crop_shape[1])
        new_batch[i] = new_batch[i][nh:nh + crop_shape[0],
                       nw:nw + crop_shape[1]]
    return new_batch

def data_augmentation(mini_batch,img_size,is_data_augmentation='False'):

    if is_data_augmentation=='True':

        mini_batch=_random_flip_leftright(mini_batch)
        mini_batch=_random_crop(mini_batch,[img_size,img_size],4)
        mini_batch=np.reshape(mini_batch,[-1,784])
    else:

        mini_batch=mini_batch

    return mini_batch

def classification_loss(logit,label):

    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=label, logits=logit))
    correct_prediction = tf.equal(tf.argmax(logit, 1), tf.argmax(label, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    return loss, accuracy
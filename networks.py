# -*- coding: utf-8 -*-
"""
Created on Wed May 15 21:45:13 2019

@author: Melon
"""

import tensorflow as tf

# Ops
def lrelu(x, alpha=0.2, name="lrelu"):
    return tf.nn.leaky_relu(x, alpha=alpha,name=name)

def conv2d(input_, output_dim, 
           k_h=5, k_w=5, d_h=2, d_w=2, stddev=0.02,
           name="conv2d"):
    with tf.variable_scope(name):
        w = tf.get_variable('kernel', [k_h, k_w, input_.get_shape()[-1], output_dim],
                            initializer=tf.truncated_normal_initializer(stddev=stddev))
        conv = tf.nn.conv2d(input_, w, strides=[1, d_h, d_w, 1], padding='SAME')
        biases = tf.get_variable('biases', [output_dim], initializer=tf.constant_initializer(0.0))
        conv = tf.nn.bias_add(conv, biases)
        return conv
    
def conv2d_transpose(input_, output_shape,
                     k_h=5, k_w=5, d_h=2, d_w=2,
                     stddev=0.02, name="conv2d_transpose",):
    with tf.variable_scope(name):
        w = tf.get_variable('kernel', [k_h, k_w, output_shape[-1], input_.get_shape()[-1]],
                            initializer=tf.random_normal_initializer(stddev=stddev))
        deconv = tf.nn.conv2d_transpose(input_, w, output_shape=output_shape,
                                        strides=[1, d_h, d_w, 1])
        biases = tf.get_variable('biases', [output_shape[-1]], initializer=tf.constant_initializer(0.0))
        deconv = tf.nn.bias_add(deconv, biases)
    return deconv

def batch_norm(x, name, training=True):
    return tf.layers.batch_normalization(x, momentum=0.99,epsilon=1e-3,scale=True,
                                         name=name,training=training)

#networks
def generator(x, o_c=3, gf_dim=32, training=True):
    with tf.variable_scope("generator",reuse=tf.AUTO_REUSE):
        s = x.get_shape().as_list()[1]
        batch_size = x.get_shape().as_list()[0]
        s2, s4, s8, s16, s32 = int(s/2), int(s/4), int(s/8), int(s/16), int(s/32)
        #Downsample
        e1 = conv2d(x, gf_dim, name='down1_conv')
        e2 = batch_norm(conv2d(lrelu(e1), gf_dim*2, name='down2_conv'),name='down2_bn',training=training)
        e3 = batch_norm(conv2d(lrelu(e2), gf_dim*4, name='down3_conv'),name='down3_bn',training=training)
        e4 = batch_norm(conv2d(lrelu(e3), gf_dim*8, name='down4_conv'),name='down4_bn',training=training)
        e5 = batch_norm(conv2d(lrelu(e4), gf_dim*8, name='down5_conv'),name='down5_bn',training=training)
        e6 = batch_norm(conv2d(lrelu(e5), gf_dim*8, name='down6_conv'),name='down6_bn',training=training)
        #Upsample
        d1 = batch_norm(conv2d_transpose(tf.nn.relu(e6),[batch_size, s32, s32, gf_dim*8],
                                         name='up1_convtranspose'),name='up1_bn')
        d1 = tf.concat([d1, e5], 3)
        d2 = batch_norm(conv2d_transpose(tf.nn.relu(d1),[batch_size, s16, s16, gf_dim*8],
                                         name='up2_convtranspose'),name='up2_bn')
        d2 = tf.concat([d2, e4], 3)
        d3 = batch_norm(conv2d_transpose(tf.nn.relu(d2),[batch_size, s8, s8, gf_dim*4],
                                         name='up3_convtranspose'),name='up3_bn')
        d3 = tf.concat([d3, e3], 3)
        d4 = batch_norm(conv2d_transpose(tf.nn.relu(d3),[batch_size, s4, s4, gf_dim*2],
                                         name='up4_convtranspose'),name='up4_bn')
        d4 = tf.concat([d4, e2], 3)
        d5 = batch_norm(conv2d_transpose(tf.nn.relu(d4),[batch_size, s2, s2, gf_dim],
                                         name='up5_convtranspose'),name='up5_bn')
        d5 = tf.concat([d5, e1], 3)
        d6 = conv2d_transpose(tf.nn.relu(d5),[batch_size, s, s, o_c],
                              name='up6_convtranspose')
        return tf.nn.tanh(d6)

def discriminator(x, name, df_dim=32):
    with tf.variable_scope(name,reuse=tf.AUTO_REUSE):
        h1 = lrelu(conv2d(x, df_dim, name='conv1'))
        h2 = lrelu(batch_norm(conv2d(h1,df_dim*2, name='conv2'),name='bn2',training=True))
        h3 = lrelu(batch_norm(conv2d(h2,df_dim*4, name='conv3'),name='bn3',training=True))
        h4 = lrelu(batch_norm(conv2d(h3,df_dim*8, name='conv4'),name='bn4',training=True))
        h5 = lrelu(batch_norm(conv2d(h4,df_dim*8, name='conv5'),name='bn5',training=True))
        h6 = conv2d(h5,1, name='conv6')
        return h6
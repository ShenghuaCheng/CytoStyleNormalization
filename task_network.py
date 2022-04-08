# -*- coding: utf-8 -*-
"""
Created on Fri Oct 26 20:59:49 2018

@author: Melon
"""

import tensorflow as tf
import numpy as np

channel_axis = 3

class net():
    channel_axis = 3
    # layers
    def _conv2d(self, x, kernel_size, num_o, stride, name, biased=True):
        num_x = x.shape[self.channel_axis].value
        with tf.variable_scope(name) as scope:
            w = tf.get_variable('kernel', shape=[kernel_size, kernel_size, num_x, num_o])
            s = [1, stride, stride, 1]
            o = tf.nn.conv2d(x, w, s, padding='SAME')
            if biased:
                b = tf.get_variable('bias', shape=[num_o])
                o = tf.nn.bias_add(o, b)
            return o

    def _fc(self, x, num_o, activation=None, name=None):### kernel  bias 与kera一致
        return tf.layers.dense(x, num_o, activation=activation, name=name)
    
    def _dropout(self, x, rate, name=None):
        return tf.layers.dropout(x, rate=rate, name=name)

    def _relu(self, x, name=None):
        return tf.nn.relu(x, name=name)

    def _add(self, x_l, name):
        return tf.add_n(x_l, name=name)

    def _max_pool2d(self, x, kernel_size, stride, name):
        k = [1, kernel_size, kernel_size, 1]
        s = [1, stride, stride, 1]
        return tf.nn.max_pool(x, k, s, padding='VALID', name=name)
    
    def _avg_pool2d(self, x, kernel_size, stride, name):
        k = [1, kernel_size, kernel_size, 1]
        s = [1, stride, stride, 1]
        return tf.nn.avg_pool(x, k, s, padding='VALID', name=name)
    
    def _max_globalpool2d(self, x, name='global_maxpooling'):
        size = x.get_shape().as_list()[1]
        k = [1, size, size, 1]
        s = [1, size, size, 1]
        return tf.squeeze(tf.nn.max_pool(x, k, s, padding='VALID'),axis=[1,2],name=name)        
    
    def _batch_norm_predict(self, x, name, epsilon=1e-3):#_batch_norm_v4
        with tf.variable_scope(name):
            shape = x.get_shape().as_list()
            gamma = tf.get_variable('gamma',shape[-1],initializer=tf.constant_initializer(1.0),trainable=True,dtype=tf.float32)
            beta = tf.get_variable('beta',shape[-1],initializer=tf.constant_initializer(0.0),trainable=True,dtype=tf.float32)
            moving_avg = tf.get_variable('moving_mean',shape[-1],initializer=tf.constant_initializer(0.0),trainable=True,dtype=tf.float32)
            moving_var = tf.get_variable('moving_variance',shape[-1],initializer=tf.constant_initializer(1.0),trainable=True,dtype=tf.float32)
            output = tf.nn.batch_normalization(x,moving_avg,moving_var,offset=beta,scale=gamma,variance_epsilon=epsilon)
            return output
        
    def _batch_norm_train(self,inputs,name=None):#_batch_norm_v3
        return tf.layers.batch_normalization(inputs, name=name, axis=3, epsilon=1e-3,
                                             momentum=0.99, training=True, 
                                             gamma_initializer=tf.random_normal_initializer(1.0, 0.02))
    
    def _batch_norm_predict_v2(self,inputs,name=None):#_batch_norm_v3
        return tf.layers.batch_normalization(inputs, name=name, axis=3, epsilon=1e-3,
                                             momentum=1, training=False, 
                                             gamma_initializer=tf.random_normal_initializer(1.0, 0.02))
    
class ResNet50(net):
    def __init__(self, inputs, num_classes, phase=True, reuse=False):
        self.inputs = inputs
        self.num_classes = num_classes
        self.channel_axis = 3
        self.reuse = reuse
        self.phase = phase
        if self.phase:# train (True) or predict (False), for BN layers
            self._batch_norm = self._batch_norm_train
        else:
            self._batch_norm = self._batch_norm_predict_v2
        self.build_network()

    def build_network(self):
        self.encoding = self.build_encoder()
        self.outputs = self.build_decoder(self.encoding)

    def build_encoder(self):
        outputs = self._conv2d(self.inputs, 7, 64, 2, name='conv1')
        outputs = self._batch_norm(outputs, name='bn_conv1')
        outputs = self._relu(outputs)
        outputs = self._max_pool2d(outputs, 3, 2, name='pool1')
        print("after stage1:", outputs.shape)
        
        outputs = self.conv_block(outputs, 256, stage=2, block='a', first_s=1)
        outputs = self.identity_block(outputs, 256, stage=2, block='b')
        outputs = self.identity_block(outputs, 256, stage=2, block='c')
        print("after stage2:", outputs.shape)
        
        outputs = self.conv_block(outputs, 512, stage=3, block='a')
        outputs = self.identity_block(outputs, 512, stage=3, block='b')
        outputs = self.identity_block(outputs, 512, stage=3, block='c')
        outputs = self.identity_block(outputs, 512, stage=3, block='d')
        print("after stage3:", outputs.shape)
        
        outputs = self.conv_block(outputs, 1024, stage=4, block='a')
        outputs = self.identity_block(outputs, 1024, stage=4, block='b')
        outputs = self.identity_block(outputs, 1024, stage=4, block='c')
        outputs = self.identity_block(outputs, 1024, stage=4, block='d')
        outputs = self.identity_block(outputs, 1024, stage=4, block='e')
        outputs = self.identity_block(outputs, 1024, stage=4, block='f')
        print("after stage4:", outputs.shape)
        
        outputs = self.conv_block(outputs, 2048, stage=5, block='a')
        outputs = self.identity_block(outputs, 2048, stage=5, block='b')
        outputs = self.identity_block(outputs, 2048, stage=5, block='c')
        print("after stage5:", outputs.shape)
        return outputs

    def build_decoder(self, encoding):
        outputs = self._avg_pool2d(encoding, 7, 7, 'avg_pool')
        outputs = self._max_globalpool2d(outputs)
        outputs = self._fc(outputs, 256, name='dense_1', activation=tf.nn.relu)
        outputs = self._fc(outputs, 1,  name='dense_2')
        print("after decoder:", outputs.shape)
        return outputs
            
    # blocks
    def identity_block(self, x, num_o, stage, block):
        first_s = 1
        assert num_o % 4 == 0, 'Bottleneck number of output ERROR!'
        conv_name_base = 'res' + str(stage) + block + '_branch'
        bn_name_base = 'bn' + str(stage) + block + '_branch'
        # branch1
        o_b1 = x
        # branch2
        o_b2a = self._conv2d(x, 1, num_o / 4, first_s, name=conv_name_base+'2a')
        o_b2a = self._batch_norm(o_b2a, name=bn_name_base+'2a')
        o_b2a = self._relu(o_b2a)

        o_b2b = self._conv2d(o_b2a, 3, num_o / 4, 1, name=conv_name_base+'2b')
        o_b2b = self._batch_norm(o_b2b, name=bn_name_base+'2b')
        o_b2b = self._relu(o_b2b)

        o_b2c = self._conv2d(o_b2b, 1, num_o, 1, name=conv_name_base+'2c')
        o_b2c = self._batch_norm(o_b2c, name=bn_name_base+'2c')
        # add
        outputs = self._add([o_b1,o_b2c], name='stage%d_add'%stage)
        # relu
        outputs = self._relu(outputs, name='stage%d_relu'%stage)
        return outputs 
    
    def conv_block(self, x, num_o, stage, block, first_s=2):
        assert num_o % 4 == 0, 'Bottleneck number of output ERROR!'
        conv_name_base = 'res' + str(stage) + block + '_branch'
        bn_name_base = 'bn' + str(stage) + block + '_branch'
        # branch1
        o_b1 = self._conv2d(x, 1, num_o, first_s, name=conv_name_base+'1')
        o_b1 = self._batch_norm(o_b1, name=bn_name_base+'1')
        # branch2
        o_b2a = self._conv2d(x, 1, num_o / 4, first_s, name=conv_name_base+'2a')
        o_b2a = self._batch_norm(o_b2a, name=bn_name_base+'2a')
        o_b2a = self._relu(o_b2a)

        o_b2b = self._conv2d(o_b2a, 3, num_o / 4, 1, name=conv_name_base+'2b')
        o_b2b = self._batch_norm(o_b2b, name=bn_name_base+'2b')
        o_b2b = self._relu(o_b2b)

        o_b2c = self._conv2d(o_b2b, 1, num_o, 1, name=conv_name_base+'2c')
        o_b2c = self._batch_norm(o_b2c, name=bn_name_base+'2c')
        # add
        outputs = self._add([o_b1,o_b2c], name='stage%d_add'%stage)
        # relu
        outputs = self._relu(outputs, name='stage%d_relu'%stage)
        return outputs
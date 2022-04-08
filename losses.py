# -*- coding: utf-8 -*-
"""
Created on Wed Jul 31 19:48:03 2019

@author: Melon
"""

import tensorflow as tf

def nsgan_loss(logits, is_real):
    """
    logits: without sigmoid
    is_real: Ture for real image; False for fake image
    """
    labels = tf.ones_like(logits) if is_real else tf.zeros_like(logits)
    loss = tf.nn.sigmoid_cross_entropy_with_logits(logits=logits, labels=labels)
    return tf.reduce_mean(loss)

def task_loss(logits, labels):    
    """
    logits: without sigmoid
    label: classification labels
    """
    loss = tf.nn.sigmoid_cross_entropy_with_logits(logits=logits,labels=labels)
    return tf.reduce_mean(loss)

def l1_loss(preds, labels):
    loss = tf.abs(preds - labels)
    return tf.reduce_mean(loss)
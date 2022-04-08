# -*- coding: utf-8 -*-
"""
Created on Thu Jul 26 16:42:34 2018

@author: Melon
"""

import os
import time
import tensorflow as tf
import numpy as np
import cv2

from networks import generator, discriminator
import losses
from task_network import ResNet50
from utils_tf import read_lines, get_color_gm, prepare_labels

class NormNet(object):
    def __init__(self, sess, args):
        """
        Args:
            sess: TensorFlow session
            args: aguments set 
        """
        self.sess = sess
        self.args = args
        
        self.flag_L1 = True if self.args['model']['lambda_L1'] else False
        self.flag_d_intra = True if self.args['model']['discriminator_intra']['lambda_L_d_intra'] else False
        self.flag_d_inter = True if self.args['model']['discriminator_inter']['lambda_L_d_inter'] else False
        self.flag_task = True if self.args['model']['tasknet']['lambda_L_task'] else False
        
        self.gf_dim = self.args['model']['generator']['gf_dim']
        if self.flag_d_intra or self.flag_d_inter:
            self.df_dim = self.args['model']['discriminator_intra']['df_dim']
            
        self.batchsize = self.args['batchsize']
        
        self.build_model()

    def build_model(self):
        self.input_shape = (self.batchsize,
                            self.args['input']['size'],
                            self.args['input']['size'],
                            self.args['input']['channel'])
        self.output_shape = (self.batchsize,
                            self.args['output']['size'],
                            self.args['output']['size'],
                            self.args['output']['channel'])
        # set placeholder
        ## s for source
#        if self.flag_L1 or self.flag_d_intra or self.flag_task:
        self.s_gm = tf.placeholder(dtype=tf.float32, shape=self.input_shape)
#        if self.flag_L1 or self.flag_d_intra or self.flag_d_inter:
        self.s_color = tf.placeholder(dtype=tf.float32, shape=self.output_shape)
        if self.flag_task:
            temp = (self.args['batchsize'], self.args['model']['tasknet']['num_classes'])
            self.s_label = tf.placeholder(dtype=tf.float32, shape=temp)
        ## t for target
        if self.flag_d_inter: 
            self.t_color = tf.placeholder(dtype=tf.float32, shape=self.output_shape)
            self.t_gm = tf.placeholder(dtype=tf.float32, shape=self.input_shape)
        
        # generate images
        if self.flag_L1 or self.flag_d_intra or self.flag_task:
            self.fake_s_color = generator(self.s_gm, gf_dim=self.gf_dim,
                                          o_c=self.output_shape[-1])
        if self.flag_d_inter:
            self.fake_t_color = generator(self.t_gm, gf_dim=self.gf_dim,
                                          o_c=self.output_shape[-1])
            
        # compute loss
        ## intra-domain
        self.loss_dict = {}
        if self.flag_d_intra:
            d_intra_logits_real =discriminator(self.s_color, df_dim=self.df_dim,
                                                 name='intra_discriminator')
            d_intra_logits_fake = discriminator(self.fake_s_color, df_dim=self.df_dim,
                                                name='intra_discriminator')
            d_intra_loss_real = losses.nsgan_loss(d_intra_logits_real, is_real=True)
            d_intra_loss_fake = losses.nsgan_loss(d_intra_logits_fake, is_real=False)
            self.d_intra_loss = d_intra_loss_real + d_intra_loss_fake
            tf.summary.scalar("d_intra_loss", self.d_intra_loss)
            self.loss_dict.update({'d_intra_loss':self.d_intra_loss})
        ## inter-domain
        if self.flag_d_inter:
            d_inter_logits_real = discriminator(self.s_color, df_dim=self.df_dim,
                                               name='inter_discriminator')
            d_inter_logits_fake = discriminator(self.fake_t_color, df_dim=self.df_dim,
                                                name='inter_discriminator')
            d_inter_loss_real = losses.nsgan_loss(d_inter_logits_real, is_real=True)
            d_inter_loss_fake = losses.nsgan_loss(d_inter_logits_fake, is_real=False)
            self.d_inter_loss = d_inter_loss_real + d_inter_loss_fake
            tf.summary.scalar("d_inter_loss", self.d_inter_loss)
            self.loss_dict.update({'d_inter_loss':self.d_inter_loss})
        ## Generator loss
        flag = False
        self.g_loss_dict = {}
        ### l1 loss
        if self.flag_L1:
            self.l1_loss = losses.l1_loss(self.fake_s_color, self.s_color)
            self.g_loss_dict.update({'g_l1':self.l1_loss})
            _lambda = self.args['model']['lambda_L1']
            self.g_loss = _lambda * self.l1_loss
            flag = True
            tf.summary.scalar("l1_loss", self.l1_loss)
        ### task loss
        if self.flag_task:
            num_classes = self.args['model']['tasknet']['num_classes']
            task_net = ResNet50(self.fake_s_color, num_classes, phase=False)
            task_pred_logits = task_net.outputs
            self.task_loss = losses.task_loss(task_pred_logits, self.s_label)
            self.g_loss_dict.update({'g_loss_task':self.task_loss})
            _lambda = self.args['model']['tasknet']['lambda_L_task']
            if flag:
                self.g_loss += _lambda * self.task_loss
            else:
                self.g_loss = _lambda * self.task_loss
                flag = True
            tf.summary.scalar("g_task_loss", self.task_loss)
        ### d-intra loss
        if self.flag_d_intra:
            self.g_loss_intra = losses.nsgan_loss(d_intra_logits_fake, True)
            self.g_loss_dict.update({'g_d_intra':self.g_loss_intra})
            _lambda = self.args['model']['discriminator_intra']['lambda_L_d_intra']
            if flag:
                self.g_loss += _lambda * self.g_loss_intra
            else: 
                self.g_loss = _lambda * self.g_loss_intra
                flag = True
            tf.summary.scalar("g_loss_intra", self.g_loss_intra)
        ### d-inter loss
        if self.flag_d_inter:
            self.g_loss_inter = losses.nsgan_loss(d_inter_logits_fake, True)
            self.g_loss_dict.update({'g_d_inter':self.g_loss_inter})
            _lambda = self.args['model']['discriminator_inter']['lambda_L_d_inter']
            if flag:
                self.g_loss += _lambda * self.g_loss_inter
            else:
                self.g_loss = _lambda * self.g_loss_inter
                flag = True
            tf.summary.scalar("g_loss_inter", self.g_loss_inter)
        tf.summary.scalar("g_loss", self.g_loss)
        self.loss_dict.update(self.g_loss_dict)
        
        #log
        self.sample = tf.concat([self.fake_s_color, self.s_gm[:,:,:,1:], self.s_color],2)
        if self.flag_d_inter:
            sample_t = tf.concat([self.fake_t_color, self.t_gm[:,:,:,1:], self.t_color],2)
            self.sample = tf.concat([self.sample,sample_t],1)
        self.sample = (self.sample+1)*127.5
        
        #divide variable group
        t_vars = tf.trainable_variables()
        global_vars = tf.global_variables()
        self.normnet_vars_global = []
        if self.flag_d_intra:
            self.d_intra_vars = [var for var in t_vars if 'intra_discriminator' in var.name]
            self.d_intra_vars_global = [var for var in global_vars if 'intra_discriminator' in var.name]
            self.normnet_vars_global += self.d_intra_vars_global
        if self.flag_d_inter:
            self.d_inter_vars = [var for var in t_vars if 'inter_discriminator' in var.name]
            self.d_inter_vars_global = [var for var in global_vars if 'inter_discriminator' in var.name]
            self.normnet_vars_global += self.d_inter_vars_global
        self.g_vars = [var for var in t_vars if 'generator' in var.name]
        self.g_vars_global = [var for var in global_vars if 'generator' in var.name]
        self.normnet_vars_global += self.g_vars_global
        if self.flag_task:
            self.tasknet_vars = [var for var in t_vars if var not in self.normnet_vars_global]
#            self.tasknet_vars_tainable = self.tasknet_vars[44:]
            self.tasknet_vars_global = [var for var in global_vars if var not in self.normnet_vars_global]
            
        #saver
        vars_save = self.normnet_vars_global
        self.saver = tf.train.Saver(var_list=vars_save, max_to_keep=20)
    
    def load(self):
        if self.args['model']['generator']['pretrained_path']:
            print(" [*] Reading checkpoint for generator")
            loader = tf.train.Saver(var_list=self.g_vars_global)
            loader.restore(self.sess, self.args['model']['generator']['pretrained_path'])
        if self.flag_d_intra and self.args['model']['discriminator_intra']['pretrained_path']:
            print(" [*] Reading checkpoint for discriminator_intra")
            loader = tf.train.Saver(var_list=self.d_intra_vars_global)
            loader.restore(self.sess, self.args['model']['discriminator_intra']['pretrained_path'])
        if self.flag_d_inter and self.args['model']['discriminator_inter']['pretrained_path']:
            print(" [*] Reading checkpoint for discriminator_inter")
            loader = tf.train.Saver(var_list=self.d_inter_vars_global)
            loader.restore(self.sess, self.args['model']['discriminator_inter']['pretrained_path'])
        if self.flag_task:
            print(" [*] Reading checkpoint for tasknet")
            loader = tf.train.Saver(var_list=self.tasknet_vars_global)
            loader.restore(self.sess, self.args['model']['tasknet']['pretrained_path'])
           
    def save(self, epoch):
        model_name = "NormNet"
        self.saver.save(self.sess, os.path.join(self.args['ckpt_path'], model_name),
                        global_step=epoch)
    
    def train(self):
        lr_g = self.args['model']['generator']['lr_g']
        optimizer = self.args['optimizer']['func']
        optims = []
        g_optim = optimizer(lr_g, **self.args['optimizer']['parameters'])\
                            .minimize(self.g_loss, var_list=self.g_vars)
        optims.append(g_optim)
        if self.flag_d_intra:
            lr_d_intra = self.args['model']['discriminator_intra']['lr_d_intra']
            d_intra_optim = optimizer(lr_d_intra, **self.args['optimizer']['parameters'])\
                            .minimize(self.d_intra_loss, var_list=self.d_intra_vars)
            optims.append(d_intra_optim)
        if self.flag_d_inter:
            lr_d_inter = self.args['model']['discriminator_inter']['lr_d_inter']
            d_inter_optim = optimizer(lr_d_inter, **self.args['optimizer']['parameters'])\
                            .minimize(self.d_inter_loss, var_list=self.d_inter_vars)
            optims.append(d_inter_optim)
            
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        train_op = tf.group(optims, update_ops)                  
        
        init_op = tf.global_variables_initializer()
        
        self.sess.run(init_op)

        self.merge = tf.summary.merge_all() 
        self.writer = tf.summary.FileWriter(self.args['log_path'], self.sess.graph)

        counter = 1
        start_time = time.time()
        
        self.load()
        self.sess.graph.finalize()
        
        # tensor to visualize
        losstensor_list = [self.loss_dict[key] for key in self.loss_dict]
        lossname_list = [key for key in self.loss_dict]
        
        s_color_dirlist = read_lines(self.args['source_domain']['dataset_path'])
        if self.flag_d_inter:
            t_color_dirlist= read_lines(self.args['target_domain']['dataset_path'])
            steps = len(t_color_dirlist)//self.batchsize
            iters = len(t_color_dirlist)
        else:
            steps = len(s_color_dirlist)//self.batchsize
            iters = len(s_color_dirlist)
        buffer_size = self.args['buffer_size']
        
        try:
            start_epoch = self.args['start_epoch']
        except: start_epoch = 0
        for epoch in range(start_epoch, self.args['epoch']):
            np.random.shuffle(s_color_dirlist)
            if self.flag_d_inter:
                np.random.shuffle(t_color_dirlist)
            for i in range(0,iters,self.batchsize*buffer_size):
                d_s_t = time.time()
                print('Loading Data')
                # source data
                _s_color_dirlist = s_color_dirlist[i:i+self.batchsize*buffer_size]
                _s_color_gm_list = get_color_gm(_s_color_dirlist)
                _s_color_list = _s_color_gm_list[:,:,:,:3]
                _s_gm_list = _s_color_gm_list[:,:,:,3:]
                _s_label_list = np.expand_dims(prepare_labels(_s_color_dirlist),-1)
                # target data
                if self.flag_d_inter:
                    _t_color_dirlist = t_color_dirlist[i:i+self.batchsize*buffer_size]
                    _t_color_gm_list = get_color_gm(_t_color_dirlist)
                    _t_color_list = _t_color_gm_list[:,:,:,:3]
                    _t_gm_list = _t_color_gm_list[:,:,:,3:]
                print('cost time: %.2f seconds'%(time.time()-d_s_t))
                for j in range(buffer_size):     
                    feed_dict={self.s_color:_s_color_list[j*self.batchsize:(j+1)*self.batchsize],
                               self.s_gm:_s_gm_list[j*self.batchsize:(j+1)*self.batchsize]}
                    if self.flag_task:
                        feed_dict.update({self.s_label:\
                                          _s_label_list[j*self.batchsize:(j+1)*self.batchsize]})
                    if self.flag_d_inter:
                        feed_dict.update({self.t_color:\
                                          _t_color_list[j*self.batchsize:(j+1)*self.batchsize],
                                          self.t_gm:\
                                          _t_gm_list[j*self.batchsize:(j+1)*self.batchsize]})
                    if j != 0:
                        tensor_list = [train_op] + losstensor_list
                        try:
                            result_list = self.sess.run(tensor_list,feed_dict=feed_dict)
                        except ValueError: print('Drop last')
                        result_loss_list = result_list[1:]
                    else:
                        tensor_list = [train_op] + losstensor_list + [self.merge, self.sample]
                        try:
                            result_list = self.sess.run(tensor_list, feed_dict=feed_dict)
                        except ValueError: print('Drop last')
                        result_loss_list = result_list[1:-2]
                        self.writer.add_summary(result_list[-2], counter)
                        cv2.imwrite(os.path.join(self.args['log_path'],'%d_%d.tif'%(epoch,i//self.batchsize+j)),
                                    np.uint8(result_list[-1][0][:,:,::-1]))
                    print("Epoch:[%2d] [%4d/%4d] time:%2.2f"%(epoch, i//self.batchsize+j, 
                          steps,time.time()-start_time), end='')
                    for idx, loss in enumerate(lossname_list):
                        print(' %s:%.3f'%(loss, result_loss_list[idx]), end='')
                    print('')
                    start_time = time.time()
                    counter += 1
            self.save(epoch)
# -*- coding: utf-8 -*-
"""
Created on Wed May 22 15:25:26 2019

@author: Melon
"""

import multiprocessing.dummy as multiprocessing
import numpy as np
import tensorflow as tf
import cv2
import os

from networks import generator
from task_network import ResNet50


def get_gm(color_dirlist, ith):#挖去背景，图像增强
    color_dir = color_dirlist[ith]
    color = cv2.imread(color_dir)[:,:,::-1]
    mask = cv2.imread(color_dir.replace('_color','_mask_encode_mask_lab_148'))[:,:,::-1]
    gray = np.expand_dims(cv2.cvtColor(color,cv2.COLOR_RGB2GRAY),-1)
    gm = np.concatenate([gray,mask],-1)
    gm = np.float32((gm/255 - 0.5)*2)
    return gm

def get_color_gm_Multiprocess(dirList,threadNum=5):
    pool = multiprocessing.Pool(threadNum)
    imgTotals = []
    for ith in range(len(dirList)):
        imgTotal = pool.apply_async(get_gm, args=(dirList,ith))
        imgTotals.append(imgTotal)
    pool.close()
    pool.join()
    imgTotal = np.array([x.get() for x in imgTotals])
    return imgTotal

def inverse_transform(images):
    return np.uint8((images+1.)*127.5)

def dirchage(_dir,save_dir):
    c = _dir.split('test')[1].split('\\')[1]
    filename = _dir.split('test')[1].split('\\')[-1]
    fold_dir = os.path.join(save_dir,c)
    if not os.path.exists(fold_dir): os.makedirs(fold_dir)
    return os.path.join(fold_dir,filename)

def get_dirlist(txt_dir):
    with open(txt_dir,'r') as f:
        dirlist = f.readlines()
    l = []
    for i in dirlist:
        l.append(i[:-1])
    return l

def preparelabel(dirlist):
    labellist = []
    for s in dirlist:
        if '_ASCUS' in s or '_HISL' in s or '_LISL' in s:
            labellist.append(1.)
        else:
            labellist.append(0.)
    return labellist

def calcu_accu(preds,labels):
    tpredict = np.sum(np.abs(preds-labels)< 0.5)
    return tpredict/len(labels)


if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = '1'
    domain = 'sfy-our'
    name = 'NormNet_sfy-ourTOsfy-3d1_L1_100_d1_1.0_d2_1.0_task_0.0'
    txt_dir = '../dataset/DA_dataset_dirlist_K/%s_test.txt'%domain
    model_dir = './ckpt_stage2_no_aug/%s/NormNet-'%name
    result_dir = './DA_result_stage2_no_aug'
    if not os.path.exists(result_dir): os.makedirs(result_dir)
    dirlist = get_dirlist(txt_dir)
    import random
    dirlist = random.sample(dirlist,1000)
    batch_size = 4
    # load img
    print("Loading img")
    gms = get_color_gm_Multiprocess(dirlist)
    labels = preparelabel(dirlist)
    # Generator setting
    gm_tensor = tf.placeholder(tf.float32,shape=[batch_size,512,512,4])
    fake_s_color = generator(gm_tensor,o_c=3,gf_dim=32,training=False)
    # task predict
    num_classes = 1
    task_net = ResNet50(fake_s_color, num_classes, phase=False)
    task_pred_logits = tf.sigmoid(task_net.outputs)
    sess = tf.Session()
    vars_g = [var for var in tf.global_variables() if 'generator' in var.name]
    vars_t = [var for var in tf.global_variables() if var not in vars_g]
    saver_g = tf.train.Saver(var_list=vars_g)
    saver_t = tf.train.Saver(var_list=vars_t)
    saver_t.restore(sess,'./pretrained_models/sfy-our/tasknet6-20')
    # train and record
    f = open(os.path.join(result_dir,domain+'.txt'),'a')
    for epoch in range(0,15):
        saver_g.restore(sess,model_dir+str(epoch))
        print("Testing")
        preds = []         
        for i in range(0,gms.shape[0],batch_size):
            gm = gms[i:i+batch_size,:,:,:]
            pred = sess.run(task_pred_logits,feed_dict={gm_tensor:gm})
            preds.append(pred)
        preds = np.squeeze(np.concatenate(preds,0),-1)
        accu = calcu_accu(preds,labels)
        record_str = '%s epoch%d'%(name,epoch)+'[accu:%.4f]'%accu
        print(record_str)
        f.write(record_str+'\n')
    f.close()
            
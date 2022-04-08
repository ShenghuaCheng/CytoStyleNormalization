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


def get_gm(color_dirlist, ith):
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

if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = '6'
    batch = 'tongji-szsq'
    name = 'NormNet_tongji-szsqTOsfy-3d1_L1_100_d1_1.0_d2_1_task_0'
    txt_dir = '../dataset/DA_dataset_dirlist_K/%s_test.txt'%batch
    save_dir = 'h:/cxh/NormalizationNet/result_formal/%s'%name
    model_dir = './ckpt/%s/NormNet-'%name
    dirlist = get_dirlist(txt_dir)
    batch_size = 10
    # load img
    print("Loading img")
    gms = get_color_gm_Multiprocess(dirlist)
    # Generator setting
    gm_tensor = tf.placeholder(tf.float32,shape=[batch_size,512,512,4])
    color_tensor = generator(gm_tensor,o_c=3,gf_dim=32,training=False)
    sess = tf.Session()
    saver = tf.train.Saver()
    for epoch in range(0,15):
        saver.restore(sess,model_dir+str(epoch))
        _save_dir = os.path.join(save_dir,'epoch%d'%epoch)
        print("Normlizing img")
        colors = [] 
        for i in range(0,gms.shape[0],batch_size):
            gm = gms[i:i+batch_size,:,:,:]
            color = sess.run(color_tensor,feed_dict={gm_tensor:gm})
            color = inverse_transform(color)
            colors.append(color)
        colors = np.concatenate(colors,0)
        print('Saving transed imgs')
        for i, img in enumerate(colors):
            _dir = dirchage(dirlist[i],_save_dir)
            cv2.imwrite(_dir,img[:,:,::-1])

    
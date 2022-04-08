# -*- coding: utf-8 -*-
"""
Created on Wed May 15 23:55:20 2019

@author: Melon
"""

import tensorflow as tf
import cv2
import numpy as np
import random
import multiprocessing.dummy as multiprocessing
import scipy.ndimage as ndi
import skimage
from skimage import  morphology

def gamma_trans(img,gamma):                                             
    #gamma建议范围0.5-1.5
    #具体做法先归一化到1，然后gamma作为指数值求出新的像素值再还原
    gamma_table = [np.power(x/255.0,gamma)*255.0 for x in range(256)]   
    #如果gamma>1, 新图像比原图像暗,如果gamma<1,新图像比原图像亮
    gamma_table = np.round(np.array(gamma_table)).astype(np.uint8)
    return cv2.LUT(img,gamma_table)

def HSV_trans(img, h_change=0, s_change=1, v_change=1): ##hsv变换
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    hsv =  np.float64(hsv)
    if h_change != 0: #random.random()
        k = random.random()*0.1 + 0.95 #0.95-1.05
        b = random.random()*6 - 3 #-3/3
        hsv[...,0] = k*hsv[...,0] + b
        hsv[...,0][ hsv[...,0] <= 0] = 0
        hsv[...,0][ hsv[...,0] >= 180] = 180
    if  s_change != 0:
        k =  random.random() + 0.7#0.7-1.7
        b = random.random()*20 - 10#-10/10
        hsv[...,1] = k*hsv[...,1] + b
        hsv[...,1][ hsv[...,1] <= 0] = 1
        hsv[...,1][ hsv[...,1] >= 255] = 255
    if  v_change != 0:
        k = random.random()*0.45 + 0.75#0.75-1.2
        b = random.random()*18 - 10#-10-8
        hsv[...,2] = k*hsv[...,2] + b
        hsv[...,2][ hsv[...,2] <= 0] = 1
        hsv[...,2][ hsv[...,2] >= 255] = 255
    hsv = np.uint8(hsv)
    img_new = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
    return img_new

def segImg_v2(img, threshold):
    img_front, img_front_mask = frontground_seg(img.copy(), threshold)
    img_lab = cv2.cvtColor(img,cv2.COLOR_RGB2LAB)
    mask_r = np.expand_dims((img_lab[:,:,1] > 148)*img_front_mask,-1)
    mask_g = np.expand_dims((~(img_lab[:,:,1] > 148))*img_front_mask,-1)
    mask_back = np.expand_dims(~img_front_mask,-1)
    mask = [np.uint8(mask_r)*255,np.uint8(mask_g)*255,np.uint8(mask_back)*0]
    mask = np.concatenate(mask,-1)#R、G、Back
    mask = np.uint8(mask)
    return mask

def frontground_seg(img, threshold):
    # 分割的到图像前景
    # threhold: Our: 28; Sfy1、2： 10
    hole_threshold = 128
    obj_threshold = 1024 
    wj1 = img.max(axis=2)
    wj2 = img.min(axis=2)
    wj3 = wj1 - wj2
    imgBin = wj3 > threshold
    imgBin = morphology.remove_small_holes(imgBin,area_threshold=hole_threshold)
    imgCon, numCon = ndi.label(imgBin)
    imgConBig = morphology.remove_small_objects(imgCon, min_size=obj_threshold)
    imgBin = imgConBig > 0
    img[imgBin==0] = 0
    return img, imgBin

def enhance(img):
    img_hsv = HSV_trans(img.copy(), h_change=0, s_change=1, v_change=1)#BGR2HSV
    # gamma 0.6-1.4
    img_gamma = gamma_trans(img_hsv.copy(), random.random()*0.8 + 0.6)
    return img_gamma

def _get_color_gm(color_dirlist, ith):#挖去背景，图像增强
    color_dir = color_dirlist[ith]
    color = cv2.imread(color_dir)[:,:,::-1]
    #mask = cv2.imread(color_dir.replace('_color','_mask_encode_mask_lab_148'))[:,:,::-1]
    mask = segImg_v2(color.copy(), 10)
    color_enhanced = enhance(color.copy())
    gray_enhanced_front = np.expand_dims(cv2.cvtColor(color_enhanced,cv2.COLOR_RGB2GRAY),-1)
    color_gm = np.concatenate([color,gray_enhanced_front,mask],-1)
    color_gm = np.float32((color_gm/255 - 0.5)*2)
    return color_gm

def get_color_gm(dirList,threadNum=5):
    pool = multiprocessing.Pool(threadNum)
    imgTotals = []
    for ith in range(len(dirList)):
        imgTotal = pool.apply_async(_get_color_gm, args=(dirList,ith))
        imgTotals.append(imgTotal)
    pool.close()
    pool.join()
    imgTotal = np.array([x.get() for x in imgTotals])
    return imgTotal

def prepare_labels(dirlist):
    labellist = []
    for s in dirlist:
        if '_ASCUS' in s or '_HISL' in s or '_LISL' in s:
            labellist.append(1.)
        else:
            labellist.append(0.)
    return np.array(labellist)

def read_lines(fname):
    f = open(fname,'r')
    color_list = []
    for line in f:
        s = line.strip('\n')
        color_list.append(s)
    return color_list

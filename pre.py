import cv2
import os
import numpy as np

## 图片处理形式
CHANGE_GRAY_SIN = 0 #转换成单尺寸灰度图像
CHANGE_BGR_SIN = 1 #转换成单尺寸BGR图像
CHANGE_GRAY_MUL = 2 #转换成多尺寸灰度图像
CHANGE_BGR_MUL = 3 #转换成多尺寸BGR图像

## 图片预处理输入后的大小
## 单尺寸预处理时
IMG_PRE_OUT_WIDTH = 244
IMG_PRE_OUT_HEIGHT = 244

## 原始图片存放位置
DIR_LIST = ["CQ","QTQ","TQ","YQ"]


def img_pro(file_name, change_type):
    """图片处理"""
    #将图片均转为float32且除255进行归一化, 若为灰度图像则需expand_dims增加维度
    if change_type == CHANGE_GRAY_SIN:
        img = cv2.imread(file_name,0)
        img = cv2.resize(img,(IMG_PRE_OUT_WIDTH, IMG_PRE_OUT_HEIGHT))
        return np.expand_dims(img.astype(np.float32)/255.0,axis=-1)

    elif change_type == CHANGE_BGR_SIN:
        img = cv2.imread(file_name)
        img = cv2.resize(img,(IMG_PRE_OUT_WIDTH, IMG_PRE_OUT_HEIGHT))
        return img.astype(np.float32)/255.0

    elif change_type == CHANGE_GRAY_MUL:
        img = cv2.imread(file_name,0)
        return np.expand_dims(img.astype(np.float32)/255.0,axis=-1)

    elif change_type == CHANGE_BGR_MUL:
        img = cv2.imread(file_name)
        return img.astype(np.float32)/255.0


def set_group(set_train_scale=0.75):
    """划分训练集和测试集"""
    files_name_train = []
    files_name_test = []
    fp_train = open("files_name_train.txt",'w')
    fp_test = open("files_name_test.txt",'w')
    for dir in DIR_LIST:
        S_files_name_train = []
        S_files_name_test = []
        fp_train.write(dir + "---------------------------------------"+'\n')
        fp_test.write(dir + "---------------------------------------"+'\n')

        files_name = os.listdir('images/figNew/'+dir)
        files_len = len(files_name)
        files_num_train = int(files_len * set_train_scale)
        files_index_train = np.random.choice(files_len, files_num_train, replace=False)
        for i in range(0,files_len):
            #去除cv2读取失败的文件
            img = cv2.imread('images/figNew/'+dir+'/'+files_name[i])
            if img.shape[0] == None or img.shape[1] == None:
                continue
            #添加文件名分别至训练集和数据集
            if i in files_index_train:
                S_files_name_train.append(files_name[i])
                fp_train.write(files_name[i] + '\n')
            else:
                S_files_name_test.append(files_name[i] )
                fp_test.write(files_name[i] + '\n')
       
        # #code test
        # a = np.random.choice(len(S_files_name_train),2,replace=False)
        # b = np.random.choice(len(S_files_name_test),2,replace=False)
        # S_files_name_train = [S_files_name_train[a[0]],S_files_name_train[a[1]]]
        # S_files_name_test = [S_files_name_test[b[0]],S_files_name_test[b[1]]]

        files_name_train.append(S_files_name_train)
        files_name_test.append(S_files_name_test)
    fp_train.close()
    fp_test.close()

    return files_name_train,files_name_test


def img_load(change_type, files_name_train, files_name_test):
    '''加载训练集和测试集'''
    set_train = []
    set_train_lables = []
    set_test = []
    set_test_lables = []
    label = 0 #代表类别

    for dir in DIR_LIST:
        for file_name in files_name_train[label]:
            set_train.append(img_pro('images/figNew/'+dir+'/'+file_name, change_type))
            set_train_lables.append(label)
        for file_name in files_name_test[label]:
            set_test.append(img_pro('images/figNew/'+dir+'/'+file_name, change_type))
            set_test_lables.append(label)

        label = label + 1 #用 0,1,2,3 分别代表四类

    return (set_train,np.array(set_train_lables,np.int32)),(set_test,np.array(set_test_lables,np.int32))


def img_pre(change_type, files_name_train, files_name_test):
    """加载数据集"""
    #在主代码文件中，要实行此函数，须先执行set_group()
    (set_train,set_train_lables),(set_test,set_test_lables) = img_load(change_type, files_name_train, files_name_test)

    if change_type == CHANGE_BGR_SIN or change_type == CHANGE_GRAY_SIN:
        set_train_images = np.array(set_train,np.float32)
        set_test_images = np.array(set_test,np.float32)
    else:
        set_train_images = set_train
        set_test_images = set_test

    return (set_train_images,set_train_lables),(set_test_images,set_test_lables)
#coding:utf-8
import numpy as np
import matplotlib.pyplot as plt
import caffe
import glob,os
import lmdb
import cv2
import caffe
from caffe.proto import caffe_pb2


def get_file_list(dataDir):
    file_list = []
    for root, dirs, files in os.walk(dataDir):
        for file in files:
            if file.endswith(".jpg"):
                 file_list.append(os.path.join(root, file))
    return file_list

def get_pairs_list(prefix,pairtxt):
    text_file = open(pairtxt, "r")
    #lines = text_file.readlines()
    lines = [line.strip('\n') for line in open(pairtxt)]
    newlines = list(filter(None, lines))
    res = [prefix + s for s in newlines]
    return res

def calculate_meanimage(file_list):
    num = len(file_list)
    cnt = 0
    total_value = np.zeros((250,250,3))
    for f in file_list:
        img = cv2.imread(f)
        total_value += img
        cnt += 1
    mean_image = total_value/cnt
    np.save('LFW_mean_image.npy',mean_image)

def BGR_mean(mean_Dir):
    #LFW_mean_image
    mean_image = np.load(mean_Dir)
    mu = mean_image.mean(0).mean(0)
    return mu



def save_to_lmdb(file_list):
    file_length = len(file_list)
    #basic setting
    lmdb_file = 'F:/Notes/Rutgers/2017_spring/DSPadv/project2/lmdb1_lfw'
    batch_size = 200

    # create the leveldb file
   # lmdb_env = lmdb.open(lmdb_file, map_size=int(1e12))
    lmdb_env = lmdb.open(lmdb_file, map_size=int(1e10))
    lmdb_txn = lmdb_env.begin(write=True)
    datum = caffe_pb2.Datum()
    x = 0
    i = 0
    j = 0

    for x in range(0,file_length,2):
        data = np.zeros((250,250,6),dtype=np.uint8)
        img1=cv2.imread(file_list[x])     #1 image
        img2=cv2.imread(file_list[x+1])     #1 image
        data[:,:,0:3] = img1
        data[:,:,3:6] = img2
        data = data.astype('int').transpose(2,0,1)
        i += 2

        label = 0
        if x%4 == 0:
            label = 1

        datum = caffe.io.array_to_datum(data, label)   #将数据以及标签整合为一个数据项

        keystr = '{:0>8d}'.format(x-1).encode('UTF-8')                #lmdb的每一个数据都是由键值对构成的，因此生成一个用递增顺序排列的定长唯一的key
        # keystr = '{:0>8d}'.format(x-1).encode()
        #keystr = '{:0>8d}'.format(x-1)
        print(keystr)
        lmdb_txn.put( keystr, datum.SerializeToString())#调用句柄，写入内存。

        # write batch
        if i == batch_size:
            j += 1
            lmdb_txn.commit()
            lmdb_txn = lmdb_env.begin(write=True)
            print (j*i, 'images have been written into lmdb')
            i = 0

# file_path = "F:/Notes/Rutgers/2017_spring/DSPadv/project2/lfw-funneled/lfw_funneled"
# filelist = get_file_list(file_path)
# calculate_meanimage(filelist)

# pair_path = "F:/Notes/Rutgers/2017_spring/DSPadv/project2/lfw-funneled/lfw_funneled/pairs_01.txt"
# prefix = "F:/Notes/Rutgers/2017_spring/DSPadv/project2/lfw-funneled/lfw_funneled/"
# pairlist = get_pairs_list(prefix,pair_path)
# print(1)
#
# save_to_lmdb(pairlist)






import numpy as np
import os
import pickle
CIFARDIR ='../cifar-10-python/cifar-10-batches-py/'
print(os.listdir(CIFARDIR))
##
'''
 ## pip install matplotlib
 ## pip install numpy
 ## pip install tensorflow 

数据集下载 https://www.cs.toronto.edu/~kriz/cifar.html

'''


###test   cifar-10-python中的数据
with open(os.path.join(CIFARDIR,"data_batch_1"),'rb') as f:
        data=pickle.load(f, encoding='bytes')
        print(type(data))
        print(data.keys())
        print(data.keys())
        print(data.keys())
        print(data.keys())
        print(data.keys())
        #print(data)
        print (type(data[b'data']))
        print(type(data[b'labels'])) 
        print (type(data[b'batch_label']))
        print (type(data[b'filenames']))
        print (data[b'data'].shape )   ##32*32=1024 *3=3072 RR-GG-BB
        print (data[b'data'][0:2])  ## 像素点
        print (data[b'labels'][0:2]) ##文件类别
        print (data[b'batch_label']) ##文件所属batch
        print (data[b'filenames'][0:2]) ##文件名字


        image_arr= data[b'data'][100]
        image_arr=image_arr.reshape((3,32,32))# 32 32 3   image arr1
        image_arr = image_arr.transpose((1,2,0))
        import matplotlib.pyplot as plt  
        from matplotlib.pyplot import imshow ##%matplotlib inline
        imshow(image_arr)
        plt.imshow(image_arr) # 对图片image进行数据处理
        plt.show() # 将图片显示出来


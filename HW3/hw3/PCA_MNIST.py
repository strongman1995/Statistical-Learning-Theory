
# coding: utf-8

# In[25]:

import tensorflow.examples.tutorials.mnist.input_data as input_data
import numpy as np
from PIL import Image
from matplotlib.pyplot import imshow
import matplotlib.pyplot as plt
get_ipython().magic(u'matplotlib inline')
# from IPython.display import Image 


# In[2]:

mnist = input_data.read_data_sets("MNIST_data/", one_hot=False)
imgs = mnist.train.images
labels = mnist.train.labels


# In[3]:

print(type(imgs))             # <type 'numpy.ndarray'>
print(type(labels))           # <type 'numpy.ndarray'>
print(imgs.shape)             # (55000, 784)
print(labels.shape)           # (55000,)


# In[4]:

def get_digit_imgs(digit):
    origin_imgs = []
    for i in range(len(imgs)):
        if labels[i] == digit:
            origin_imgs.append(imgs[i])
    return origin_imgs


# In[5]:

origin_imgs_8 = get_digit_imgs(8)
origin_imgs_2 = get_digit_imgs(2)
print(np.array(origin_imgs_8).shape)
print(np.array(origin_imgs_2).shape)


# 由于一张图片是一个784维的一维数组，变成我们想看的图片就需要把它reshape成28x28的二维数组。
# 
# 由于tensorflow中MNIST都是灰度图（L），所以shape是（55000，784），每张图的dtype是float32，如果是彩色图（RGB），shape可能是（55000，784，3），图的dtype是uint8，从array转到Image需要用下面的方法：

# In[6]:

def plot(recon_img_list):
    fig, axes = plt.subplots(1, 4, figsize=(8, 8))
    # centering
    axes[0].imshow(np.array(recon_img_list[0].reshape(28,28), dtype=float), 'gray')
    axes[0].set_title("Original")
    axes[1].imshow(np.array(recon_img_list[1].reshape(28,28), dtype=float), 'gray')
    axes[1].set_title("90%")
    axes[2].imshow(np.array(recon_img_list[2].reshape(28,28), dtype=float), 'gray')
    axes[2].set_title("60%")
    axes[3].imshow(np.array(recon_img_list[3].reshape(28,28), dtype=float), 'gray')
    axes[3].set_title("30%")

    fig, axes_ = plt.subplots(1, 4, figsize=(8, 8))
    # without centering
    axes_[0].imshow(np.array(recon_img_list[4].reshape(28,28), dtype=float), 'gray')
    axes_[0].set_title("Original")
    axes_[1].imshow(np.array(recon_img_list[5].reshape(28,28), dtype=float), 'gray')
    axes_[1].set_title("90%")
    axes_[2].imshow(np.array(recon_img_list[6].reshape(28,28), dtype=float), 'gray')
    axes_[2].set_title("60%")
    axes_[3].imshow(np.array(recon_img_list[7].reshape(28,28), dtype=float), 'gray')
    axes_[3].set_title("30%")
    plt.show()


# In[216]:

def eigValPct(eigVals, percentage):
    '''
    通过方差的百分比来计算将数据降到多少维是比较合适的
        eigVals:特征值
        percentage:百分比
    return: 需要降到的维度数num
    '''
    sortArray = np.sort(eigVals)  # 使用numpy中的sort()对特征值按照从小到大排序
    sortArray = sortArray[-1::-1]  # 特征值从大到小排序
    arraySum = sum(sortArray)  # 数据全部的方差arraySum
    tempSum = 0
    num = 0
    for i in sortArray:
        tempSum += i
        num += 1
        if tempSum >= arraySum*percentage:
            return num

def pca(dataMat, percentage=0.9, centering=True):
    '''
    dataMat:已经转换成矩阵matrix形式的数据集，列表示特征；
    percentage:取前多少个特征需要达到的方差占比，默认为0.9
    '''
    # 每个向量同时都减去 均值
    meanVals = dataMat.mean(axis=0) # shape:(784,)
    meanRemoved = dataMat-meanVals
#     dataMat_ = dataMat if centering else dataMat+meanVals # shape:(100, 784)
#     dataMat_ = dataMat-meanVals if centering else dataMat # shape:(100, 784)

    # cov()计算协方差矩阵
    covMat = np.cov(dataMat, rowvar=0) if centering else (1.0/dataMat.shape[0])*dataMat.T.dot(dataMat) # shape：(784, 784)

    # 计算特征值(Find eigenvalues and eigenvectors)
    # 利用numpy中寻找特征值和特征向量的模块linalg中的eig()方法
    eigVals, eigVects = np.linalg.eig(np.mat(covMat))
    
    # 要达到方差的百分比percentage，需要前k个向量
    k = eigValPct(eigVals, percentage)
    
    eigValInd = np.argsort(eigVals)  # 对特征值eigVals从小到大排序
    
    eigValInd = eigValInd[:-(k+1):-1]  # 从排好序的特征值，从后往前取k个，这样就实现了特征值的从大到小排列
    
    redEigVects = eigVects[:, eigValInd]  # 返回排序后特征值对应的特征向量redEigVects（主成分）
    
    lowDDataMat = meanRemoved*redEigVects  # 将原始数据投影到主成分上得到新的低维数据lowDDataMat
    
    reconMat = (lowDDataMat*redEigVects.T)+meanVals  # 得到重构数据reconMat
    
    return lowDDataMat, reconMat

def get_recon(imgs, index):
    """
    输入：
        imgs：图像矩阵
        index：图像索引
    返回：不同参数下重构的索引图像
    """
    recon_img_list = []
    for centering in [True, False]:
        recon_img_list.append(imgs[index])
        for percent in [0.9,0.6,0.3]:
            # 调用PCA进行降维
            low_d_feat_for_imgs, recon_mat_for_imgs = pca(np.array(imgs), percent, centering)
            recon_img_list.append(np.asarray(recon_mat_for_imgs[index]).reshape(-1))
            print("Centering data:{} percent:{} low_shape:{} recon_shape:{}"                   .format(centering, percent, low_d_feat_for_imgs.shape, recon_mat_for_imgs.shape))
    return recon_img_list


# In[217]:

recon_img_list_0 = get_recon(imgs,4)


# In[218]:

recon_img_list_1 = get_recon(imgs,5)


# In[219]:

plot(recon_img_list_0)


# In[220]:

plot(recon_img_list_1)


# In[ ]:




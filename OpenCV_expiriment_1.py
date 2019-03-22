import cv2 
import torch 
import numpy as np 
import matplotlib.pyplot as plt
import pandas as pd 
import csv
import math
from math import sqrt
from numpy import exp

'''
类ImInit
这是一个用来导入、灰度处理和输出、打印图像用的类

self初始变量
self.imagepath         string                     导入图片路径  
self.color             cv2下属属性                 选用合适的颜色属性（如灰度）
self.name              string                     图片名字
self.author            string                     作者名字（Jing）
函数
image_file(self):      导入图像、提取图像灰度矩阵用
image_print(self,x):   打印图像，其中x=0时打印灰度图，x=1时打印原始彩图；都有延时判断

作者：井一韬（Jing）  
'''
class ImInit:
    def __init__(self,imagepath,color,name,author):              #初始函数用来初始换类里面的固有的属性和方法
        self.imagepath = imagepath                               #self参数似乎是伴随整个类的指向自己参数
        self.color = color
        self.name = name
        self.author = author
        
    def image_file(self):                                        #定义在类里面的方法（函数）imagepath = "E:\\OpenCV\\image\\Sherry_1.JPG"
        self.image = cv2.imread(self.imagepath)
        self.gray = cv2.cvtColor(self.image,self.color)
        self.rows,self.cols = self.gray.shape
    def image_print(self,x):
        if x==0 :
            cv2.imshow(self.name,self.gray)
            cv2.waitKey(0)
        elif x==1 :
            cv2.imshow(self.name,self.image)
            cv2.waitKey(0)


#实例1
ImSherry = ImInit(imagepath = "E:\\OpenCV\\image\\Sherry.PNG",color = cv2.COLOR_BGR2GRAY,name = "Sherry",author = "Jing")
ImSherry.image_file()                                               #这一张实验用的名字为 "Sherry" 的照片大小 129*143 像素
plt.subplot(1, 2, 1)
plt.imshow(ImSherry.gray,'gray')                                    #使用plt包输出图像，属性 "gray" 灰度Rivers.JPG
#plt.show()


'''
print(np.max(ImSherry.gray))
print(np.min(ImSherry.gray))
                                            
print(ImSherry.gray)                                                #输出灰度矩阵COLOR_BGR2GRAY  
print(ImSherry.gray.shape)                                          #输出图像大小 143*129
print(ImSherry.rows)
print(ImSherry.cols)
'''
'''           
#实例2
ImMiracle = ImInit(imagepath = "E:\\OpenCV\\image\\Miracle_1.JPG",color = cv2.COLOR_BGR2GRAY,name = "Miracle",author = "Jing")
ImMiracle.image_file()                                               #这一张实验用的名字为 "Miracle" 
ImMiracle.image_print(0)                                             #输出延缓显示
print(ImMiracle.gray)                                                #输出灰度矩阵COLOR_BGR2GRAY  
print(ImMiracle.gray.shape)                                          #输出图像大小 1440*1080
'''
'''
类FFT：
这是一个用来快速傅里叶变化的类，将图像快速进行频域转化

self初始变量：
self.image :             matrix                           图像矩阵输入
self.name :              string                           图像名称
self.author :            string                           作者（Jing）
函数：
FFTchange(self,x):        快速傅里叶变化用，同时将图像中心变化，将图像归一化，取整数化；x=0时取幅值，x=1时取相位,x=3时直接取傅里叶变化后的值

作者：井一韬（Jing）
'''
class FFT:
    def __init__(self,image,name,author):                           #初始函数用来初始换类里面的固有的属性和方法
        self.image = image                                          #self参数似乎是伴随整个类的指向自己参数
        self.name = name
        self.author = author
    def FFTchange(self,x):
        self.FFT = np.fft.fft2(self.image)                          #对图像灰度进行快速傅里叶变换（FFT）
        self.FFTimage = np.fft.fftshift(self.FFT)                   #对变换后的图像，将DC分量（其实就是F(0,0)点，低频中心）移到矩阵的中心        
        if x==0 :
            self.FFTimage = np.log2(np.abs(self.FFTimage))          #对变换后的图像，log化缩小其范围，方便显示处理，这里取得的是频域幅值的图
            self.FFTimage = self.FFTimage.astype(int)               #如果要显示的话，图像变量类型int化
        if x==1 :
            self.FFTimage = np.angle(self.FFTimage)                 #对变换后的图像，angle取其相角方向，这样的话我们取得的是相位图
            self.FFTimage = self.FFTimage.astype(int)               #如果要显示的话，图像变量类型int化
        if x==3 :
            self.FFTimage = np.fft.fftshift(self.FFT)


#实例
FFTSherry = FFT(image = ImSherry.gray,name = "FFTSherry",author = "Jing")
FFTSherry.FFTchange(1)                                               #使用之前的实例1的Sherry的图像进行变换实验
'''
plt.imshow(FFTSherry.FFTimage,'gray')                                #使用plt包输出图像，属性 "gray" 灰度
plt.show()
print(FFTSherry.FFTimage)                                            #显示图像矩阵
print(np.max(FFTSherry.FFTimage))
print(np.min(FFTSherry.FFTimage))
'''

'''
类IFFT：
这是一个用来快速反傅里叶变化的类，将图像快速进行时域转化

self初始变量：
self.ifftimage :         matrix                           频域图像矩阵输入，可以直接输入FFT类的x=3图像
self.name :              string                           图像名称
self.author :            string                           作者（Jing）
函数：
IFFTchange(self):        快速反傅里叶变化用，同时将图像中心变化回去

作者：井一韬（Jing）
'''
class IFFT:
    def __init__(self,ifftimage,name,author):                         #初始函数用来初始换类里面的固有的属性和方法
        self.ifftimage = ifftimage                                    #self参数似乎是伴随整个类的指向自己参数
        self.name = name
        self.author = author
    def IFFTchange(self):                        
        self.IFFT = np.fft.ifftshift(self.ifftimage)                  #对变换后的图像，将DC分量（其实就是F(0,0)点，低频中心）移到矩阵的中心
        self.IFFTimage = np.fft.ifft2(self.IFFT)                      #对图像灰度进行快速反傅里叶变换（IFFT）        
        self.IFFTimage = np.abs(self.IFFTimage)

'''
FFTSherry.FFTchange(3)
IFFTSherry = IFFT(ifftimage = FFTSherry.FFTimage,name = "IFFTSherry",author = "Jing")
IFFTSherry.IFFTchange()
plt.imshow(IFFTSherry.IFFTimage,'gray')                               #使用plt包输出图像，属性 "gray" 灰度
plt.show()
'''

'''
函数 distance：
这是一个计算两个点欧式距离的函数

参数：数组arry x,y
返回值：dis 距离

作者：井一韬（Jing）
'''
def distance(x,y):
    dis = sqrt((x[0]-y[0])**2+(x[1]-y[1])**2)
    return dis

'''
类IHPF：
这是一个用来做理想高通滤波器的类，用来进行高通滤波

self初始变量：
self.shape :             2 values                         频域图像矩阵shape大小输入，cols rows
self.D0 :                int                              滤波的范围
self.name :              string                           图像名称
self.author :            string                           作者（Jing）
函数：
IHPFchange(self):        高通滤波函数，生成滤波矩阵mask

作者：井一韬（Jing）
'''
class IHPF:
    def __init__(self,shape,D0,name,author):
        self.shape = shape
        self.D0 = D0
        self.name = name
        self.author = author
    def IHPFchange(self):
        self.rows,self.cols = self.shape
        self.mask = np.ones(self.shape,np.uint8)
        for i in range(self.mask.shape[0]):
            for j in range(self.mask.shape[1]):
                dis = distance((int(self.rows/2),int(self.cols/2)),(i,j))
                if dis <= self.D0 :                                    #当距离小于dis的时候，我们使值等于0
                    self.mask[i,j] = 0                                 #self.mask[int(self.rows/2-self.D0):int(self.rows/2+self.D0),int(self.cols/2-self.D0):int(self.cols/2+self.D0)] = 1

'''
类GHPF：
这是一个用来做高斯高通滤波器的类，用来进行高通滤波

self初始变量：
self.shape :             2 values                         频域图像矩阵shape大小输入，cols rows
self.D0 :                int                              滤波的参数
self.name :              string                           图像名称
self.author :            string                           作者（Jing）
函数：
GHPFchange(self):        高通滤波函数，生成滤波矩阵mask

作者：井一韬（Jing）
'''
class GHPF:
    def __init__(self,shape,D0,name,author):
        self.shape = shape
        self.D0 = D0
        self.name = name
        self.author = author
    def GHPFchange(self):
        self.rows,self.cols = self.shape
        self.mask = np.zeros(self.shape)                                 #,np.uint8 这里要注意数据格式float
        for i in range(self.mask.shape[0]):
            for j in range(self.mask.shape[1]):
                self.dis = distance((int(self.rows/2),int(self.cols/2)),(i,j))
                self.mask[i,j] = 1 - np.exp(-(self.dis**2)/(2*(self.D0**2)))#这里根据高斯滤波器的公式进行运算

'''
#理想高通滤波器 实例
FFTSherry.FFTchange(3)
IHPFSherry = IHPF(shape = FFTSherry.FFTimage.shape,D0 = 40,name = "IHPFSherry",author = "Jing")
IHPFSherry.IHPFchange()
print(np.max(IHPFSherry.mask))
FFTSherry.FFTimage = FFTSherry.FFTimage * IHPFSherry.mask
IFFTSherry = IFFT(ifftimage = FFTSherry.FFTimage,name = "IFFTSherry",author = "Jing")
IFFTSherry.IFFTchange()
plt.imshow(IFFTSherry.IFFTimage,'gray')                    #使用plt包输出图像，属性 "gray" 灰度
plt.show()
print(np.max(IFFTSherry.IFFTimage))
'''
'''
#高斯高通滤波器实例
FFTSherry.FFTchange(3)
GHPFSherry = GHPF(shape = FFTSherry.FFTimage.shape,D0 = 80,name = "GHPFSherry",author = "Jing")
GHPFSherry.GHPFchange()
FFTSherry.FFTimage =FFTSherry.FFTimage + FFTSherry.FFTimage * GHPFSherry.mask
IFFTSherry = IFFT(ifftimage = FFTSherry.FFTimage,name = "IFFTSherry",author = "Jing")
IFFTSherry.IFFTchange()
plt.imshow(IFFTSherry.IFFTimage,'gray')                    #使用plt包输出图像，属性 "gray" 灰度
plt.show()
'''

'''
类BHPF：
这是一个用来做巴特沃斯高通滤波器的类，用来进行高通滤波

self初始变量：
self.shape :             2 values                         频域图像矩阵shape大小输入，cols rows
self.D0 :                int                              滤波的参数
self.n                   int                              巴特沃斯滤波器的参数
self.name :              string                           图像名称
self.author :            string                           作者（Jing）
函数：
BHPFchange(self):        高通滤波函数，生成滤波矩阵mask

作者：井一韬（Jing）
'''
class BHPF:
    def __init__(self,shape,D0,n,name,author):
        self.shape = shape
        self.D0 = D0
        self.n = n
        self.name = name
        self.author = author
    def BHPFchange(self):
        self.rows,self.cols = self.shape
        self.mask = np.zeros(self.shape)                                  #,np.uint8 这里要注意数据格式float
        for i in range(self.mask.shape[0]):
            for j in range(self.mask.shape[1]):
                self.dis = distance(((self.rows/2),(self.cols/2)),(i,j))   
                self.mask[i,j] = 1 - 1/((1+(self.D0/self.dis))**(2*self.n))#这里根据巴特沃斯滤波器进行运算                             

'''
#巴特沃斯高通滤波器实例
FFTSherry.FFTchange(3)
BHPFSherry = BHPF(shape = FFTSherry.FFTimage.shape,D0 = 50,n = 2,name = "BHPFSherry",author = "Jing")
BHPFSherry.BHPFchange()
FFTSherry.FFTimage = FFTSherry.FFTimage * BHPFSherry.mask + FFTSherry.FFTimage
IFFTSherry = IFFT(ifftimage = FFTSherry.FFTimage,name = "IFFTSherry",author = "Jing")
IFFTSherry.IFFTchange()
plt.imshow(IFFTSherry.IFFTimage,'gray')                    #使用plt包输出图像，属性 "gray" 灰度
plt.show()
'''

'''
类ILPF：
这是一个用来做理想低通滤波器的类，用来进行高通滤波

self初始变量：
self.shape :             2 values                         频域图像矩阵shape大小输入，cols rows
self.D0 :                int                              滤波的范围
self.name :              string                           图像名称
self.author :            string                           作者（Jing）
函数：
ILPFchange(self):        高通滤波函数，生成滤波矩阵mask

作者：井一韬（Jing）
'''
class ILPF:
    def __init__(self,shape,D0,name,author):
        self.shape = shape
        self.D0 = D0
        self.name = name
        self.author = author
    def ILPFchange(self):
        self.rows,self.cols = self.shape
        self.mask = np.zeros(self.shape,np.uint8)
        for i in range(self.mask.shape[0]):
            for j in range(self.mask.shape[1]):
                dis = distance((int(self.rows/2),int(self.cols/2)),(i,j))
                if dis <= self.D0 :
                    self.mask[i,j] = 1                     

'''
类GLPF：
这是一个用来做高斯低通滤波器的类，用来进行高通滤波

self初始变量：
self.shape :             2 values                         频域图像矩阵shape大小输入，cols rows
self.D0 :                int                              滤波的参数
self.name :              string                           图像名称
self.author :            string                           作者（Jing）
函数：
GLPFchange(self):        低通滤波函数，生成滤波矩阵mask

作者：井一韬（Jing）
'''
class GLPF:
    def __init__(self,shape,D0,name,author):
        self.shape = shape
        self.D0 = D0
        self.name = name
        self.author = author
    def GLPFchange(self):
        self.rows,self.cols = self.shape
        self.mask = np.zeros(self.shape)                   #,np.uint8
        for i in range(self.mask.shape[0]):
            for j in range(self.mask.shape[1]):
                self.dis = distance((int(self.rows/2),int(self.cols/2)),(i,j))
                self.mask[i,j] = np.exp(-(self.dis**2)/(2*(self.D0**2)))

'''
类BLPF：
这是一个用来做巴特沃斯低通滤波器的类，用来进行低通滤波

self初始变量：
self.shape :             2 values                         频域图像矩阵shape大小输入，cols rows
self.D0 :                int                              滤波的参数
self.n                   int                              巴特沃斯低通滤波器参数
self.name :              string                           图像名称
self.author :            string                           作者（Jing）
函数：
GHPFchange(self):        低通滤波函数，生成滤波矩阵mask

作者：井一韬（Jing）
'''
class BLPF:
    def __init__(self,shape,D0,n,name,author):
        self.shape = shape
        self.D0 = D0
        self.n = n
        self.name = name
        self.author = author
    def BLPFchange(self):
        self.rows,self.cols = self.shape
        self.mask = np.zeros(self.shape)                   #,np.uint8
        for i in range(self.mask.shape[0]):
            for j in range(self.mask.shape[1]):
                self.dis = distance(((self.rows/2),(self.cols/2)),(i,j))  #sqrt((int(self.rows/2)-i)**2+(int(self.cols/2)-j)**2) 
                self.mask[i,j] = 1/((1+(self.D0/self.dis))**(2*self.n)) 

'''
#巴特沃斯低通滤波器实例
FFTSherry.FFTchange(3)
BLPFSherry = BLPF(shape = FFTSherry.FFTimage.shape,D0 = 10,n = 2,name = "BLPFSherry",author = "Jing")
BLPFSherry.BLPFchange()
FFTSherry.FFTimage = FFTSherry.FFTimage * BLPFSherry.mask + FFTSherry.FFTimage
IFFTSherry = IFFT(ifftimage = FFTSherry.FFTimage,name = "IFFTSherry",author = "Jing")
IFFTSherry.IFFTchange()
plt.subplot(1, 2, 2)
plt.imshow(IFFTSherry.IFFTimage,'gray')                    #使用plt包输出图像，属性 "gray" 灰度
plt.show()
'''

FFTSherry.FFTchange(3)
GLPFSherry = GLPF(shape = FFTSherry.FFTimage.shape,D0 = 10,name = "GLPFSherry",author = "Jing")
GLPFSherry.GLPFchange()
FFTSherry.FFTimage = FFTSherry.FFTimage * GLPFSherry.mask + FFTSherry.FFTimage
IFFTSherry = IFFT(ifftimage = FFTSherry.FFTimage,name = "IFFTSherry",author = "Jing")
IFFTSherry.IFFTchange()
plt.subplot(1, 2, 2)
plt.imshow(IFFTSherry.IFFTimage,'gray')                    #使用plt包输出图像，属性 "gray" 灰度
plt.show()


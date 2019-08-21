# -*- coding: utf-8 -*-
"""
Created on Fri Jul 19 18:34:25 2019

@author: Liu qi
"""
import os
os.chdir('D:/smu mitb curriculum/machine learning/project')
import cv2
from tqdm import tqdm
import numpy as np

#logistics:100*100
#k-adjacent:200*200
#cnn:150*150
Train_dir="chest_xray/train/"
Test_dir="chest_xray/test/"
Val_dir="chest_xray/val/"

#logistics regression 
def get_data(Dir,h,w):
    #X = []
    size=h*w
    X = np.zeros((0,size))
    y = []
    for nextDir in os.listdir(Dir):
        if not nextDir.startswith('.'):
            if nextDir in ['NORMAL']:
                label = 0
            elif nextDir in ['PNEUMONIA']:
                label = 1
            else:
                label = 2
                
            temp = Dir + nextDir
                
            for file in tqdm(os.listdir(temp)):
                img = cv2.imread(temp + '/' + file,cv2.IMREAD_GRAYSCALE)
                if img is not None:
                    img = cv2.resize(img, (h, w),interpolation=cv2.INTER_CUBIC)
                    #img = np.asarray(img)
                    img =img.reshape(1,size)
                    X = np.vstack((X,img))
                    #X.append(img)
                    y.append(label)
                    
    #X =np.asarray(X)
    #X = np.vstack((X,img))
    y = np.asarray(y)
    return X,y
h,w = 100,100
size=h*w
X_train, y_train = get_data(Train_dir,h,w)
X_test, y_test = get_data(Test_dir,h,w)

#数据增广_rotate
def im_rotate(img,angle,center,scale):
    M = cv2.getRotationMatrix2D(center,angle,scale)
    img = cv2.warpAffine(img,M,(w,h))
    return img

np.random.seed(2019)
row_num=X_train.shape[0]
use_index = np.random.choice([True, False], row_num, replace = True, p = [0.5, 0.5])
#img=X_train[1]
#img =img.reshape(150,150)
center = (h/2,w/2)
scale = 1.0
angle = np.random.randint(1, 360)
X_ro=np.zeros((0,size))
y_ro = []
for i in tqdm(range(row_num)):
    if use_index[i]:
        img=X_train[i]
        img =img.reshape(h,w)
        img=im_rotate(img,angle,center,scale)
        img =img.reshape(1,size)
        X_ro=np.vstack((X_ro,img))
        y_ro.append(y_train[i])
y_ro = np.asarray(y_ro)

#数据增广_crop
def randomCrop(img):
    crop_seed = np.random.randint(0, 9)
    img=img[crop_seed:crop_seed+h,crop_seed:crop_seed+w]
    return img
X_cr=np.zeros((0,size))
y_cr = []
for i in tqdm(range(row_num)):
    if use_index[i]:
        img=X_train[i]
        img =img.reshape(h,w)
        img = cv2.resize(img, (h+10, w+10),interpolation=cv2.INTER_CUBIC)
        img=randomCrop(img)
        img =img.reshape(1,size)
        X_cr=np.vstack((X_cr,img))
        y_cr.append(y_train[i])
y_cr = np.asarray(y_cr)

#数据增广_Gaussian Noise
import random
def gaussianNoisy(img, mean=0.2, sigma=0.3):
    for j in range(len(img)):
        img[j] += random.gauss(mean, sigma)
        if img[j]<0:
            img[j]=0
        elif img[j]>255:
            img[j]=255
    return img
X_gn=np.zeros((0,size))
y_gn = []
for i in tqdm(range(row_num)):
    if use_index[i]:
        img=X_train[i]
        img=gaussianNoisy(img, mean=0.2, sigma=0.3)
        X_gn=np.vstack((X_gn,img))
        y_gn.append(y_train[i])
y_gn = np.asarray(y_gn)

#合并
X_train=np.vstack((X_train,X_ro))
X_train=np.vstack((X_train,X_cr))
X_train=np.vstack((X_train,X_gn))

y_train=np.hstack((y_train,y_ro))
y_train=np.hstack((y_train,y_cr))
y_train=np.hstack((y_train,y_gn))
#logistics regression 
# transform to 0~1
from sklearn import preprocessing
min_max_scaler = preprocessing.MinMaxScaler()
X_train = min_max_scaler.fit_transform(X_train)
X_test = min_max_scaler.transform(X_test)

from sklearn import linear_model
from sklearn import metrics
def print_score(y_true, y_pred):
    print('  mitrics : ', metrics.confusion_matrix(y_true, y_pred))
    print(' accuracy : ', metrics.accuracy_score(y_true, y_pred))
    print('precision : ', metrics.precision_score(y_true, y_pred))
    print('   recall : ', metrics.recall_score(y_true, y_pred))
    print('       F1 : ', metrics.f1_score(y_true, y_pred))

lr=linear_model.LogisticRegression(solver = 'liblinear',C=0.1)
lr.fit(X_train,y_train) 
y_train_pred=lr.predict(X_train) 
y_test_pred =lr.predict(X_test)
print_score(y_train,y_train_pred)
print_score(y_test,y_test_pred)

#k-adjacent 
class NearestNeighbor(object):
  def __init__(self):
    pass

  def train(self, X, y):
    #X是NXD的数组，其中每一行代表一个样本，Y是N行的一维数组，对应X的标签
    # 最近邻分类器就是简单的记住所有的数据
    self.Xtr = X
    self.ytr = y

  def predict(self, X, k):
    #X是NXD的数组，其中每一行代表一个图片样本
    #看一下测试数据有多少行
    num_test = X.shape[0]
    # 确认输出的结果类型符合输入的类型
    Ypred = np.zeros(num_test, dtype = self.ytr.dtype)

    # 循环每一行，也就是每一个样本
    '''for i in range(num_test):
      # 找到和第i个测试图片距离最近的训练图片
      # 计算他们的L1距离
      distances = np.sum(np.abs(self.Xtr - X[i,:]), axis = 1)
      #distances = np.sqrt(np.sum(np.square(self.Xtr - X[i,:]), axis = 1))#L2
      min_index = np.argmin(distances) # 拿到最小那个距离的索引
      Ypred[i] = self.ytr[min_index] # 预测样本的标签，其实就是跟他最近的训练数据样本的标签'''
    for i in range(num_test):
        y_vote = np.zeros((k ,1), dtype = self.ytr.dtype)
        distances = np.sum(np.abs(self.Xtr - X[i,:]), axis = 1)
        #distances = np.sqrt(np.sum(np.square(self.Xtr - X[i,:]), axis = 1))#L2
        for j in range(k):
            distances[np.argmin(distances)] = np.max(distances)
            min_index = np.argmin(distances)
            y_vote[j]=self.ytr[min_index]
        Ypred[i] = round(np.mean(y_vote))   
    return Ypred

h,w = 200,200
size=h*w
X_train, y_train = get_data(Train_dir,h,w)
X_test, y_test = get_data(Test_dir,h,w)

nn = NearestNeighbor() # 创建一个最近邻分类器的类，相当于初始化
nn.train(X_train, y_train) # 把训练数据给模型，训练
y_predict = nn.predict(X_test,5) # 预测测试集的标签
# 算一下分类的准确率，这里取的是平均值
print ('accuracy: %f' % ( np.mean(y_predict == y_test) ))
from sklearn import metrics
def print_score(y_true, y_pred):
    print('  mitrics : ', metrics.confusion_matrix(y_true, y_pred))
    print(' accuracy : ', metrics.accuracy_score(y_true, y_pred))
    print('precision : ', metrics.precision_score(y_true, y_pred))
    print('   recall : ', metrics.recall_score(y_true, y_pred))
    print('       F1 : ', metrics.f1_score(y_true, y_pred))
print_score(y_test,y_predict)


#cnn
h,w = 150,150
size=h*w
def get_data(Dir,h,w):
    X = []
    size=h*w
    #X = np.zeros((0,size))
    y = []
    for nextDir in os.listdir(Dir):
        if not nextDir.startswith('.'):
            if nextDir in ['NORMAL']:
                label = 0
            elif nextDir in ['PNEUMONIA']:
                label = 1
            else:
                label = 2
                
            temp = Dir + nextDir
                
            for file in tqdm(os.listdir(temp)):
                img = cv2.imread(temp + '/' + file,cv2.IMREAD_GRAYSCALE)
                if img is not None:
                    img = cv2.resize(img, (h, w),interpolation=cv2.INTER_CUBIC)
                    #img = np.asarray(img)
                    #img =img.reshape(1,size)
                    #X = np.vstack((X,img))
                    X.append(img)
                    y.append(label)
                    
    X =np.asarray(X)
    #X = np.vstack((X,img))
    y = np.asarray(y)
    return X,y
X_train, y_train = get_data(Train_dir,h,w)
X_test, y_test = get_data(Test_dir,h,w)
X_train=X_train.reshape((5216, 150, 150, 1))
X_test=X_test.reshape((624, 150, 150, 1))
import tensorflow as tf
#from tensorflow.keras.utils.np_utils import to_categorical
tf.keras.utils.to_categorical
y_train = tf.keras.utils.to_categorical(y_train, 2)
y_test = tf.keras.utils.to_categorical(y_test, 2)

from tensorflow.keras.callbacks import ReduceLROnPlateau , ModelCheckpoint
lr_reduce = ReduceLROnPlateau(monitor='val_acc', factor=0.1, epsilon=0.0001, patience=1, verbose=1)
filepath="weights.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')

#def swish_activation(x):
   # return (K.sigmoid(x) * x)

from tensorflow.keras import layers, models
from tensorflow.keras.optimizers import SGD , RMSprop

model = models.Sequential()
model.add(layers.Conv2D(16, (3, 3), activation='relu', padding="same", input_shape=(150,150,1)))
model.add(layers.Conv2D(16, (3, 3), padding="same", activation='relu'))
model.add(layers.MaxPooling2D(pool_size=(2, 2)))

model.add(layers.Conv2D(32, (3, 3), activation='relu', padding="same"))
model.add(layers.Conv2D(32, (3, 3), padding="same", activation='relu'))
model.add(layers.MaxPooling2D(pool_size=(2, 2)))

model.add(layers.Conv2D(64, (3, 3), activation='relu', padding="same"))
model.add(layers.Conv2D(64, (3, 3), padding="same", activation='relu'))
model.add(layers.MaxPooling2D(pool_size=(2, 2)))

model.add(layers.Conv2D(96, (3, 3), dilation_rate=(2, 2), activation='relu', padding="same"))
model.add(layers.Conv2D(96, (3, 3), padding="valid", activation='relu'))
model.add(layers.MaxPooling2D(pool_size=(2, 2)))

model.add(layers.Conv2D(128, (3, 3), dilation_rate=(2, 2), activation='relu', padding="same"))
model.add(layers.Conv2D(128, (3, 3), padding="valid", activation='relu'))
model.add(layers.MaxPooling2D(pool_size=(2, 2)))

model.add(layers.Flatten())

model.add(layers.Dense(64, activation='relu'))
#model.add(layers.Dense(32, activation='relu'))
model.add(layers.Dropout(0.5))
model.add(layers.Dense(2 , activation='sigmoid'))

model.compile(loss='binary_crossentropy',
                  #optimizer='adam',
                  optimizer=RMSprop(lr=0.00005),
                  metrics=['accuracy'])

print(model.summary())

#batch_size = 256
epochs = 6

history = model.fit(X_train, y_train, validation_data = (X_test , y_test) ,callbacks=[lr_reduce,checkpoint] ,
          epochs=6)

model.load_weights("weights.hdf5")

from sklearn import metrics
def print_score(y_true, y_pred):
    print('  mitrics : ', metrics.confusion_matrix(y_true, y_pred))
    print(' accuracy : ', metrics.accuracy_score(y_true, y_pred))
    print('precision : ', metrics.precision_score(y_true, y_pred))
    print('   recall : ', metrics.recall_score(y_true, y_pred))
    print('       F1 : ', metrics.f1_score(y_true, y_pred))
y_train_pred = model.predict(X_train)
y_train_pred = np.argmax(y_train_pred,axis = 1) 
y_train=np.argmax(y_train,axis = 1) 
y_test_pred = model.predict(X_test)
y_test_pred = np.argmax(y_test_pred,axis = 1) 
y_test= np.argmax(y_test,axis = 1) 
print_score(y_train,y_train_pred)
print_score(y_test,y_test_pred)

'''scratch '''
img = cv2.imread('chest_xray/train/NORMAL/IM-0115-0001.jpeg',cv2.IMREAD_GRAYSCALE)
print('数据类型:', type(img))
print('数组类型:', img.dtype)
print('数组形状：', img.shape)
print('数组最大值：{}，最小值：{}'.format(img.max(), img.min()))
img=cv2.resize(img,(160,160),interpolation=cv2.INTER_CUBIC)
print('数据类型:', type(img))
print('数组类型:', img.dtype)
print('数组形状：', img.shape)
print('数组最大值：{}，最小值：{}'.format(img.max(), img.min()))
angle = np.random.randint(1, 360)
h,w = img.shape[:2]
#if center is None:
center = (w/2,h/2)
M = cv2.getRotationMatrix2D(center,angle=180,scale=1.0)
img = cv2.warpAffine(img,M,(w,h))
#X = np.vstack((X,img))
img = np.asarray(img)
img =img.reshape(1,22500)
crop_seed = np.random.randint(0, 9)
img=img[crop_seed:crop_seed+150,crop_seed:crop_seed+150]
y_vote=np.zeros((2 ,0))
for i in range(num_test):
    y_vote=np.zeros((k,1), dtype = self.ytr.dtype)
    for j in range(k):
        distances[np.argmin(distances)] = np.max(distances)
        min_index = np.argmin(distances)
        y_vote[j]=self.ytr[min_index]
     Ypred[i] = round(np.mean(y_vote))       
round(np.mean(y_test))
y_test
a=np.array([1,2,3])
a[2]
y_vote=np.zeros((3,1))

for j in range(3):
    y_vote[j]=a[j]
import h5py
import json
import random
import numpy as np
import sys
import re
import matplotlib.pyplot as plt
import math

f = h5py.File('D:\用户目录\下载\MillionSongSubset\AdditionalFiles\subset_msd_summary_file.h5','r')
train_txt = 'D:\kaggle\kaggle_visible_evaluation_triplets_new.txt'
song_track = 'D:\kaggle\song_track.json'
train_v3 = 'D:/kaggle/train_v3.txt'
train_v4_aug = 'D:/kaggle/train_v4_aug.txt'

'''
group_keys = list(f.keys())
analysis = group_keys[0]
metadata = group_keys[1]
musicbrainz = group_keys[2]

group = f[analysis]
key = 'songs'

table = group[key].value
track = []
for line in table:
    trackid = str(line[-1],encoding='utf-8')
    track.append(trackid)
track = set(track)
'''

'''data augment
ori_train = []
with open(train_v3,'r',encoding='utf-8') as file:
    for line in file:
        ori_train.append(line)

with open(train_v4_aug,'w',encoding='utf-8') as newf:
    for i in range(40000):
        index = random.randint(0,len(ori_train)-1)
        newf.write(ori_train[index])
'''

# L = [['a','11'],['b','22'],['c','33']]
# L = np.array(L)
# for column in range(len(L[0])):
#     if column == 0:
#         L[:,column] = np.zeros(len(L))
#
# print(L)
    # Python实现正态分布
    # 绘制正态分布概率密度函数

u = 0   # 均值μ
u01 = -2
sig = math.sqrt(0.5)  # 标准差δ
sig01 = math.sqrt(1)
#sig02 = math.sqrt(5)
#sig_u01 = math.sqrt(0.5)
x = np.linspace(u - 3*sig, u + 3*sig, 50)
x_01 = np.linspace(u - 6 * sig, u + 6 * sig, 50)
#x_02 = np.linspace(u - 10 * sig, u + 10 * sig, 50)
#x_u01 = np.linspace(u - 10 * sig, u + 1 * sig, 50)
y_sig = np.exp(-(x - u) ** 2 /(2* sig **2))/(math.sqrt(2*math.pi)*sig)
plt.xlabel('x')
plt.ylabel('f(x)')
plt.ylim(0, 0.5)
y_sig01 = np.exp(-(x_01 - u) ** 2 /(2* sig01 **2))/(math.sqrt(2*math.pi)*sig01)
#y_sig02 = np.exp(-(x_02 - u) ** 2 / (2 * sig02 ** 2)) / (math.sqrt(2 * math.pi) * sig02)
#y_sig_u01 = np.exp(-(x_u01 - u01) ** 2 / (2 * sig_u01 ** 2)) / (math.sqrt(2 * math.pi) * sig_u01)

#plt.plot(x, y_sig,c="black", linewidth=2)
plt.plot(x_01, y_sig01, "black", linewidth=2)
#plt.plot(x_02, y_sig02, "b-", linewidth=2)
#plt.plot(x_u01, y_sig_u01, "m-", linewidth=2)
# plt.plot(x, y, 'r-', x, y, 'go', linewidth=2,markersize=8)
plt.grid(True)
plt.savefig('C:/Users/Administrator.USER-20161227PQ/Desktop/paper figure/figure1.png',dpi=300)
plt.show()

x = np.arange(-15,15,0.1)
#生成sigmiod形式的y数据
y=1/(1+np.exp(-x))
#设置x、y坐标轴的范围
plt.xlabel('x')
plt.ylabel('F(x)')
plt.xlim(-8,8)
plt.ylim(0, 1)
#绘制图形
plt.plot(x,y, c='black',linewidth=2)
plt.grid(True)
plt.savefig('C:/Users/Administrator.USER-20161227PQ/Desktop/paper figure/figure2.png',dpi=300)
plt.show()



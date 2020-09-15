from config import train_v
import matplotlib.pyplot as plt
import math

with open('D:/kaggle/kaggle_visible_evaluation_triplets_new.txt','r') as f:
    l0 = 0
    l1 = 1
    cnt = 0
    for line in f:
        _,_,label = line.strip().split('\t')
        cnt += 1
        if int(label) == 0:
            l0 += 1
        if int(label) == 1:
            l1 += 1
    print(cnt,l0,l1)

x = [0,0.2]
y = [l0, l1]
plt.bar(x,y,color='black', linewidth=2, width=0.05,)
plt.rcParams['font.sans-serif'] = ['simHei']
plt.rcParams['axes.unicode_minus'] = False
plt.xticks(x, [u"0", u"1"])
plt.xlabel('标签')
plt.ylabel('样本个数')
plt.xlim(-0.1,0.3)
plt.ylim(0, 1250933)
for x,y in zip([0,0.2],y):
    plt.text(x, y+1.3, str(round(y/cnt*100,1))+'%', ha='center', va= 'bottom')
#plt.grid(True)
#plt.savefig('C:/Users/Administrator.USER-20161227PQ/Desktop/paper figure/figure6.png',dpi=300)
plt.show()
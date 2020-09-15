import matplotlib.pyplot as plt
import math


L1 = []
L2 = []
x = [i for i in range(200)]
with open('./tree','r') as f:
    for line in f:
        f_str = line.strip()
        if f_str.find('train-error') != -1:
            L = f_str.split('\t')
            train = L[1]
            test = L[2]
            train_auc = round(float(train.split(':')[1]),6)
            test_auc = round(float(test.split(':')[1]),6)
            L1.append(train_auc)
            L2.append(test_auc)

plt.xlabel('Iteration')
plt.ylabel('Error')
plt.plot(x,L1, c='black',linewidth=2)
plt.plot(x,L2,'--',c='black',linewidth=2,)
plt.legend(labels=['train_error','test_error'])

plt.grid(True)
plt.savefig('C:/Users/Administrator.USER-20161227PQ/Desktop/paper figure/figure8.png',dpi=300)
plt.show()
import matplotlib.pyplot as plt
import math



y = [0.712, 0.70955, 0.71077, 0.71078201, 0.71091604, 0.7163439 , 0.71218924, 0.71084902,0.7138645 ]
x = [0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5]
plt.plot(x,y, c='black',linewidth=2)
plt.xlabel('Inverse of regularization strength C')
plt.ylabel('AUC')
# plt.xlim(0,0.8)
plt.ylim(0.65, 0.75)
plt.grid(True)
plt.savefig('C:/Users/Administrator.USER-20161227PQ/Desktop/paper figure/figure6.png',dpi=300)
plt.show()


y = [0.73542863, 0.73963638, 0.74064121, 0.73938517, 0.73385857, 0.73096967, 0.72531747,0.721,0.718]
x = [2,3,4,5,6,7,8,9,10]
plt.plot(x,y, c='black',linewidth=2)
plt.xlabel('Maximum depth of the tree')
plt.ylabel('AUC')
plt.grid(True)
plt.savefig('C:/Users/Administrator.USER-20161227PQ/Desktop/paper figure/figure7.png',dpi=300)
plt.show()
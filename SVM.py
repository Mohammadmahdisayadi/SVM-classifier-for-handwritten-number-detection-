import matplotlib.pyplot as plt 

from sklearn import datasets 
from sklearn import svm
import numpy as np 
import needfcn as nf

digits = datasets.load_digits()

clf = svm.SVC(gamma=0.01,C=100)
print(len(digits.data))



#%%
for i in range(16):
    x,y = digits.data[:-(i+1)],digits.target[:-(i+1)]
    clf.fit(x,y)
    a = clf.predict(digits.data)

f = plt.figure()
for index in range(16):
    image = digits.images[-(index +1)]
    sub = f.add_subplot(4,4,index+1)
    sub.imshow(image,cmap = plt.cm.gray_r,interpolation = "nearest")
    plt.xticks([]),plt.yticks([])
    sub.set_title('predict:'+str(a[-(index+1)])+' true:'+str(digits.target[-(index+1)]))
  



from sklearn.metrics import confusion_matrix
c = confusion_matrix(digits.target, a)
m,n = c.shape
cm = np.zeros((m+1,n+1))
cm[:m,:n] = c

noC = 1
noc=10

import math as mt


   
for i in range(noc):
    cm[i,noc] = round(100*(cm[i,i]/np.sum(cm[i,:],axis=0)),2)
    cm[noc,i] = round(100*(cm[i,i]/np.sum(cm[:,i],axis=0)),2)
    if (mt.isnan(float(cm[noc,i]))):
        cm[noc,i]=0


import seaborn as sb 
name2 = ['0','1','2','3','4','5','6','7','8','9','Acc']  
name3 = ['0','1','2','3','4','5','6','7','8','9','Val']  
  
plt.figure(),sb.heatmap(cm, xticklabels=name2, yticklabels=name3,annot=True,annot_kws={"size": 10},fmt='.0f')    
plt.title('SVM confusion matrix'),plt.xlabel('Predict Values'),plt.ylabel('True Values')



# data to plot
n_groups = 9
means_frank = (96, 74, 74, 88, 88, 8,58,97,100)
means_guido = (95, 64, 64, 84, 84, 2,43,96,100)

# create plot
fig, ax = plt.subplots()
index = np.arange(n_groups)
bar_width = 0.35
opacity = 0.8

rects1 = plt.bar(index, means_frank, bar_width,
alpha=opacity,
color='b',
label='Overall Accuracy')

rects2 = plt.bar(index + bar_width, means_guido, bar_width,
alpha=opacity,
color='g',
label='Average Validity')

plt.xlabel('classifier type')
plt.title('OA and AV')
plt.xticks(index + bar_width, ('MAP', 'ML', 'MD', 'SAM', 'CC', 'MF','PP','NN','SVM'))
plt.legend()


nf.autolabel(rects1,ax)
nf.autolabel(rects2,ax)

fig.tight_layout()  

#%% kappa

OA = np.zeros((1,noC))
AV = np.zeros((1,noC))
AA = np.zeros((1,noC))
OV = np.zeros((1,noC))


OA = np.round(100*np.sum(np.diag(cm[0:noc,0:noc],k=0))/np.sum(np.sum(cm[0:noc,0:noc])))
AV = np.round((1/noc)*np.sum(cm[noc,:]))
AA = np.round((1/noc)*np.sum(cm[:,noc]))


OA_new = OA/100

Ni = np.sum(cm[:noc,:noc],axis = 1)
Mj = np.sum(cm[:noc,:noc],axis = 0)
N = 9600

Pe = np.matmul(np.transpose(Ni),Mj)/(9600**2)
kappa = (OA_new - Pe)/(1 - Pe)


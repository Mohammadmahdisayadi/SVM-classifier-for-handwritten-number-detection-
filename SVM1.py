import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_olivetti_faces
faces=fetch_olivetti_faces()
print(faces.DESCR)

print(faces.keys())
print(faces.images.shape)
print(faces.data.shape)
print(faces.target.shape)

print(np.max(faces.data))
print(np.min(faces.data))
print(np.mean(faces.data))

def print_faces(images,target , top_n):
    ##set up figure size in inches
    fig=plt.figure(figsize=(12,12))
    fig.subplots_adjust(left=0 , right=1 ,bottom=0, top=1, hspace=0.05,wspace=0.05)
    for  i in range(top_n):
        #we will print images in matrix 20x20
        p=fig.add_subplot(4,5,i+1,xticks=[],yticks=[])
        p.imshow(images[i],cmap=plt.cm.bone)
        #label the image with target value
        p.text(0,14,str(target[i]))
        p.text(0,60,str(i))
        
print_faces(faces.images,faces.target,20)


from sklearn.svm import SVC
svc_1 = SVC(kernel='linear')

from sklearn.cross_validation import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
        faces.data, faces.target, test_size=0.25, random_state=0)






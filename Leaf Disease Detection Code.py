import cv2
from skimage.feature import greycomatrix,greycoprops
import numpy as np
from sklearn.metrics.cluster import entropy
import pandas as pd
l=[]
l1=[]
l2=['Unhealthy','Healthy','Unhealthy','Unhealthy','Healthy','Healthy','Healthy','Unhealthy','Unhealthy','Unhealthy']
for i in range(10):
    imgn = "Untitled"+str(i)+".jpg"
    img = cv2.imread(imgn)
    img1 = img
    #print(img.shape)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)  # Converting RGB image to HSV image
    #cv2.imshow('HSV image', hsv)
    #cv2.waitKey(0)
    lower_blue = np.array([14,32.64,22.185])
    upper_blue = np.array([34,255,232.815])
    mask = cv2.inRange(hsv, lower_blue, upper_blue) # Using Convolution mask/filter on hsv image
    res = cv2.bitwise_and(img1,img1, mask= mask)   # Applying bitwise operator between mask and originl image
    #cv2.imshow('HSV image', mask)
    #cv2.waitKey(0)
    #cv2.imshow('HSV image', res) 
    #cv2.waitKey(0)
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()
    res1 = cv2.cvtColor(res, cv2.COLOR_BGR2GRAY)
    glcm = greycomatrix(res1, [1], [0])        # Using grey level correlation matrix for extracting properties
    contrast = greycoprops(glcm, 'contrast')  # Contrast
    #print(contrast[0][0])
    Energy =greycoprops(glcm, 'energy')   # Energy
    #print(Energy[0][0])
    Homogeneity =greycoprops(glcm, 'homogeneity')  # Homogeneity
    #print(Homogeneity[0][0])
    m = res.mean()  # Mean
    s = res.std()   # Mean
    v = res.var()   # Mean
    #print(m,s,v)
    e = entropy(res)
    #print(e)
    rms = np.sqrt(np.mean(res**2))
    #print(rms)
    #print(res.sum())
    smoothness = 1 - (1/(1+res.sum()))
    #print(smoothness)
    l.append(contrast[0][0])
    l.append(Energy[0][0])
    l.append(Homogeneity[0][0])
    l.append(m)
    l.append(s)
    l.append(v)
    l.append(e)
    l.append(rms)
    l.append(smoothness)
    l.append(l2[i])
    l1.append(l)
    l=[]
#print(l1)
#print(len(l1))
df = pd.DataFrame(l1, columns = ['Contrast', 'Energy','Homogeneity','Mean','Standard_Deviation','Variance','Entropy','RMS','Smoothness','Type'])
#print(df) 
y=df['Type']
df=df.drop(['Type'],axis=1)
#print(y)
#print(df)
x=df
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X = sc_X.fit_transform(x)
X
y1=[]
for i in y:
    if i=='Healthy':
        y1.append(0)
    else:
        y1.append(1)
#print(y1)
#print(X)
#print(x)
from sklearn.linear_model import LogisticRegression # Using ML Classifiers
logmodel = LogisticRegression()
logmodel.fit(x,y1)
#print(logmodel)

imgn = "net.jpeg"
img = cv2.imread(imgn)
a=0
b=0
c=img.shape[0]
d=img.shape[1]
e=int((a+c)/2)
f=int((b+d)/2)
#print(a,b,c,d,e,f)
dim11=e-a
dim12=f-b
dim21=e-a
dim22=d-f
dim31=c-e
dim32=f-b
dim41=c-e
dim42=d-f
#print(dim11," ",dim12)
#print(dim21," ",dim22)
#print(dim31," ",dim32)
#print(dim41," ",dim42)
img1=img[a:e,b:f]
img2=img[a:e,f:d]
img3=img[e:c,b:f]
img4=img[e:c,f:d]
#cv2.imshow("cropped", img1)
#cv2.waitKey(0)
#cv2.imshow("cropped", img2)
#cv2.waitKey(0)
#cv2.imshow("cropped", img3)
#cv2.waitKey(0)
#cv2.imshow("cropped", img4)
#cv2.waitKey(0)
#cv2.destroyAllWindows()
img_list=[img1,img2,img3,img4]
from mpi4py import MPI        # Using MPI for Python to divide leaf image into 4 parts and allcoate each image to each thread (4 threads, 1 master thread)
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
if rank==0:
    imgn = "net.jpeg"
    img = cv2.imread(imgn)
    cv2.imshow("Tested leaf", img)
    cv2.waitKey(0)
    data1 = comm.recv(source=1, tag=1)
    data2 = comm.recv(source=2, tag=2)
    data3 = comm.recv(source=3, tag=3)
    data4 = comm.recv(source=4, tag=4)
    sumval=data1+data2+data3+data4
    if sumval>0:
        print('The leaf is unhealthy')
    else:
        print('The leaf is healthy')
else:
    img_it=img_list[rank-1]
    l3=[]
    l4=[]
    hsv = cv2.cvtColor(img_it, cv2.COLOR_BGR2HSV)
    cv2.imshow('HSV image', hsv)
    cv2.waitKey(0)
    lower_blue = np.array([14,32.64,22.185])
    upper_blue = np.array([34,255,232.815])
    mask = cv2.inRange(hsv, lower_blue, upper_blue)
    res = cv2.bitwise_and(img_it,img_it, mask= mask)
    #cv2.imshow('HSV image', mask)
    #cv2.waitKey(0)
    #cv2.imshow('HSV image', res)
    #cv2.waitKey(0)
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()
    res1 = cv2.cvtColor(res, cv2.COLOR_BGR2GRAY)
    glcm = greycomatrix(res1, [1], [0])
    contrast = greycoprops(glcm, 'contrast')
    Energy =greycoprops(glcm, 'energy')
    #print(Energy[0][0])
    Homogeneity =greycoprops(glcm, 'homogeneity')
    #print(Homogeneity[0][0])
    m = res.mean()
    s = res.std()
    v = res.var()
    e = entropy(res)
    rms = np.sqrt(np.mean(res**2))
    smoothness = 1 - (1/(1+res.sum()))
    l3.append(contrast[0][0])
    l3.append(Energy[0][0])
    l3.append(Homogeneity[0][0])
    l3.append(m)
    l3.append(s)
    l3.append(v)
    l3.append(e)
    l3.append(rms)
    l3.append(smoothness)
    #print(l3)
    l4.append(l3)
    l3=[]
    l5=logmodel.predict(l4)
    if l5[0]==0:
        print("The leaf part tested is healthy")
    else:
        print("The leaf part tested is unhealthy")
    #print(str(l5[0])+"is prediction for"+str(rank))
    comm.send(l5[0], dest=0, tag=rank)
    
    
    

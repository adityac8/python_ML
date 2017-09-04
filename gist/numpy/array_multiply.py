import numpy as np
b=[[1,2,3],[4,5,6],[7,8,9]]
b=np.array(b)
a=np.zeros([6,3])
j=0
for i in range(len(b)):
    a[j]=b[i]
    a[j+1]=b[i]
    j+=2

#Extract a single digit and its left half from mnist dataset and save the image  
from keras.datasets import mnist
import matplotlib.pyplot as plt

#import using mnist load_data 
(tx,ty),(_,_)=mnist.load_data()
print tx.shape

#single digit
digit=tx[3]
print digit.shape
plt.imshow(digit)

#left half of digit
x_left=digit[:,:-14]
print x_left.shape
plt.imshow(x_left)

plt.imsave('/home/ubu/output1.png',digit)
plt.imsave('/home/ubu/output2.png',x_left)

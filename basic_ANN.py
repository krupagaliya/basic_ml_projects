# Credit goes to ***Lazy Programmer***

import numpy as np
import matplotlib.pyplot as plt

#randn defines gaussian destribution

# adding of np.array defines the position of plotting points in scatter graph
Nclass = 300
x1 = np.random.randn(Nclass,2) + np.array([0,-2])
x2 = np.random.randn(Nclass,2) + np.array([2,2])
x3 = np.random.randn(Nclass,2) + np.array([-2,4])

#arrange in column
X = np.vstack([x1,x2,x3])
print(X)

#labeled Y 
Y = np.array([0]*Nclass+[1]*Nclass+[2]*Nclass)
print(Y)

print("--------------------")
print(X.shape)
print(Y.shape)

plt.scatter(X[:,0],X[:,1],c=Y,alpha=0.5)
plt.show()

def sigmoid(x):
	return 1/ (1 + np.exp((-x)))

def forward(X,w1,b1,w2,b2):
	z = sigmoid(X.dot(w1)+ b1)
	z2 = z.dot(w2) + b2
	#now apply softmax function code
	expa = np.exp(z2)
	y = expa / expa.sum(axis=1,keepdims=1)
	return y

#determine classification rate	
#Y as in labeled 
# P is ours predicted
def classification_score(Y,P):
	n_correct = 0
	n_total = 0
	for i in range(len(Y)):
		n_total +=1
		if Y[i] == P[i]:
			n_correct+=1
	return float(n_correct)/n_total		

#setting dimentionals
input_set = 2
Hidden_layer_size = 3
multi_class = 3

#weights and bias unit setting
w1 = np.random.randn(input_set,Hidden_layer_size)
b1 = np.random.randn(Hidden_layer_size)
w2 = np.random.randn(Hidden_layer_size,multi_class)
b2 = np.random.randn(multi_class)


training = forward(X,w1,b1,w2,b2)
final = np.argmax(training,axis=1)
print("-------------------")
print("Final shape = ",final.shape)
print("Labeled Y shape = ", Y.shape)	

if final.shape == Y.shape:
	print("Congrass .. :) ")
	print("Classification score",classification_score(Y,final))		
else:
	print(final.shape,"and ",Y.shape,"Should same")



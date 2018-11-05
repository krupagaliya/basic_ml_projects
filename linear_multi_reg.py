import numpy as np
x1 =   [0.18 ,0.89]
x2 =   [1.0, 0.26] 
x3 =   [0.92 ,0.11]
x4 =   [0.07 ,0.37]
x5 =   [0.85 ,0.16]
x6 =   [0.99, 0.41]
x7 =   [0.87, 0.47]
x = [x1,x2,x3,x4,x5,x6,x7]
y  = [[109.85],[155.72],[137.66],[76.17],[139.75],[162.6],[151.77]]  

X = np.array(x)
Y = np.array(y)
iters = 2000
alpha = 0.05
#need to add 1 to each row of Matrix X
ones = np.ones([X.shape[0],1])
X = np.concatenate((ones,X),axis=1)
print(X.shape)
print(Y.shape)
theta = np.zeros([1,3])
print(theta.shape)

def computeCost(X,Y,theta):
    temp = np.dot(X,theta.T)
    
    tobesummed = np.power((temp-Y),2)
    
    return np.sum(tobesummed)/(2 * len(X))

cost = computeCost(X,Y,theta)
print(cost)

def gradientDescent(X,y,theta,iters,alpha):  
    cost = np.zeros(iters)
    for i in range(iters):
        theta = theta - (alpha/len(X)) * np.sum(X *(np.dot(X,theta.T) - y) , axis=0)
        cost[i] = computeCost(X, y, theta)
    # print(cost)
    return theta,cost
g,cost1 = gradientDescent(X,Y,theta,iters,alpha)
#print(g)

print(cost1)

finalCost = computeCost(X,y,g)
print("final error",finalCost)

#make predictions
eq = g[0][0] + g[0][1]*(0.57)  + g[0][2]*( 0.83)
print(eq)

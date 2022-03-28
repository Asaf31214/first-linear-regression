import math
import numpy as np
import matplotlib.pyplot as plt
N=100
alpha=0.000001
x=np.random.rand(N)*100
y=(2*x)+((np.random.rand(N)-0.5)*40)
plt.scatter(x,y)
m=0 #incline
def lossfunc(x_data,y_data,incline):
    loss=0
    for q in range(N):
        loss+=y_data[q]-(x_data[q]*incline)
    return loss
print("first loss:",lossfunc(x,y,m),"first m:",m)
print("--------applying gradient descent----------")
for i in range(1000):
    sigma=0
    for j in range(N):
        sigma+=(m*x[j]-y[j])*x[j]
    m-=alpha*sigma/N
print("m:",m)
print("last loss:",lossfunc(x,y,m))
def rmse(x,y,m):
    sqe=0
    for e in range(N):
        sqe+=(y[e]-m*x[e])**2
    return math.sqrt(sqe/N)
print("root mean squared error:",rmse(x,y,m))
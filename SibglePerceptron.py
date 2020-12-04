# -*- coding: utf-8 -*-
"""
Created on Tue Mar 12 15:34:13 2019

@author: HP
"""

import numpy
import theano
import matplotlib.pyplot as plt

#Defining the theano variables
#Matrix
x=theano.tensor.matrix(name='x')
#Array
wval=numpy.asarray([numpy.random.randn(),numpy.random.randn()])
#Shared variables - will be required when needed to be shared within functions
w=theano.shared(wval,name='w')
b=theano.shared(0.5,name='b')


#inputs
#OR gate
xdata = [[0,0], #matrix
         [1,0],
         [0,1],
         [1,1]]
ydata = [0,1,1,0] #Array#[0,1,1,1]  [0,0,0,1]
#Vector product of tensors x & w
z = theano.tensor.dot(x,w)+b

#Activation fn will be for every node
#applying activation fn

ahat = 1/(1+theano.tensor.exp(-z)) #predicted y
#Defining the correct output variable

a=theano.tensor.vector('a')


#Defining the cost fn
# -(ylog(p)+(1-y)log(1-p))

# p  is predicted y
#A cost function is a measure of how
 
cost= -(a*theano.tensor.log(ahat)+(1-a)*theano.tensor.log(1-ahat)).sum()

#Reasons for cost fn - w , b
#Partial differentiation of cost w r.t. 'w','b'
#gradient descent

dcostdw=theano.tensor.grad(cost,w)

dcostdb=theano.tensor.grad(cost,b)

#apply GDA to compute the updated weights
wn=w-0.005*dcostdw
bn=b-0.005*dcostdb

#training fn

train=theano.function([x,a],[ahat,cost],updates=[(w,wn),(b,bn)])

cost1=[]
for i in range(60000):
    pred_val,cost_val=train(xdata,ydata)
    print(cost_val)
    cost1.append(cost_val)
    
print('Final ouptput are')

for i in range(len(xdata)):
    print('The o/p of x1=%d and x2=%d is %.3f'%(xdata[i][0],xdata[i][1],pred_val[i]))
    
    
plt.plot(cost1,color='red')
plt.show()    
    
        








































































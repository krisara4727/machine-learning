#using sklearn 

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LogisticRegression 
from sklearn.metrics import accuracy_score

data=pd.read_csv('../desktop/marks.txt')
print(data.head(10))
x=data.iloc[:, :-1]
y=data.iloc[:,-1]
admitted=data.loc[y==1]
not_admitted=data.loc[y==0]

plt.scatter(admitted.iloc[:,0],admitted.iloc[:,1],s=10,label='Admitted')
plt.scatter(not_admitted.iloc[:,0],not_admitted.iloc[:,1],s=10,label="Not_Admitted")
plt.legend()
#plt.show()

model=LogisticRegression()
model.fit(x,y)
predicted_classes=model.predict(x)
accuracy=accuracy_score(y.values.flatten(),predicted_classes)
parameters=model.coef_
print(parameters)
print(accuracy)


'''
#without using sklearn 

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

data=pd.read_csv('../desktop/marks.txt')
print(data.head(10))
x=data.iloc[:, :-1]
y=data.iloc[:,-1]
admitted=data.loc[y==1]
not_admitted=data.loc[y==0]

plt.scatter(admitted.iloc[:,0],admitted.iloc[:,1],s=10,label='Admitted')
plt.scatter(not_admitted.iloc[:,0],not_admitted.iloc[:,1],s=10,label="Not_Admitted")
plt.legend()
plt.show()

x=np.c_[np.ones((x.shape[0],1)),x]
y=y[:, np.newaxis]
theta=np.zeros((x.shape[1],1))

def sigmoid(x):
	return 1/(1+np.exp(-x))
def net_input(theta,x):
	return np.dot(x,theta)
	
def probability(theta,x):
	return sigmoid(net_input(theta,x))

def cost_function(self,theta,x,y):
	m=x.shape[0]
	total_cost=-(1/m)*np.sum(y*np.log(probability(theta,x))+(1-y)*np.log(1-probability(theta,x)))
	return total_cost
def gradient(self,theta,x,y):
	m=x.shape[0]
	return (1/m)*np.dot(x.T,sigmoid(net_input(theta,x))-y)
def fit(self,x,y,theta):
	opt_weights=fmin_tnc(func=cost_function,x0=theta,fprime=gradient,args=(x,y.flatten()))
	return opt_weights[0]
parameters=fit(x,y,theta)
x_values=[np.min(x[:,1]-5),np.max(x[:,]+5)]
y_values=-(parameters[0]+np.dot(parameters[1],x_values))/parameters[2]
plt.plot(x_values,y_values,label='Decision Boundary')
plt.xlabel('Marks in 1st Exam')
plt.ylabel('Marks in 2nd Exam')
plt.legend()
plt.show()

#ACCURACY OF THE MODEL

def predict(self,x):
	theta=parameters[:,np.newaxis]
	return probability(theta,x)
	
def accuracy(self,x,actual_classes,probab_threshold=0.5):
	predicted_classes=(predict(x)>=probab_threshold).astype(int)
	predicted_classes=predicted_classes.flatten()
	accuracy=np.mean(predicted_classes==actual_classes)
	return accuracy*100

accuracy(x,y.flatten()) 
'''

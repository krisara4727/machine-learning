import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
from sklearn.linear_model import LinearRegression
#from sklearn.metrics import mean_squared_error ,r2_error
data=pd.read_csv('../downloads/headbrain.csv')
data.head()
x=data.iloc[:,2:3].values
y=data.iloc[:,3:4].values

l=len(x)
first=l//4
x_train=x[:3*first]
x_test=x[3*first+1:]
y_train=y[:3*first]
y_test=y[3*first+1:]

#from sklearn.model_selection import train_test_split
#x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=1/4,random_state=0)
regressor=LinearRegression()
regressor.fit(x_train,y_train)
y_pred=regressor.predict(x_test)
accuracy=regressor.score(x_test,y_test)
print(accuracy*100,'%')
plt.scatter(x_train,y_train,c='red')
plt.show()
plt.plot(x_test,y_pred)   
plt.scatter(x_test,y_test,c='green')
plt.show()
plt.xlabel('headsize')
plt.ylabel('brain weight')


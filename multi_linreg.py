from sys import platform as sys_pf
if sys_pf == 'darwin':
    import matplotlib
    matplotlib.use("TkAgg")
from pandas import DataFrame
import matplotlib.pyplot as plt
from sklearn import linear_model
import statsmodels.api as sm
import tkinter as tk
stock_market={'Year': [2017,2017,2017,2017,2017,2017,2017,2017,2017,2017,2017,2017,2016,2016,2016,2016,2016,2016,2016,2016,2016,2016,2016,2016],
                'Month': [12, 11,10,9,8,7,6,5,4,3,2,1,12,11,10,9,8,7,6,5,4,3,2,1],
                'Interest_Rate': [2.75,2.5,2.5,2.5,2.5,2.5,2.5,2.25,2.25,2.25,2,2,2,1.75,1.75,1.75,1.75,1.75,1.75,1.75,1.75,1.75,1.75,1.75],
                'Unemployment_Rate': [5.3,5.3,5.3,5.3,5.4,5.6,5.5,5.5,5.5,5.6,5.7,5.9,6,5.9,5.8,6.1,6.2,6.1,6.1,6.1,5.9,6.2,6.2,6.1],
                'Stock_Index_Price': [1464,1394,1357,1293,1256,1254,1234,1195,1159,1167,1130,1075,1047,965,943,958,971,949,884,866,876,822,704,719]        
                }
df=DataFrame(stock_market,columns=['Year','Month','Interest_Rate','Unemployment_Rate','Stock_Index_Price'])
#to check linearity between the one of the independent 
#varibale and the dependent variable

plt.scatter(df['Interest_Rate'],df['Stock_Index_Price'],color='red')
plt.title('Stock Index Price Vs Interest Rate',fontsize=14)
plt.xlabel('Interest Rate', fontsize=14)
plt.ylabel('Stock Index Price',fontsize=14)
plt.grid(True)
plt.show()

plt.scatter(df['Unemployment_Rate'], df['Stock_Index_Price'], color='green')
plt.title('Stock Index Price Vs Unemployment Rate', fontsize=14)
plt.xlabel('Unemployment Rate', fontsize=14)
plt.ylabel('Stock Index Price', fontsize=14)
plt.grid(True)
plt.show()

X=df[['Interest_Rate','Unemployment_Rate']]
Y=df['Stock_Index_Price']

regr=linear_model.LinearRegression()
regr.fit(X,Y)
print('Intercept : \n',regr.intercept_)
print('Coefficients: \n',regr.coef_)

New_Interest_Rate=2.75
New_Unemployment_Rate=5.3
print('Predicted Stock Index price: \n',regr.predict([[New_Interest_Rate,New_Unemployment_Rate]]))
print(df.head(5))
#with statsmodel
X=sm.add_constant(X)
model=sm.OLS(Y,X).fit()
predictions=model.predict(X)
print_model=model.summary()
print(print_model)

#creating a graphical user interface for user inputs
root=tk.Tk()		#python interface for tk
canvas1=tk.Canvas(root,width=1200,height=450)
canvas1.pack()
#with sklearn
Intercept_result=('Intercept: ',regr.intercept_)
label_intercept=tk.Label(root,text=Intercept_result,justify='center')
canvas1.create_window(260,220,window=label_intercept)

Coefficients_result=('Coefficients: ',regr.coef_)
label_Coefficients=tk.Label(root,text=Coefficients_result,justify='center')
canvas1.create_window(260,240,window=label_Coefficients)

#with statsmodels
print_model=model.summary()
label_model=tk.Label(root,text=print_model,justify='center',relief='solid',bg='LightSkyBlue1')
canvas1.create_window(820,220,window=label_model)

#new interest rate and input box
label1=tk.Label(root,text='Type Interest Rate:')
canvas1.create_window(100,100,window=label1)

entry1=tk.Entry(root)
canvas1.create_window(270,100,window=entry1)

#new unemployment rate and input box
label2 = tk.Label(root, text=' Type Unemployment Rate: ')
canvas1.create_window(90, 130, window=label2)

entry2 = tk.Entry (root) # create 2nd entry box
canvas1.create_window(270, 130, window=entry2)

def values():
	global New_Interest_Rate
	New_Interest_Rate=float(entry1.get())
	
	global New_Unemployment_Rate #our 2nd input variable
	New_Unemployment_Rate = float(entry2.get())
	prediction_result=('predicted stock index price: ',regr.predict([[New_Interest_Rate,New_Unemployment_Rate]]))
	label_prediction=tk.Label(root,text=prediction_result,bg='orange')
	canvas1.create_window(260,280,window=label_prediction)
    
button1=tk.Button(root,text='Predict Stock Index Price',command=values,bg='orange')
canvas1.create_window(270,160,window=button1)
root.mainloop()


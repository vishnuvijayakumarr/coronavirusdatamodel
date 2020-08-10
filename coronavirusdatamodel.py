#Coronavirus Modeling through Linear Regresssion
#Vishnu Vijayakumar, August 2020
import pandas as pd
import matplotlib.pyplot as plt
import sklearn
import numpy as np
from datetime import datetime, timedelta
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn import linear_model

casedata = pd.read_csv(r'C:\Users\Vishnu Vijayakumar\Desktop\reference.csv')

casedata = casedata[["Date","Confirmed", "Recovered", "Deaths"]]
print(casedata)
casedata.Date=pd.to_datetime(casedata.Date)
casedata['Date']=casedata['Date'].values.astype(float)
print(casedata.dtypes)

predict = "Confirmed"
X = np.array(casedata.drop([predict], 1)) #?
y = np.array(casedata[predict]) #?

x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, test_size = 0.25) 
linear = linear_model.LinearRegression()
linear.fit(x_train, y_train)
acc = linear.score(x_test, y_test)
print(acc)

print('Coefficient: \n', linear.coef_)
print('Intercept: \n', linear.intercept_)

predictions = linear.predict(x_test)

for x in range(len(predictions)):
    print(predictions[x], x_test[x], y_test[x])

casedata2 = pd.DataFrame({'Actual': y_test.flatten(), 'Predicted': predictions.flatten()})
print(casedata2)



plt.scatter(x_test[:,0], y_test, color='gray')
plt.plot(x_test[:,0], predictions,color='red', linewidth=2)
plt.show()




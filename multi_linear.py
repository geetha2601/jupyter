#importing needed packages

import matplotlib.pyplot as plt
import pandas as pd
import pylab as pl
import numpy as np
#
#downloading data



#reading data
df = pd.read_csv("FuelConsumption.csv")
df.head()

#data exploration
df.describe()

#select some features
cdf = df[['ENGINESIZE','CYLINDERS','FUELCONSUMPTION_COMB','CO2EMISSINONS']]
cdf.head(9)

#plot features
viz = cdf[['ENGINESIZE','CYLINDERS','FUELCONSUMPTION_COMB']]
viz.hist()
plt.show()

#plot features vs emissions

plt.scatter(cdf.FUELCONSUMPTION_COMB , cdf.CO2EMISSIONS , color = 'blue')
plt.xlabel("FUELCONSUMPTION_COMB")
plt.ylabel("Emission")
plt.show()

plt.scatter(cdf.ENGINESIZE , cdf.CO2EMISSIONS , color = 'blue')
plt.xlabel("ENGINESIZE")
plt.ylabel("Emission")
plt.show()

plt.scatter(cdf.CYLINDERS , cdf.CO2EMISSIONS , color = 'blue')
plt.xlabel("Cylinders")
plt.ylabel("Emission")
plt.show()

#creating test and train datasets
msk = np.random.rand(len(df))<0.8
train = cdf[msk]
test = cdf[~msk]

#train data distribution
plt.scatter(train.ENGINESIZE,train.CO2EMISSIONS,color='blue')
plt.xlabel("Engine size")
plt.ylebel("Emission")
plt.show()

#modeling
from sklearn import linear_model
regr = linear_model.LineraRegression()
train_x = np.asanyarray(train[['ENGINESIZE']])
train_y = np.asanyarray(train[['CO2EMISSIONS']])
rege.fit(train_x,train_y)

print('coefficients : ',regr.coef_)
print('intercept: ',regr.intercept_)

#plot outputs

plt.scatter(train.ENGINESIZE,train.CO@EMISSIONS,color = 'blue')
plt.plot(train_x,regr.coef_[0][0]*train_x + regr_intercept_[0],'-r')
plt.xlabel("Engine size")
plt.ylabel("Emission")

#evaluation
from sklearn.metrics import r2_score

test_x = np.asanyarray(test[['ENIGINESIZE']])
test_y = np.asanyarray(test[['CO2EMISSIONS']])
test_y_hat = regr.predict(test_x)

print("Mean absolute error: %.2f" % np.mean(np.absolute(test_y_hat - test_y)))
print("Residual sum of squares (MSE): %.2f" % np.mean((test_y_hat - test_y) ** 2))
print("R2-score: %.2f" % r2_score(test_y_hat , test_y) )
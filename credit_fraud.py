import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import gridspec


path = "credit.csv"
data = pd.read_csv(path)

data.head()

print(data.shape)
print(data.describe())

fruad = data[data['Class']==1]
valid = data[data['Class']==0]
outlierFraction = len(fruad)/float(len(valid))
print(outlierFraction)
print("Fruad Cases: {}".format(len(data[data['Class']==1])))
print("Valid Cases: {}".format(len(data[data['Class']==0])))

print("Amount details of the fraudulent transactions: ")
fruad.Amount.describe()

print("Amount details of valid transactions: ")
valid.Amount.describe()

corrmat = data.corr()
fig = plt.figure(figsize=(12,9))
sns.heatmap(corrmat,vmax = .8,square = True)
plt.show()

X = data.drop(['Class'],axis = 1)
Y = data["Class"]

print(X.shape)
print(Y.shape)

from sklearn.model_selection import train_test_split

xTrain,Xtest,yTrain,yTest = train_test_split(xData,yData,test_size = 0.2,random_state=42)

from sklearn.ensemble import RandomForestClassifier
rfc = RandomForestClassifier()
rfc.fit(xTrain,yTrain)
ypred = rfc.predict(xTest)

from sklearn.metrics import classification_report,accuracy_score
from sklearn.metrics import precision_score,recall_score
from sklearn.metrics import f1_score,matthews_corrcoef
from sklearn.metrics import confusion_matrix

n_outliers = len(fruad)
n_errors = (ypred != yTest).sum()
print("The model used is Random Forest classifier") 

acc = accuracy_score(yTest,ypred)
print("The accuracy is {}".format(acc)) 
  
prec = precision_score(yTest, yPred) 
print("The precision is {}".format(prec)) 
  
rec = recall_score(yTest, yPred) 
print("The recall is {}".format(rec)) 
  
f1 = f1_score(yTest, yPred) 
print("The F1-Score is {}".format(f1)) 
  
MCC = matthews_corrcoef(yTest, yPred) 
print("The Matthews correlation coefficient is{}".format(MCC)) 


LABELS = ['Normal','Fruad']
conf_matrix = confusion_matrix(yTest,ypred)
plt.figure(figsize=(12,12))
sns.heatmap(conf_matrix,xticklabels = LABELS,yticklabels = LABELS , annot = True,fmt = 'd');
plt.title("Confusion Matrix")
plt.ylabel("True class")
plt.xlabel("Predicted class")
plt.show()


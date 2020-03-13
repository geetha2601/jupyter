#import pandas for dataset
import pandas as pd
df = pd.read_csv("SpeedDatingData.csv",encoding = "ISO-8859-1")
df.head()
#cleaning data
df = df.drop("iid",axis = 1)
df = df.drop("id",axis = 1)
df = df.drop("partner",axis = 1)

df1 = df
df2 = df
#prepare data
df1 = df1.drop("like", axis=1)
df1 = df1.drop("like_o", axis=1)
df1 = df1.drop("field", axis=1)
df1 = df1.drop("mn_sat", axis=1)
df1 = df1.drop("tuition", axis=1)
df1 = df1.drop("from", axis=1)
df1 = df1.drop("career", axis=1)
df1 = df1.drop("zipcode", axis=1)
df1 = df1.drop("income", axis=1)
df1 = df1.drop("undergra", axis=1)
df1 = df1.drop("dec", axis=1)
df1 = df1.drop("dec_o", axis=1)

#Remove where columns value = nan
#df1 = df1.dropna(axis=1, how='any')
#or replace Nan by 0 
df1 = df1.fillna(0)

#Some Stats
#Total
print(df1.shape)

#Repartition home/femme
men = df1.loc[df1['gender'] == 1]
women = df1.loc[df1['gender'] == 0]
print("Number of men : ", men.shape[0])
print("Number of women : ",women.shape[0])

#Women/men matchin
print("Men who match :",men.loc[df1['match'] == 1].shape[0])
print("Women who match :",women.loc[df1['match'] == 0].shape[0])


#Classifier we will test : 
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier

#Data splitting 
from sklearn.model_selection import train_test_split
import random

def accuracy(y_pred,y) : 
    return (sum(y_pred == y))/len(y)


def find_best(clfs):
    accuracyRes = 0.
    bestClf = None
    for clf in clfs:
        #try changing data test
        clf.fit(x_train,y_train)
        y_pred=clf.predict(x_test)
        accuracyValue = accuracy(y_pred,y_test)
        if accuracyValue > accuracyRes:
            accuracyRes = accuracyValue
            bestClf = clf
    return accuracyRes,bestClf


def data_split(x, y, prob):
        train,test=[],[]
        d=zip(x,y)
        for line in d:
            if(random.random()<prob):
                train.append(line)
            else :
                test.append(line)
        
        x_train,y_train=list(zip(*train))
        x_test,y_test=list(zip(*test))
        return x_train,y_train, x_test,y_test
    
y = df1["match"].as_matrix()
x = df1.drop("match", axis=1).as_matrix()

#x_train,y_train, x_test,y_test= train_test_split(x, y, test_size=.5)
x_train,y_train, x_test,y_test=data_split(x, y, .5)


#decision tree
decisionTreeClfs = []

for i in range(1,10):
    decisionTreeClfs.append(DecisionTreeClassifier(max_depth=i))    
        
accuracyValue,bestDecisionTreeClf = find_best(decisionTreeClfs)
print(accuracyValue)
print(bestDecisionTreeClf)


#MLP classifier
multiLayerPerceptronClassifiers = []
for i in range(70,90,5):
    multiLayerPerceptronClassifiers.append(MLPClassifier(max_iter=i))
    for j in range(60,80,5):
        multiLayerPerceptronClassifiers.append(MLPClassifier(max_iter=i,hidden_layer_sizes=j))

        
accuracyValue,bestMLPClf = find_best(multiLayerPerceptronClassifiers)
print(accuracyValue)
print(bestMLPClf)

#KNN classifier

kNeighborsClassifiers = []
for i in range(1,50,5):
    kNeighborsClassifiers.append(KNeighborsClassifier(algorithm="ball_tree",n_neighbors=i))
    kNeighborsClassifiers.append(KNeighborsClassifier(algorithm="brute",n_neighbors=i))
    kNeighborsClassifiers.append(KNeighborsClassifier(algorithm="kd_tree",n_neighbors=i))

        
accuracyValue,bestKNeighborsClf = find_best(kNeighborsClassifiers)
print(accuracyValue)
print(bestKNeighborsClf)

#Random Forest Classifier
randomForestClassifiers = []

for i in range(70,90,5):
    randomForestClassifiers.append(RandomForestClassifier(max_depth=i))
    for j in range(10,30,5):
        randomForestClassifiers.append(RandomForestClassifier(max_depth=i,n_estimators=j))
        
accuracyValue,bestRandomForestClf = find_best(randomForestClassifiers)
print(accuracyValue)
print(bestRandomForestClf)


#overall best classifier
overallClassifier = [bestRandomForestClf,bestKNeighborsClf,bestMLPClf,bestDecisionTreeClf]
accuracyValue,bestOverall = find_best(overallClassifier)

print(accuracyValue)

print(bestOverall)
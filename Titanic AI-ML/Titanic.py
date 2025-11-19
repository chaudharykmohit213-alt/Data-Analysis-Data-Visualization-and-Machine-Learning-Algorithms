import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from plotly.offline import plot
import plotly.express as px

pd.set_option('display.width', 500) 
pd.set_option('display.max_columns', None)
pd.set_option('display.max_colwidth', None)
sns.set_style('whitegrid')
train=pd.read_csv('titanic_train.csv')
#print(train.head())
#sns.heatmap(train.isnull(),yticklabels=False,cbar=False,cmap='viridis')
#sns.countplot(x='Survived',data=train,hue='Sex',palette='magma')
#sns.countplot(x='Survived',data=train,hue='Pclass',palette=["blue","green","red"],edgecolor="black",lw=0.4,alpha=0.9)
#sns.histplot(train['Age'].dropna(),kde=False,bins=30,color='blue',edgecolor="black",lw=0.4,alpha=1)
#print(train.info())
#sns.countplot(x='SibSp',data=train,palette=["blue","green","red"],edgecolor="black",lw=0.5,alpha=1)
#train['Fare'].hist(color='blue',edgecolor="black",lw=0.5,alpha=1,bins=40,figsize=(10,4))
#fig=px.histogram(train,x='Fare',nbins=50)
'''fig.update_traces(
    marker=dict(
        color='red',  # fill color
        line=dict(
            width=0.5,  # edge line width
            color='black'  # edge line color
        )
    )
)'''
#plot(fig,filename='Fare_Distribution.html',auto_open=True)
#sns.boxplot(x='Pclass',y='Age',data=train,palette=["blue","green","red"])
#plt.show()
def impute_age(cols):
    Age=cols[0]
    Pclass = cols[1]

    if pd.isnull(Age):

        if Pclass==1:
            return 37
        elif Pclass ==2:
            return 29
        else:
            return 24
    else:
        return Age

train['Age']=train[['Age','Pclass']].apply(impute_age,axis=1)
train.drop('Cabin',axis=1,inplace=True)
#print(train.head())
train.dropna(inplace=True)
#sns.heatmap(train.isnull(),yticklabels=False,cbar=False,cmap='viridis')
#plt.show()
sex = pd.get_dummies(train['Sex'],drop_first=True)
embark = pd.get_dummies(train['Embarked'],drop_first=True)
train.drop(['Sex','Embarked','Name','Ticket'],axis=1,inplace=True)
train = pd.concat([train,sex,embark],axis=1)
train[['male', 'Q', 'S']] = train[['male', 'Q', 'S']].astype(int)
#print(train.head())
from sklearn.model_selection import train_test_split
X=train.drop('Survived',axis=1)
Y=train['Survived']
X_train, X_test, Y_train, Y_test = train_test_split(X,Y,test_size=0.30,random_state=101)
from sklearn.linear_model import LogisticRegression
logmodel = LogisticRegression(solver='saga',max_iter=5000)
logmodel.fit(X_train,Y_train)
predictions = logmodel.predict(X_test)
from sklearn.metrics import classification_report
print(classification_report(Y_test,predictions))
from sklearn.metrics import confusion_matrix
print(confusion_matrix(Y_test,predictions))






















        

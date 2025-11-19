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
ad_data=pd.read_csv('advertising.csv')
#print(ad_data.head())
#print(ad_data.info())
#print(ad_data.describe())
#sns.histplot(x='Age',data=ad_data,edgecolor='black',color='red',lw=0.5,bins=30,alpha=1)
#sns.jointplot(x='Age',y='Area Income',data=ad_data,edgecolor='black',color='red',lw=0.5,alpha=1)
#sns.jointplot(x='Age',y='Daily Time Spent on Site',data=ad_data,kind='kde',color='red',joint_kws={'fill': True})
#sns.jointplot(x='Daily Time Spent on Site',y='Daily Internet Usage',data=ad_data,edgecolor='black',color='green',lw=0.5,alpha=1)
#sns.pairplot(ad_data,hue='Clicked on Ad')
#plt.show()
ad_data.drop(['Ad Topic Line','City','Country'],axis=1,inplace=True)
ad_data['Timestamp']=pd.to_datetime(ad_data['Timestamp']).dt.hour
print(ad_data.head())
from sklearn.model_selection import train_test_split
X=ad_data.drop(['Clicked on Ad','Timestamp'],axis=1)
Y=ad_data['Clicked on Ad']
X_train, X_test, Y_train, Y_test = train_test_split(X,Y,test_size=0.30,random_state=101)
from sklearn.linear_model import LogisticRegression
logmodel = LogisticRegression()
logmodel.fit(X_train,Y_train)
predictions = logmodel.predict(X_test)
from sklearn.metrics import classification_report
print(classification_report(Y_test,predictions))
from sklearn.metrics import confusion_matrix
print(confusion_matrix(Y_test,predictions))

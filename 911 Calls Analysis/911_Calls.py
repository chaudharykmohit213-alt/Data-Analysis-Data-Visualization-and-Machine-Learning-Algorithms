import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import random

pd.set_option('display.width', 500) 
pd.set_option('display.max_columns', None)
pd.set_option('display.max_colwidth', None)


df=pd.read_csv('911.csv')
#print(df.info())
#print(df['zip'].value_counts().head())
#print(df['twp'].value_counts().head())
#print(df['title'].nunique())
df['Reason']=df['title'].apply(lambda x:x.split(':')[0])
#print(df['Reason'].value_counts())
#sns.countplot(data=df, x='Reason', palette=['blue', 'green', 'orange'])
#plt.show()
#print(type(df['timeStamp'][0]))
df['timeStamp']=df['timeStamp'].apply(pd.to_datetime)
#print(df['timeStamp'].iloc[0].hour)
#print(df['timeStamp'].iloc[0].day)
#print(df['timeStamp'].iloc[0].month)
df['Hour']=df['timeStamp'].apply(lambda x:x.hour)
df['Day']=df['timeStamp'].apply(lambda x:x.day_name())
df['Month']=df['timeStamp'].apply(lambda x:x.month)
#fig=sns.countplot(x='Day',data=df,hue='Reason')
#fig.legend(bbox_to_anchor=(1.1, 1), loc='upper right')
#plt.show()
#fig=sns.countplot(x='Month',data=df,hue='Reason')
#fig.legend(bbox_to_anchor=(1.2, 1.2), loc='upper right')
#plt.tight_layout()
#plt.show()
by_month=df.groupby('Month')
#print(by_month.sum(numeric_only=True).head())
my_new_df=by_month.sum(numeric_only=True)
#my_new_df.plot(y='e',kind='line')
print(my_new_df.head())
my_new_df=my_new_df.reset_index()
sns.lmplot(x='Month',y='e',data=my_new_df)
plt.grid(True, color='lightgrey', linewidth=1.0)  
plt.show()

#df['Hour']=
#df['Day']=
#df['Month']=



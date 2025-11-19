import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

df =pd.read_csv('kc_house_data.csv')
#print(df.isnull().sum())
#print(df.describe().transpose())
plt.figure(figsize=(12,8))
#sns.histplot(df['price'],edgecolor='black',lw=0.5)
#sns.countplot(x=df['bedrooms'],data=df)
#plt.show()
#print(df.corr(numeric_only='True')['price'].sort_values())
#sns.scatterplot(x='price',y='sqft_living',data=df)
#sns.boxplot(x='bedrooms',y='price',data=df)
#sns.scatterplot(x='price',y='long',data=df)
#sns.scatterplot(x='price',y='lat',data=df)
#sns.scatterplot(x='long',y='lat',data=df,hue='price')
#plt.show()
#print(df.sort_values('price',ascending=False).head(20))
#non_top_1_per = df.sort_values('price',ascending=False).iloc[216:]
#sns.scatterplot(x='long',y='lat',data=non_top_1_per,hue='price',edgecolor=None,alpha=0.2,palette='RdYlGn')
#plt.show()
#sns.boxplot(x='waterfront',y='price',data=df)
#plt.show()
df=df.drop('id',axis=1)
df['date']=pd.to_datetime(df['date'])
#print(df['date'].head())
df['year']=df['date'].apply(lambda date:date.year)
df['month']=df['date'].apply(lambda date:date.month)
#sns.boxplot(x='month',y='price',data=df)
#plt.show()
#fig = df.groupby('month').mean()['price'].plot()
#plt.show()
df=df.drop('date',axis=1)
df = df.drop('zipcode',axis=1)
#print(df['yr_renovated'].value_counts())
X= df.drop('price',axis=1).values
y= df['price'].values
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.30,random_state=101)
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_test =  scaler.transform(X_test)
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense


model=Sequential()

model.add(Dense(19,activation='relu'))
model.add(Dense(19,activation='relu'))
model.add(Dense(19,activation='relu'))
model.add(Dense(19,activation='relu'))

model.add(Dense(1))

model.compile(optimizer='adam',loss='mse')
model.fit(x=X_train,y=y_train,validation_data=(X_test,y_test),batch_size=128,epochs=500,verbose=3)
loss_df = pd.DataFrame(model.history.history)
loss_df.plot()
plt.show()

from sklearn.metrics import mean_squared_error,mean_absolute_error,explained_variance_score
predictions = model.predict(X_test)
print((mean_squared_error(y_test,predictions))**0.5)
print(mean_absolute_error(y_test,predictions))
print(explained_variance_score(y_test,predictions))

plt.figure(figsize=(12,6))
plt.scatter(y_test,predictions)
plt.plot(y_test,y_test,'r')
plt.show()

single_house = df.drop('price',axis=1).iloc[0]
single_house =scaler.transform(single_house.values.reshape(-1,19))
print(model.predict(single_house))
























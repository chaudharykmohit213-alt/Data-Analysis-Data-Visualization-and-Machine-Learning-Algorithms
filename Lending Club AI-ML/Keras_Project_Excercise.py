import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

pd.set_option('display.width', 500) 
pd.set_option('display.max_columns', None)
pd.set_option('display.max_colwidth', None)

df = pd.read_csv("lending_club_loan_two.csv")

print(df.info())


#print(df.iloc[0])
#sns.countplot(x='loan_status',data= df)
#plt.show()
#sns.histplot(x='loan_amnt',color='blue',edgecolor="black",lw=0.4,alpha=0.8,data=df,bins=40)
#plt.show()
#print(df.corr(numeric_only=True))
#figsize=(15,8)
#sns.heatmap(data= df.corr(numeric_only=True),annot=True)
#plt.show()
data_info = pd.read_csv('lending_club_info.csv',index_col='LoanStatNew')
#print("Title Des: " + data_info.loc['title']['Description']+"\n")
#print("Emp Len Des: " + data_info.loc['emp_length']['Description']+"\n")
#print("Mort Acc Des: " + data_info.loc['mort_acc']['Description']+"\n")
#print("Pub Rec Bankrupties Des: " + data_info.loc['pub_rec_bankruptcies']['Description']+"\n")

#print(data_info.loc['loan_amnt']['Description'])
#sns.scatterplot(x='loan_amnt',y='installment',data=df,color='red',edgecolor='black',lw=0.5,s=10)
#sns.boxplot(x='loan_status',y='loan_amnt',data=df)
#print(df.groupby('loan_status')['loan_amnt'].describe())
#plt.show()
#print(len(df['emp_title'].unique()))
#print(df['sub_grade'].unique())
#sns.countplot(x='grade',data=df,hue='loan_status')
#subgrade_order= ['A1','A2','A3','A4', 'A5', 'B1', 'B2', 'B3', 'B4', 'B5', 'C1', 'C2', 'C3', 'C4', 'C5', 'D1', 'D2', 'D3', 'D4', 'D5', 'E1', 'E2', 'E3', 'E4', 'E5', 'F1', 'F2', 'F3', 'F4', 'F5', 'G1', 'G2', 'G3', 'G4', 'G5']
#fig=plt.figure(figsize=(12,6))
#FGList= ['F1','F2', 'F3', 'F4', 'F5', 'G1', 'G2', 'G3', 'G4', 'G5']
#sns.countplot(x='sub_grade',data=df,order=subgrade_order,palette='coolwarm')
#sns.countplot(x='sub_grade',data=df,order=subgrade_order,palette='coolwarm',hue='loan_status')
#sns.countplot(x='sub_grade',data=df[df['sub_grade'].isin(FGList)] ,order=FGList,hue='loan_status')
#plt.show()
df['loan_repaid']=df['loan_status'].apply(lambda x: int(x== 'Fully Paid'))
#print(df.iloc[0])
#print(df.corr(numeric_only=True).transpose())
#ax=sns.barplot(data=df.corr(numeric_only=True).sort_values(by='loan_repaid',ascending=True).drop('loan_repaid',axis=0)['loan_repaid'])
#ax.set_xticklabels(ax.get_xticklabels(), rotation=90, ha='right')
#plt.tight_layout() 
#plt.show()
#print(len(df['pub_rec_bankruptcies'].unique()))
#print(df['mort_acc'].unique())
#print(df['pub_rec_bankruptcies'].unique())
#print(df.isnull().sum())
#print(((df.isnull().sum())*100)/len(df))
df=df.drop('emp_title',axis=1)
#print(df['emp_length'].unique())
#df=df.dropna(axis=0,subset = ['emp_length'])
emp_length_order = ['< 1 year','1 year','2 years','3 years','4 years','5 years','6 years','7 years','8 years','9 years','10+ years' ]
#sns.countplot(x='emp_length',data=df,order=emp_length_order)
#sns.countplot(x='emp_length',data=df,order=emp_length_order,hue='loan_status')
'''
my_dict={}
for item in emp_length_order:
    temp_df=df[df['emp_length']==item]
    paid=(temp_df['loan_repaid']==1).sum()
    #print(paid)
    not_paid=(temp_df['loan_repaid']==0).sum()
    #print(not_paid)
    x=str(item)
    my_dict[x]=(not_paid)/(not_paid+paid)

my_ser=pd.Series(my_dict)
#print(my_ser)
#sns.barplot(my_ser)
#plt.show()'''

df=df.drop('emp_length',axis=1)
df=df.drop('title',axis=1)
#print(data_info.loc['mort_acc']['Description'])
#print(df['mort_acc'].value_counts())
#print(df.corr(numeric_only=True).sort_values(by='mort_acc',ascending=True)['mort_acc'])
my_dict=dict(df.groupby('total_acc').mean(numeric_only=True)['mort_acc'])
df['mort_acc'] = df.apply(
    lambda row: my_dict.get(row['total_acc'], row['mort_acc']) if pd.isnull(row['mort_acc']) else row['mort_acc'],
    axis=1
)
#print(df.isnull().sum())
df=df.dropna(axis=0,subset = ['revol_util'])
df=df.dropna(axis=0,subset = ['pub_rec_bankruptcies'])
#print(df.isnull().sum())
#int_cols = df.select_dtypes(include='object').columns
#print(int_cols)
df['term'] = df['term'].apply(lambda x: int(x.split()[0]))
df=df.drop('grade',axis=1)
cat_feats=['sub_grade']
df_with_dummies=pd.get_dummies(df,columns=cat_feats,drop_first=True)
bool_cols = df_with_dummies.select_dtypes(include='bool').columns
df_with_dummies[bool_cols]=df_with_dummies[bool_cols].astype(int)
#print(df_with_dummies.iloc[0])
'''
print("Verification Status : ")
print(df['verification_status'].unique())
print("Application Type : ")
print(df['application_type'].unique())
print("Initial List Status : ")
print(df['initial_list_status'].unique())
print("Purpose : ")
print(df['purpose'].unique())
'''
cat_feats=['verification_status', 'application_type','initial_list_status','purpose']
df_with_dummies_2=pd.get_dummies(df_with_dummies,columns=cat_feats,drop_first=True)
bool_cols = df_with_dummies_2.select_dtypes(include='bool').columns
df_with_dummies_2[bool_cols]=df_with_dummies_2[bool_cols].astype(int)
#print(df['home_ownership'].value_counts())
df_with_dummies_2['home_ownership'] = df_with_dummies_2.apply(
    lambda row: 'OTHER' if (row['home_ownership']=='NONE' or row['home_ownership']=='ANY')  else row['home_ownership'],
    axis=1
)
#print(df['home_ownership'].value_counts())
#print(df_with_dummies_2['home_ownership'].value_counts())
cat_feats = ['home_ownership']
df_with_dummies_3=pd.get_dummies(df_with_dummies_2,columns=cat_feats,drop_first=True)
bool_cols = df_with_dummies_3.select_dtypes(include='bool').columns
df_with_dummies_3[bool_cols]=df_with_dummies_3[bool_cols].astype(int)
#print(df_with_dummies_3.columns)
df_with_dummies_3['zip_code'] = df_with_dummies_3['address'].apply(lambda x: x.split()[-1])
#print(df_with_dummies_3.iloc[0])
#print(df_with_dummies_3['zip_code'].value_counts())

cat_feats = ['zip_code']
df_with_dummies_4=pd.get_dummies(df_with_dummies_3,columns=cat_feats,drop_first=True)
bool_cols = df_with_dummies_4.select_dtypes(include='bool').columns
df_with_dummies_4[bool_cols]=df_with_dummies_4[bool_cols].astype(int)
df_with_dummies_4=df_with_dummies_4.drop('address',axis=1)
df_with_dummies_4=df_with_dummies_4.drop('issue_d',axis=1)
#print(df_with_dummies_4['earliest_cr_line'].iloc[0:5])
df_with_dummies_4['earliest_cr_year']=df_with_dummies_4['earliest_cr_line'].apply(lambda x: int(x.split("-")[-1]))
#print(df_with_dummies_4['earliest_cr_year'].iloc[0:5])
df_with_dummies_4=df_with_dummies_4.drop('earliest_cr_line',axis=1)
df_with_dummies_4=df_with_dummies_4.drop('loan_status',axis=1)


from sklearn.model_selection import train_test_split
X = df_with_dummies_4.drop('loan_repaid',axis=1).values
y = df_with_dummies_4['loan_repaid'].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,random_state=101)

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_test =  scaler.transform(X_test)

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,Dropout


model=Sequential()

model.add(Dense(78,activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(39,activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(19,activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(1,activation='sigmoid'))

model.compile(loss='binary_crossentropy',optimizer='adam')

from tensorflow.keras.callbacks import EarlyStopping  
early_stop = EarlyStopping(monitor='val_loss',mode='min',verbose=1,patience=5)
model.fit(x=X_train,y=y_train,epochs=30,validation_data=(X_test,y_test),verbose=3,callbacks=[early_stop])

loss_df=pd.DataFrame(model.history.history)
loss_df.plot()
plt.show()

from tqdm import tqdm

print("X_test shape:", X_test.shape)

batch_size = 10000
num_samples = X_test.shape[0]
predictions_list = []

for start in tqdm(range(0, num_samples, batch_size), desc="Predicting"):
    end = min(start + batch_size, num_samples)
    batch_preds = model.predict(X_test[start:end], verbose=0)
    predictions_list.append(batch_preds)

predictions = np.vstack(predictions_list)
predictions = (predictions > 0.5).astype("int32")


from sklearn.metrics import classification_report,confusion_matrix
print(classification_report(y_test,predictions))
print(confusion_matrix(y_test,predictions))

import random
random.seed(101)

# Pick random index from processed dataframe
random_ind = random.randint(0, len(df_with_dummies_4) - 1)

# Extract customer features
new_customer = df_with_dummies_4.drop('loan_repaid', axis=1).iloc[random_ind]

# Convert to array and reshape
new_customer = np.array(new_customer).reshape(1, -1)

# Scale features to match training data
new_customer = scaler.transform(new_customer)

# Make prediction
new_customer_pred = model.predict(new_customer)
new_customer_pred = (new_customer_pred > 0.5).astype("int32")

print("Random Customer Data:\n", df_with_dummies_4.iloc[random_ind])
print("\nMy Prediction:")
print("Predicted 'loan_repaid':", int(new_customer_pred[0][0]))
































































#group_df['my_perc']=int(group_df['loan_repaid']==1)/int(group_df['loan_repaid']!=1)
#print(group_df['my_perc'])

























































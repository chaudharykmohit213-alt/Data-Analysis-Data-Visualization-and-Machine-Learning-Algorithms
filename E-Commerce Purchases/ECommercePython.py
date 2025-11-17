import pandas as pd
ecom=pd.read_csv('Ecommerce Purchases')
pd.set_option('display.width', 500) 
pd.set_option('display.max_columns', None)
pd.set_option('display.max_colwidth', None)
#print(ecom.head())
#print(ecom.info())
#print(ecom['Purchase Price'].mean())
#print(ecom['Purchase Price'].max())
#print(ecom['Purchase Price'].min())
#print(len(ecom[ecom['Language']=='en']))
#print(len(ecom[ecom['Job']=='Lawyer']))
#print(ecom['AM or PM'].value_counts())
#print(ecom.value_counts("Job").head())
#print(ecom[ecom['Lot']=='90 WT']['Purchase Price'])
#print(ecom.iloc[513])
#print(ecom[ecom['Credit Card']==4926535242672853]['Email'])
#print(len(ecom[(ecom['CC Provider']=='American Express')&(ecom['Purchase Price']>95)]))
'''i=0
counter=0
while (i<10000):
    if '25' in ecom.iloc[i]["CC Exp Date"].split("/"):
        counter +=1
    i=i+1

print("The number of Credit Cards expiting in 2025 are {}".format(counter))
'''
def splitemail(mystr):

    mylist=mystr.split("@")
    return mylist[1]

ecom['New List']=ecom['Email'].apply(splitemail)
print(ecom.value_counts("New List").head())



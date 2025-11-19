import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
'''
pd.set_option('display.width', 500) 
pd.set_option('display.max_columns', None)
pd.set_option('display.max_colwidth', None)'''
df = pd.read_csv("College_Data")
df = df.set_index(df.columns[0])
#print(df.head())
#print(df.info())
#print(df.describe())
sns.set_style("darkgrid")
#sns.scatterplot(x="Room.Board",y="Grad.Rate",data=df,hue="Private",edgecolor="black",lw=0.5)
#sns.scatterplot(x="Outstate",y="F.Undergrad",data=df,hue="Private",edgecolor="black",lw=0.5)
#sns.histplot(x="Outstate",data=df,hue="Private",bins=20)
#sns.histplot(x="Grad.Rate",data=df,hue="Private",bins=20)
#sp_school=df[df['Grad.Rate']>100]
#print(sp_school)
df.loc['Cazenovia College', 'Grad.Rate'] = 100
#print(df.loc['Cazenovia College'])
#sns.histplot(x="Grad.Rate",data=df,hue="Private",bins=20)
#plt.show()
from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters=2)
df_kmeans=df.drop('Private',axis=1)
kmeans.fit(df_kmeans)
print(kmeans.cluster_centers_)
df['Cluster']=(df['Private']=='Yes').astype(int)
#print(df.loc['Cazenovia College'])
#print(df['Cluster'].head())
#print(kmeans.labels_)
from sklearn.metrics import classification_report,confusion_matrix
print(confusion_matrix(df['Cluster'],kmeans.labels_))
print(classification_report(df['Cluster'],kmeans.labels_))












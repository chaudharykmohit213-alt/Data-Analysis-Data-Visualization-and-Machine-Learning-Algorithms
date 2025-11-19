import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from plotly.offline import plot
import plotly.express as px

pd.set_option('display.width', 500) 
pd.set_option('display.max_columns', None)
pd.set_option('display.max_colwidth', None)
iris = sns.load_dataset('iris')
sns.set_style('darkgrid')
'''g=sns.PairGrid(iris,hue='species')
g.map_diag(sns.histplot)
g.map_upper(sns.scatterplot)
g.map_lower(sns.scatterplot)
g.add_legend()
plt.show()'''

#sns.kdeplot(data=iris,x='sepal_width',y='sepal_length',fill=True,cmap='magma')
#plt.show()

#print(iris.iloc[0])
X=iris.drop('species',axis=1)
y=iris['species']
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix
X_train, X_test, y_train, y_test = train_test_split(X
    ,y , test_size=0.30, random_state=101)
print("\n🧪 SVM without scaling:")
model = SVC()
model.fit(X_train, y_train)
predictions = model.predict(X_test)
print(confusion_matrix(y_test, predictions))
print(classification_report(y_test, predictions))

# GridSearch without scaling
param_grid = {
    'C': [0.1, 1, 10, 100, 1000],
    'gamma': [1, 0.1, 0.01, 0.001, 0.0001],
    'kernel': ['rbf']
}
grid = GridSearchCV(SVC(), param_grid, refit=True, verbose=3)
grid.fit(X_train, y_train)
print("Best Params (no scaling):", grid.best_params_)
grid_predictions = grid.predict(X_test)
print(confusion_matrix(y_test, grid_predictions))
print(classification_report(y_test, grid_predictions))























































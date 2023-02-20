import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import load_iris

iris = load_iris()
print(type(iris))

iris_df = pd.DataFrame(data= np.c_[iris['data'], iris['target']],
                     columns= iris['feature_names'] + ['target'])
print(iris_df.head())
print(iris_df.describe())
print(iris_df.info())

plt.ion()
iris_df.hist()
plt.show()

plt.figure()
cols = ['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)','petal width (cm)']
corr_matrix = iris_df[cols].corr()
heatmap = sns.heatmap(corr_matrix,cbar=True,annot=True,square=True,fmt='.2f',annot_kws={'size': 15},yticklabels=cols,xticklabels=cols,cmap='Dark2')
plt.show()

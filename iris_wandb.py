import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import wandb
# start a new wandb run to track this script
wandb.init(
    project="iris_wandb_proj",
)

from sklearn.datasets import load_iris

iris = load_iris()
print(type(iris))

iris_df = pd.DataFrame(data= np.c_[iris['data'], iris['target']],
                     columns= iris['feature_names'] + ['target'])
                     
head = iris_df.head()
describe = iris_df.describe()
info = iris_df.info()
wandb.log({
    'head': head,
    'describe': describe,
    'info': info,
    })

ax = iris_df.hist()
wandb.log({'hist': wandb.Image(ax[0,0].get_figure())})

fig2 = plt.figure()
cols = ['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)','petal width (cm)']
corr_matrix = iris_df[cols].corr()
heatmap = sns.heatmap(corr_matrix,cbar=True,annot=True,square=True,fmt='.2f',annot_kws={'size': 15},yticklabels=cols,xticklabels=cols,cmap='Dark2')
wandb.log({'corr': wandb.Image(fig2)})

wandb.finish()
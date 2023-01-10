#!/usr/bin/env python
# coding: utf-8

# In[2]:


import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn


# In[3]:


import pandas as pd

df = pd.read_csv(r'C:\DS\DS610\Project\Data\sample_submission.csv')
print(df)


# In[13]:


import pandas as pd

dataTrain = pd.read_csv(r'C:\DS\DS610\Project\Data\train.csv')
dataTrain.head()


# In[15]:


dataTest = pd.read_csv(r'C:\DS\DS610\Project\Data\test.csv')
idCol = dataTest.Id.to_numpy()
dataTest.tail()


# In[16]:


fig, axes = plt.subplots(1, 2, sharex=True, figsize=(20,10))
sns.heatmap(ax=axes[0], yticklabels=False, data=dataTrain.isnull(), cbar=False, cmap="rocket")
sns.heatmap(ax=axes[1], yticklabels=False, data=dataTest.isnull(), cbar=False, cmap="crest")
axes[0].set_title('Heatmap of missing values in training data')
axes[1].set_title('Heatmap of missing values in testing data')
plt.show()


# In[18]:


def show_values(axs, orient="v", space=.01):
    def _single(ax):
        if orient == "v":
            for p in ax.patches:
                _x = p.get_x() + p.get_width() / 2
                _y = p.get_y() + p.get_height() + (p.get_height()*0.01)
                value = '{:.1f}'.format(p.get_height())
                ax.text(_x, _y, value, ha="center") 
        elif orient == "h":
            for p in ax.patches:
                _x = p.get_x() + p.get_width() + float(space)
                _y = p.get_y() + p.get_height() - (p.get_height()*0.5)
                value = '{:.1f}'.format(p.get_width())
                ax.text(_x, _y, value, ha="left")

    if isinstance(axs, np.ndarray):
        for idx, ax in np.ndenumerate(axs):
            _single(ax)
    else:
        _single(axs)


# In[19]:


fig, axes = plt.subplots(1, 2, sharex=True, figsize=(20,10))
nanTrain = {}
for column in dataTrain.columns[1:]:
    perc =  dataTrain[column].isna().sum()/len(dataTrain[column])
    if perc >= 0.01:
        nanTrain[str(column)] = perc
nanTrain = {key: value*100 for key, value in sorted(nanTrain.items(), key=lambda item: item[1], reverse=True)}
a = sns.barplot(ax=axes[0], y=list(nanTrain.keys()), x=list(nanTrain.values()), palette="coolwarm", ci=None)
plt.xlabel("NaN Values (%)")
plt.ylabel("Labels")
plt.title('NaN values in training data:')
#===================================================================================================
nanTest = {}
for column in dataTest.columns[1:]:
    perc =  dataTest[column].isna().sum()/len(dataTest[column])
    if perc >= 0.01:
        nanTest[str(column)] = perc
nanTest = {key: value*100 for key, value in sorted(nanTest.items(), key=lambda item: item[1], reverse=True)}
b = sns.barplot(ax=axes[1], y=list(nanTest.keys()), x=list(nanTest.values()), palette="flare", ci=None)

axes[0].set_title('Missing data in training set')
axes[1].set_title('Missing data in training set')
axes[0].set_xlabel('NaN Values (%)')
axes[0].set_ylabel('Labels')
axes[1].set_xlabel('NaN Values (%)')
axes[1].set_ylabel('Labels')

show_values(a, "h", space=0.3)
show_values(b, "h", space=0.3)


# In[21]:


dataTrain.info()


# In[22]:


dataTrain.columns


# In[24]:


plt.figure(figsize=(22,11))
sns.heatmap(dataTrain.corr(), cmap="viridis", annot=True)
plt.show()


# In[25]:


plt.figure(figsize=(20,8))
plt.subplot(1, 3, 1)
ax1 = sns.kdeplot(dataTrain.LotFrontage, color=sns.color_palette('magma')[2], shade=True)
plt.subplot(1, 3, 2)
ax2 = sns.kdeplot(dataTrain.LotArea, color=sns.color_palette('flare')[2], shade=True)
plt.subplot(1, 3, 3)
ax3 = sns.kdeplot(dataTrain.MSSubClass, color=sns.color_palette('viridis')[2], shade=True)
ax1.grid(False)
ax2.grid(False)
ax3.grid(False)
plt.show()


# In[35]:


from sklearn.impute import SimpleImputer
imputer = SimpleImputer(missing_values=np.nan, strategy='most_frequent')
dataTrain = pd.DataFrame(imputer.fit_transform(dataTrain), columns=dataTrain.columns)
dataTest = pd.DataFrame(imputer.fit_transform(dataTest), columns=dataTest.columns)
dataTrain = dataTrain.reset_index(drop=True)


# In[36]:


dataTrain.isnull().sum()


# In[38]:


dataTest.isnull().sum()


# In[39]:


dataTrain.head()


# In[40]:


dataTrain = pd.get_dummies(dataTrain, columns=['MSZoning', 'Street',
                                               'Alley', 'LotShape', 'LandContour', 'Utilities', 'LotConfig',
                                               'LandSlope', 'Neighborhood', 'Condition1', 'Condition2', 'BldgType',
                                               'HouseStyle',
                                               'RoofStyle', 'RoofMatl', 'Exterior1st', 'Exterior2nd', 'MasVnrType',
                                               'ExterQual', 'ExterCond', 'Foundation', 'BsmtQual',
                                               'BsmtCond', 'BsmtExposure', 'BsmtFinType1',
                                               'BsmtFinType2', 'Heating',
                                               'HeatingQC', 'CentralAir', 'Electrical',
                                               'KitchenQual',
                                               'Functional', 'FireplaceQu', 'GarageType',
                                               'GarageFinish', 'GarageQual',
                                               'GarageCond', 'PavedDrive',
                                               'EnclosedPorch', '3SsnPorch', 'PoolQC',
                                               'Fence', 'MiscFeature', 'SaleType',
                                               'SaleCondition'])
lab = dataTrain.SalePrice.to_numpy()
dataTrain.drop(columns=['Id', 'SalePrice'], inplace=True)
dataTrain.head()


# In[41]:


dataTest = pd.get_dummies(dataTest, columns=[ 'MSZoning', 'Street',
                                               'Alley', 'LotShape', 'LandContour', 'Utilities', 'LotConfig',
                                               'LandSlope', 'Neighborhood', 'Condition1', 'Condition2', 'BldgType',
                                               'HouseStyle',
                                               'RoofStyle', 'RoofMatl', 'Exterior1st', 'Exterior2nd', 'MasVnrType',
                                               'ExterQual', 'ExterCond', 'Foundation', 'BsmtQual',
                                               'BsmtCond', 'BsmtExposure', 'BsmtFinType1',
                                               'BsmtFinType2', 'Heating',
                                               'HeatingQC', 'CentralAir', 'Electrical',
                                               'KitchenQual',
                                               'Functional', 'FireplaceQu', 'GarageType',
                                               'GarageFinish', 'GarageQual',
                                               'GarageCond', 'PavedDrive',
                                               'EnclosedPorch', '3SsnPorch', 'PoolQC',
                                               'Fence', 'MiscFeature', 'SaleType',
                                               'SaleCondition'])


# In[42]:


dataTrain, dataTest = dataTrain.align(dataTest, join='inner', axis=1)


# In[43]:


dataTrain.shape, dataTest.shape


# In[44]:


dataTest.head()


# In[45]:


yTrain = lab
xTest = dataTest.to_numpy()
xTrain = dataTrain.to_numpy()
xTrain.shape, yTrain.shape, xTest.shape


# In[47]:


def SMAPE(a, f):
    return 1/len(a) * np.sum(2 * np.abs(f-a) / (np.abs(a) + np.abs(f))*100)


# Linear regression
# 

# In[48]:


from sklearn.linear_model import LinearRegression
lRegressor = LinearRegression()
lRegressor.fit(xTrain, yTrain)
lError = SMAPE(yTrain, lRegressor.predict(xTrain))
lError


# LightGBM Regression
# 

# In[52]:


lgbmRegressor = LGBMRegressor()
lgbmRegressor.fit(xTrain, yTrain)
lgbmError = SMAPE(yTrain, lgbmRegressor.predict(xTrain))
lgbmError


# Gradient Boosting Regression

# In[53]:


from sklearn.ensemble import GradientBoostingRegressor

gbRegressor = GradientBoostingRegressor()
gbRegressor.fit(xTrain, yTrain)
gbError = SMAPE(yTrain, gbRegressor.predict(xTrain))
gbError


# Bayesian Ridge

# In[54]:


from sklearn.linear_model import BayesianRidge

brRegressor = BayesianRidge()
brRegressor.fit(xTrain, yTrain)
brError = SMAPE(yTrain, brRegressor.predict(xTrain))
brError


# Elastic net

# In[65]:


from sklearn.linear_model import ElasticNet

enRegressor = ElasticNet()
enRegressor.fit(xTrain, yTrain)
enError = SMAPE(yTrain, enRegressor.predict(xTrain))
enError


# CatBoost Regression

# In[69]:


from catboost import CatBoostRegressor
cbRegressor = CatBoostRegressor()
cbRegressor.fit(xTrain, yTrain, silent=True)
cbError = SMAPE(yTrain, cbRegressor.predict(xTrain))
cbError


# Extreme Gradient Boosting

# In[71]:


from xgboost.sklearn import XGBRegressor
xgbRegressor = CatBoostRegressor()
xgbRegressor.fit(xTrain, yTrain, silent=True)
xgbError = SMAPE(yTrain, xgbRegressor.predict(xTrain))
xgbError


# Random Forest Regression

# In[72]:


from sklearn.ensemble import RandomForestRegressor
rfRegressor = RandomForestRegressor()
rfRegressor.fit(xTrain, yTrain)
rfError = SMAPE(yTrain, rfRegressor.predict(xTrain))
rfError


# In[ ]:


Support vector regression


# In[73]:


from sklearn.svm import SVR
svRegressor = SVR()
svRegressor.fit(xTrain, yTrain)
svError = SMAPE(yTrain, svRegressor.predict(xTrain))
svError


# Evaluation of the different models

# In[74]:


errors = [lError, lgbmError, gbError, brError, enError, krError, cbError, xgbError, rfError, svError]
dataPerf = pd.DataFrame(data={'Model': ['LinearRegression', 'LGBM', 'GradientBoosting', 'Bayesian Ridge', 'ElasticNet', 'KernelRidge', 'CatBoost', 'XGB', 'RandomForest', 'SupportVector'], 'Error': errors})

plt.figure(figsize=(12, 8))
sns.barplot(x="Model", y="Error", data=dataPerf, palette="magma")
plt.title('Performance analysis of different classifiers')
plt.show()


# In[75]:


dataPerf


# In[76]:


print(f'Minimum SMAPE: {min(errors)}')


# In[ ]:





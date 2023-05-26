#!/usr/bin/env python
# coding: utf-8

# # Import Libraries

# In[1]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, StandardScaler , MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import BernoulliNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, recall_score, precision_score, confusion_matrix, classification_report


# # Load Dataset

# In[2]:


Train = pd.read_csv(r"D:\Anime\ML Project\train.csv")
Test  = pd.read_csv(r"D:\Anime\ML Project\test.csv")


# In[3]:


Data = pd.concat([Train, Test])
Data.head()


# # Data Cleaning

# ## Drop unneeded row

# In[4]:


Data = Data.drop('Unnamed: 0', axis=1)
Data.head()


# In[5]:


Data.shape


# In[6]:


Data.info()


# ## Drop Nan value 

# In[7]:


Data.isnull().sum()


# In[8]:


Data =Data.dropna().copy()


# In[9]:


Data.isnull().sum()


# ## check if there is any duplicated value

# In[10]:


Data.duplicated().any() 


# ## see count , mean , std .. etc

# In[11]:


Data.describe()


# ## see if there is any outliers ..

# In[12]:


sns.boxplot(x=Data['Departure Delay in Minutes'])


# In[13]:


sns.boxplot(x=Data['Arrival Delay in Minutes'])


# In[14]:


Data.loc[Data['Departure Delay in Minutes'] > 1100]

Data.loc[Data['Arrival Delay in Minutes'] > 1100]


# In[15]:


Data.shape


# In[16]:


Outliers = Data[Data['Arrival Delay in Minutes'] > 1100].index
Data.drop(Outliers, inplace=True)
Data.shape


# # Data Visualization

#  ## linear relationship between Departure Delay in Minutes and Arrival

# In[17]:


plt.figure(figsize=(10,5), dpi=100)
sns.scatterplot(data=Data,x='Arrival Delay in Minutes',y='Departure Delay in Minutes',hue='satisfaction',palette='gist_rainbow_r', alpha=0.8)


# ## Feature Counts

# In[18]:


plt.figure(figsize=(10,10))
plt.subplot(2,2,1)
sns.countplot(data=Data, x="Class" , hue="satisfaction", palette="afmhot")
plt.title("Count Plot of Class")
plt.legend()

plt.subplot(2,2,2)
sns.countplot(data=Data, x="Gate location" , hue="satisfaction", palette="Paired")
plt.title("Count Plot of Gate location")
plt.legend()

plt.subplot(2,2,3)
sns.countplot(data=Data, x="Customer Type" , hue="satisfaction", palette="CMRmap_r")
plt.title("Count Plot of Customer Type")
plt.legend()

plt.subplot(2,2,4)
sns.countplot(data=Data, x="Checkin service" , hue="satisfaction", palette="seismic_r")
plt.title("Count Plot of Checkin service")
plt.legend()
plt.tight_layout()
plt.show()


# # Encoding

# ## Target Encoding

# In[19]:


Data['satisfaction'].value_counts()


# In[20]:


Data['satisfaction']


# In[21]:


Satisfaction= {'neutral or dissatisfied':0,'satisfied':1}
Data['satisfaction'] = Data['satisfaction'].map(Satisfaction)
Data['satisfaction']


# ## Features Encoding

# ### Label Encoding

# In[22]:


Data['Gender']


# In[23]:


Gender= {'Male':0,'Female':1}
Data['Gender'] = Data['Gender'].map(Gender)
Data['Gender']


# In[24]:


Data['Customer Type']


# In[25]:


Customer= {'Loyal Customer':0,'disloyal Customer':1}
Data['Customer Type'] = Data['Customer Type'].map(Customer)
Data['Customer Type']


# In[26]:


Data['Type of Travel']


# In[27]:


Taverl_type = {'Personal Travel':0,'Business travel':1}
Data['Type of Travel'] = Data['Type of Travel'].map(Taverl_type)
Data['Type of Travel']


# ### One Hot Encoding

# In[28]:


Data['Class']


# In[29]:


Dummies = pd.get_dummies(Data['Class'])
Data = pd.concat([Data, Dummies], axis=1)
Data = Data.drop('Class', axis=1)
Data


# # Scale the numeric features

# In[30]:


num_cols = ['Flight Distance', 'Departure Delay in Minutes', 'Arrival Delay in Minutes']
scaler = StandardScaler()
Data[num_cols] = scaler.fit_transform(Data[num_cols])


# In[31]:


Data


# # Data Correlations

# ## Correlation with BALANCE variable

# In[32]:


cor_target = abs(Data.corr()['satisfaction'])
#Selecting highly correlated features
relevant_features = cor_target[cor_target>0.05]
relevant_features.sort_values(ascending=False)


# ## people, even though they had a good online boarding experience, they weren't satisified
# 

# In[33]:


sns.boxplot(x='satisfaction', y='Online boarding', data=Data)


# ## Correlation matrix

# In[34]:


correlation_matrix = Data.corr()
plt.figure(figsize=(30, 20))
sns.set(font_scale=1.5)
heatmap = sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', annot_kws={'fontsize': 16})
heatmap.set_xticklabels(heatmap.get_xticklabels(), fontsize=16)
heatmap.set_yticklabels(heatmap.get_yticklabels(), fontsize=16)
plt.title('Correlation Matrix', fontsize=24)
plt.show()


# # Determine Features and Target

# In[35]:


Data.shape


# In[36]:


Features = Data.drop(['id','satisfaction'], axis=1)
Target = Data['satisfaction']


# In[37]:


Features


# In[38]:


Target


# # Splitting dataset to tree parts Training, Validation and Testing.

# In[39]:


X, X_test, y, y_test = train_test_split(Features, Target,
                                        test_size = 0.20, random_state = 2)
X_train, X_val, y_train, y_val = train_test_split(X, y,
                                        test_size = 0.20, random_state = 2)


# In[40]:


print(X_train.shape)
print(y_train.shape)
print(X_test.shape)
print(y_test.shape)


# In[41]:


train_acc=[]
val_acc=[]
test_acc=[]


# # **Model**

# ## KNN Model

# ### The best values for HyperParameters using GridSearchCV 

# **n_neighbors**: This hyperparameter determines the number of nearest neighbors to consider when making a prediction. ***The default value is 5***.
# 
# **p**: This hyperparameter determines the distance metric used to calculate the distances between the nearest neighbors. When p=1, the distance metric is Manhattan distance, and when p=2, the distance metric is Euclidean distance. ***The default value is p=2 (Euclidean distance)***.

# In[42]:


# # define the parameter grid to search over
# param_grid = {
#     'n_neighbors': [3, 5, 7, 9],
#     'p': [1, 2]
# }

# # create a KNN model instance
# knn = KNeighborsClassifier()

# # create a GridSearchCV instance with the parameter grid and the KNN model
# grid_search = GridSearchCV(estimator=knn, param_grid=param_grid, cv=5, scoring='accuracy')

# # fit the GridSearchCV instance on the training data
# grid_search.fit(X_train, y_train)
# print(grid_search.best_estimator_)
# y_pred_grid_search = grid_search.predict(X_test)
# # print the best hyperparameters and corresponding score
# print("Train Score: ",grid_search.fit(X_train, y_train).score(X_train, y_train))
# print("Validation Accuracy:", accuracy_score(y_val, grid_search.predict(X_val)))
# print("Test Accuracy:", accuracy_score(y_test, y_pred_grid_search))
# print("Best Hyperparameters: ", grid_search.best_params_)
# print("Best Score: ", grid_search.best_score_)


# **best values** :
#                n_neighbors=5 ***(Default)***, 
#                p=1

# ### Train the KNN model on the training data

# In[43]:


KNN = KNeighborsClassifier(p=1)
KNN.fit(X_train, y_train)


# ### Make predictions on the test data

# In[44]:


y_pred_KNN = KNN.predict(X_test)


# ### Evaluate the performance of the model

# In[45]:


train_acc.append(KNN.score(X_train, y_train))
val_acc.append(accuracy_score(y_val, KNN.predict(X_val)))
test_acc.append(accuracy_score(y_test, y_pred_KNN))
print("Train Accuracy:", KNN.score(X_train, y_train))
print("Validation Accuracy:", accuracy_score(y_val, KNN.predict(X_val)))
print("Test Accuracy:", accuracy_score(y_test, y_pred_KNN))
print("Recall:", recall_score(y_test, y_pred_KNN))
print("Precision:", precision_score(y_test, y_pred_KNN))
print("Confusion matrix:\n", confusion_matrix(y_test, y_pred_KNN))
print("Classification report:\n", classification_report(y_test, y_pred_KNN))


# ## Naive Bayes Model

# ### The best values for HyperParameters using GridSearchCV 

# **alpha**: This hyperparameter is a smoothing parameter that prevents zero probabilities. It represents the additive smoothing parameter, where alpha = 0 corresponds to no smoothing and alpha = 1 is Laplace smoothing. ***The default value is alpha = 1.0***.
# 
# **binarize**: This hyperparameter is used to binarize the input features, where any value greater than binarize is set to 1 and any value less than or equal to binarize is set to 0. If binarize = None, the input features are not binarized. The ***default value is binarize = 0.0***.

# In[46]:


# from sklearn.naive_bayes import BernoulliNB
# from sklearn.model_selection import GridSearchCV

# # define the parameter grid to search over
# param_grid = {
#     'alpha': [0.1, 0.5, 1.0, 2.0, 5.0],
#     'binarize': [0.0, 0.5, 1.0],
# }

# # create a Bernoulli Naive Bayes model instance
# bnb = BernoulliNB()

# # create a GridSearchCV instance with the parameter grid and the Bernoulli Naive Bayes model
# grid_search = GridSearchCV(estimator=bnb, param_grid=param_grid, cv=5, scoring='accuracy')

# # fit the GridSearchCV instance on the training data
# grid_search.fit(X_train, y_train)
# print(grid_search.best_estimator_)
# y_pred_grid_search = grid_search.predict(X_test)
# # print the best hyperparameters and corresponding score
# print("Train Score: ",grid_search.fit(X_train, y_train).score(X_train, y_train))
# print("Validation Accuracy:", accuracy_score(y_val, grid_search.predict(X_val)))
# print("Test Accuracy:", accuracy_score(y_test, y_pred_grid_search))
# print("Best Hyperparameters: ", grid_search.best_params_)
# print("Best Score: ", grid_search.best_score_)


# **best values** :
#                alpha=0.1,
#                binarize=0.5

# ### Train Naive Bayes model on the training data

# In[47]:


BNB = BernoulliNB(alpha=0.1,binarize=0.5)
BNB.fit(X_train, y_train)


# ### Make predictions on the test data

# In[48]:


y_pred_BNB = BNB.predict(X_test)


# ### Evaluate the performance of the model

# In[49]:


train_acc.append(BNB.score(X_train, y_train))
val_acc.append(accuracy_score(y_val, BNB.predict(X_val)))
test_acc.append(accuracy_score(y_test, y_pred_BNB))
print("Train Accuracy:", BNB.score(X_train, y_train))
print("Validation Accuracy:", accuracy_score(y_val, BNB.predict(X_val)))
print("Test Accuracy:", accuracy_score(y_test, y_pred_BNB))
print("Recall:", recall_score(y_test, y_pred_BNB))
print("Precision:", precision_score(y_test, y_pred_BNB))
print("Confusion matrix:\n", confusion_matrix(y_test, y_pred_BNB))
print("Classification report:\n", classification_report(y_test, y_pred_BNB))


# ## Decision Tree Classifier Model

# ### The best values for HyperParameters using GridSearchCV 

# **criterion**: This hyperparameter determines the function used to measure the quality of a split. The two options are 'gini' (Gini impurity) and 'entropy' (information gain). ***The default value is 'gini'***.
# 
# **max_depth**: This hyperparameter determines the maximum depth of the decision tree. ***The default value is None***, which means that nodes are expanded until all leaves are pure or until all leaves contain less than min_samples_split samples.
# 
# **splitter**: This hyperparameter determines the strategy used to choose the split at each node. The two options are 'best' (choose the best split) and 'random' (choose the best random split). ***The default value is 'best'***.

# In[50]:


# # define the parameter grid to search over
# param_grid = {
# #     'criterion': ['gini', 'entropy'],
# #     'max_depth': [3, 5, 10,15, None],
# #     'splitter': ['best', 'random'],
# }

# # create a decision tree model instance
# dt = DecisionTreeClassifier()

# # create a GridSearchCV instance with the parameter grid and the decision tree model
# grid_search = GridSearchCV(estimator=dt, param_grid=param_grid, cv=5, scoring='accuracy')

# # fit the GridSearchCV instance on the training data
# grid_search.fit(X_train, y_train)
# print(grid_search.best_estimator_)
# y_pred_grid_search = grid_search.predict(X_test)
# # print the best hyperparameters and corresponding score
# print("Train Score: ",grid_search.fit(X_train, y_train).score(X_train, y_train))
# print("Validation Accuracy:", accuracy_score(y_val, grid_search.predict(X_val)))
# print("Test Accuracy:", accuracy_score(y_test, y_pred_grid_search))
# print("Best Hyperparameters: ", grid_search.best_params_)
# print("Best Score: ", grid_search.best_score_)


# **best values** :
#                criterion='entropy',
#                max_depth=15,
#                splitter='random

# ### Train Decision Tree Classifier

# In[51]:


DT = DecisionTreeClassifier(criterion='entropy',
                           max_depth=15,
                           splitter='random',
                           min_samples_split=5)
DT.fit(X_train, y_train)


# ### Make predictions on the test data

# In[52]:


y_pred_DT = DT.predict(X_test)


# ### Evaluate the performance of the model

# In[53]:


train_acc.append(DT.score(X_train, y_train))
val_acc.append(accuracy_score(y_val, DT.predict(X_val)))
test_acc.append(accuracy_score(y_test, y_pred_DT))
print("Train Accuracy:", DT.score(X_train, y_train))
print("Validation Accuracy:", accuracy_score(y_val, DT.predict(X_val)))
print("Test Accuracy:", accuracy_score(y_test, y_pred_DT))
print("Recall:", recall_score(y_test, y_pred_DT))
print("Precision:", precision_score(y_test, y_pred_DT))
print("Confusion matrix:\n", confusion_matrix(y_test, y_pred_DT))
print("Classification report:\n", classification_report(y_test, y_pred_DT))
CM=confusion_matrix(y_test, y_pred_DT)


# ### Importance of Selected features for model 

# In[54]:


df = pd.DataFrame(DT.feature_importances_)
df.index=X_train.columns
df = df.rename(columns={0: 'Decision Tree'})
plt.figure(figsize=(10, 10))
ax = df.plot.barh()
ax.set_xlabel('Importance', fontsize=12)
ax.set_ylabel('Features', fontsize=12)
ax.set_yticklabels(ax.get_yticklabels(), fontsize=10)
ax.tick_params(axis='x', labelsize=10)
plt.title('Feature Importances', fontsize=16)
plt.show()


# ## Random Forest Classifier Model

# ### The best values for HyperParameters using GridSearchCV 

# **n_estimators**: This hyperparameter determines the number of decision trees in the random forest. ***The default value is 100***.
# 
# **max_depth**: This hyperparameter determines the maximum depth of each decision tree. ***The default value is None***, which means that nodes are expanded until all leaves are pure or until all leaves contain less than min_samples_split samples.

# In[55]:


# # define the parameter grid to search over
# param_grid = {
#     'n_estimators': [50, 100, 200], 
#     'max_depth'; [3 , 5 ,7 , 10 , 12 , None]
# }

# # create a Random Forest model instance
# rf = RandomForestClassifier()

# # create a GridSearchCV instance with the parameter grid and the Random Forest model
# grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=5, scoring='accuracy')

# # fit the GridSearchCV instance on the training data
# grid_search.fit(X_train, y_train)
# print(grid_search.best_estimator_)
# y_pred_grid_search = grid_search.predict(X_test)
# # print the best hyperparameters and corresponding score
# print("Train Score: ",grid_search.fit(X_train, y_train).score(X_train, y_train))
# print("Validation Accuracy:", accuracy_score(y_val, grid_search.predict(X_val)))
# print("Test Accuracy:", accuracy_score(y_test, y_pred_grid_search))
# print("Best Hyperparameters: ", grid_search.best_params_)
# print("Best Score: ", grid_search.best_score_)


# **best values** :
#                n_estimators = 100 ***(Default)***,
#                max_depth = 12

# ### Train Random Forest classifier on the training data

# In[56]:


RFC = RandomForestClassifier( max_depth = 12)
RFC.fit(X_train, y_train)


# ### Make predictions on the test data

# In[57]:


y_pred_RFC = RFC.predict(X_test)


# ### Evaluate the performance of the model

# In[58]:


train_acc.append(RFC.score(X_train, y_train))
val_acc.append(accuracy_score(y_val, RFC.predict(X_val)))
test_acc.append(accuracy_score(y_test, y_pred_RFC))
print("Train Accuracy:", RFC.score(X_train, y_train))
print("Validation Accuracy:", accuracy_score(y_val, RFC.predict(X_val)))
print("Test Accuracy:", accuracy_score(y_test, y_pred_RFC))
print("Recall:", recall_score(y_test, y_pred_RFC))
print("Precision:", precision_score(y_test, y_pred_RFC))
print("Confusion matrix:\n", confusion_matrix(y_test, y_pred_RFC))
print("Classification report:\n", classification_report(y_test, y_pred_RFC))


# ### Importance of Selected features for model 

# In[59]:


df = pd.DataFrame(RFC.feature_importances_)
df.index=X_train.columns
df = df.rename(columns={0: 'Random Forest'})
plt.figure(figsize=(10, 10))
ax = df.plot.barh()
ax.set_xlabel('Importance', fontsize=12)
ax.set_ylabel('Features', fontsize=12)
ax.set_yticklabels(ax.get_yticklabels(), fontsize=10)
ax.tick_params(axis='x', labelsize=10)
plt.title('Feature Importances', fontsize=16)
plt.show()


# ## Gradient Boosting Classifier Model

# ### The best values for HyperParameters using GridSearchCV 

# **n_estimators**: This hyperparameter determines the number of weak learners (decision trees) in the boosting process. ***The default value is 100***.
# 
# **learning_rate**: This hyperparameter controls the contribution of each weak learner to the final prediction. A smaller learning rate results in a slower learning process but can lead to better generalization performance. ***The default value is 0.1***.
# 

# In[60]:


# # define the parameter grid to search over
# param_grid = {
#     'n_estimators': [50, 100, 200, 1000],
#     'learning_rate': [0.01, 0.1, 1]
# }

# # create a Gradient Boosting model instance
# gb = GradientBoostingClassifier()

# # create a GridSearchCV instance with the parameter grid and the Gradient Boosting model
# grid_search = GridSearchCV(estimator=gb, param_grid=param_grid, cv=5, scoring='accuracy')

# # fit the GridSearchCV instance on the training data
# grid_search.fit(X_train, y_train)
# print(grid_search.best_estimator_)
# y_pred_grid_search = grid_search.predict(X_test)
# # print the best hyperparameters and corresponding score
# print("Train Score: ",grid_search.fit(X_train, y_train).score(X_train, y_train))
# print("Validation Accuracy:", accuracy_score(y_val, grid_search.predict(X_val)))
# print("Test Accuracy:", accuracy_score(y_test, y_pred_grid_search))
# print("Best Hyperparameters: ", grid_search.best_params_)
# print("Best Score: ", grid_search.best_score_)


# **best values** :
#                n_estimators = 100 ***(Default)*** ,
#         learning_rate = 0.1 ***(Default)***

# ### Train Gradient Boosting classifier on the training data

# In[61]:


GBC=GradientBoostingClassifier()
GBC.fit(X_train,y_train)


# ### Make predictions on the test data

# In[62]:


y_pred_GBC=GBC.predict(X_test)


# ### Evaluate the performance of the model

# In[63]:


train_acc.append(GBC.score(X_train, y_train))
val_acc.append(accuracy_score(y_val, GBC.predict(X_val)))
test_acc.append(accuracy_score(y_test, y_pred_GBC))
print("Train Accuracy:", GBC.score(X_train, y_train))
print("Validation Accuracy:", accuracy_score(y_val, GBC.predict(X_val)))
print("Test Accuracy:", accuracy_score(y_test, y_pred_GBC))
print("Recall:", recall_score(y_test, y_pred_GBC))
print("Precision:", precision_score(y_test, y_pred_GBC))
print("Confusion matrix:\n", confusion_matrix(y_test, y_pred_GBC))
print("Classification report:\n", classification_report(y_test, y_pred_GBC))


# ### Importance of Selected features for model 

# In[64]:


df = pd.DataFrame(GBC.feature_importances_)
df.index=X_train.columns
df = df.rename(columns={0: 'Random Forest'})
plt.figure(figsize=(10, 10))
ax = df.plot.barh()
ax.set_xlabel('Importance', fontsize=12)
ax.set_ylabel('Features', fontsize=12)
ax.set_yticklabels(ax.get_yticklabels(), fontsize=10)
ax.tick_params(axis='x', labelsize=10)
plt.title('Feature Importances', fontsize=16)
plt.show()


# ## Logistic Regression Classifier Model

# ### The best values for HyperParameters using GridSearchCV 

# **C**: This hyperparameter determines the inverse of the regularization strength. Smaller values of C result in stronger regularization, which can help to reduce overfitting. ***The default value is 1.0***.
# 
# **penalty**: This hyperparameter determines the type of regularization used. The two options are 'l1' (L1 regularization) and 'l2' (L2 regularization). L1 regularization can be useful for feature selection, while L2 regularization can be useful for reducing the impact of outliers. ***The default value is 'l2'***.
# 
# **multi_class**: This hyperparameter determines the method used to handle multi-class classification problems. The three options are 'ovr' (one-vs-rest), 'multinomial' (softmax regression), and 'auto' (automatically select the best method based on the data and solver). ***The default value is 'ovr'***.

# In[65]:


# # define the parameter grid to search over
# param_grid = {
#     'C': [0.001, 0.01, 0.1, 1, 10, 100],
#     'solver' : ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga'],
#     'penalty': ['l1', 'l2'],
#     'multi_class': ['ovr', 'multinomial']
# }

# # create a logistic regression model instance
# lr = LogisticRegression(max_iter=10000)

# # create a GridSearchCV instance with the parameter grid and the logistic regression model
# grid_search = GridSearchCV(estimator=lr, param_grid=param_grid, cv=5, scoring='accuracy',verbose=1,return_train_score=True)

# # fit the GridSearchCV instance on the training data
# grid_search.fit(X_train, y_train)
# print(grid_search.best_estimator_)
# y_pred_grid_search = grid_search.predict(X_test)
# # print the best hyperparameters and corresponding score
# print("Train Score: ",grid_search.fit(X_train, y_train).score(X_train, y_train))
# print("Validation Accuracy:", accuracy_score(y_val, grid_search.predict(X_val)))
# print("Test Accuracy:", accuracy_score(y_test, y_pred_grid_search))
# print("Best Hyperparameters: ", grid_search.best_params_)
# print("Best Score: ", grid_search.best_score_)


# **best values** :
#                C=0.1,
#                penalty='l2' ***(Default)*** ,
#                multi_class='multinomial'

# ### Train Logistic Regression

# In[66]:


LR = LogisticRegression(C=0.1,max_iter=10000)
LR.fit(X_train, y_train)


# ### Make predictions on the test data

# In[67]:


y_pred_LR = LR.predict(X_test)


# ### Evaluate the performance of the model

# In[68]:


train_acc.append(LR.score(X_train, y_train))
val_acc.append(accuracy_score(y_val, LR.predict(X_val)))
test_acc.append(accuracy_score(y_test, y_pred_LR))
print("Train Accuracy:", LR.score(X_train, y_train))
print("Validation Accuracy:", accuracy_score(y_val, LR.predict(X_val)))
print("Test Accuracy:", accuracy_score(y_test, y_pred_LR))
print("Recall:", recall_score(y_test, y_pred_LR))
print("Precision:", precision_score(y_test, y_pred_LR))
print("Confusion matrix:\n", confusion_matrix(y_test, y_pred_LR))
print("Classification report:\n", classification_report(y_test, y_pred_LR))


# ## SVM Classifier Model

# ### Train SVM classifier on the training data

# In[69]:


SVC = SVC(kernel = 'poly', degree = 4, random_state = 2, C = 50)
SVC.fit(X_train, y_train)


# ### Make predictions on the test data

# In[70]:


y_pred_SVC = SVC.predict(X_test)


# ### Evaluate the performance of the model

# In[71]:


train_acc.append(SVC.score(X_train, y_train))
val_acc.append(accuracy_score(y_val, SVC.predict(X_val)))
test_acc.append(accuracy_score(y_test, y_pred_SVC))
print("Train Accuracy:", SVC.score(X_train, y_train))
print("Validation Accuracy:", accuracy_score(y_val, SVC.predict(X_val)))
print("Test Accuracy:", accuracy_score(y_test, y_pred_SVC))
print("Recall:", recall_score(y_test, y_pred_SVC))
print("Precision:", precision_score(y_test, y_pred_SVC))
print("Confusion matrix:\n", confusion_matrix(y_test, y_pred_SVC))
print("Classification report:\n", classification_report(y_test, y_pred_SVC))


# # Comparison Between Models

# In[72]:


# Set the names of the models
models_names = ['KNN', 'BernoulliNB', 'DecisionTree', 'RandomForest', 'GradientBoosting', 'LogisticRegression', 'SVM']
models=[KNN,BNB,DT,RFC,GBC,LR,SVC]

# Set the position of the bars on the x-axis
x = np.arange(len(models_names))

# Define the colors for the bars
train_color = '#1f77b4'
val_color = '#ff7f0e'

# Set the width of the bars
width = 0.35

# Create the figure and axes objects
fig, ax = plt.subplots(figsize=(50, 30))

# Create the bars for train accuracy
rects1 = ax.bar(x, train_acc, width, label='Train Accuracy', color=train_color)

# Create the bars for validation accuracy
rects2 = ax.bar(x + width, val_acc, width, label='Validation Accuracy', color=val_color)

# Add labels, title, and legend
ax.set_xlabel('\nModels', fontsize=50)
ax.set_ylabel('\nAccuracy', fontsize=50)
ax.set_title('Comparison of Training and Validation Accuracies', fontsize=40)
ax.set_xticks(x + width / 2)
ax.set_xticklabels(models_names, fontsize=28)
ax.legend(fontsize=28)

# Add values above each bar
def autolabel(rects):
    for rect in rects:
        height = rect.get_height()
        ax.annotate('{:.3f}'.format(height),
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=24)

autolabel(rects1)
autolabel(rects2)

# Find the highest train accuracy value and its corresponding model name
highest_train_acc = max(train_acc)
highest_train_acc_model = models_names[train_acc.index(highest_train_acc)]
print(f'The highest training accuracy is {highest_train_acc:.3f} and is achieved by the {highest_train_acc_model} model.')
Best_Model = models[train_acc.index(highest_train_acc)]

# Set the font size of the y-tick labels
ax.tick_params(axis='y', which='major', labelsize=28)

# Show the plot
plt.show()


# # Best Model 

#  ## Train accuracy vs Validation accuracy vs Test accuracy

# In[73]:


# Set the names of the accuracies
acc_names = ['Train Accuracy', 'Validation Accuracy', 'Test Accuracy']

# Set the accuracies
train_accs = Best_Model.score(X_train,y_train)
val_accs = Best_Model.score(X_val,y_val)
test_accs = Best_Model.score(X_test,y_test)

# Set the position of the bars on the x-axis
x = np.arange(len(acc_names))

# Define the colors for the bars
train_color = '#1f77b4'  # blue
val_color = '#ff7f0e'  # orange
test_color = '#2ca02c'  # green

# Set the width of the bars
width = 0.35

# Create the figure and axes objects
fig, ax = plt.subplots(figsize=(10, 8))

# Create the bars for accuracies
rects = ax.bar(x, [train_accs, val_accs, test_accs], width, color=[train_color, val_color, test_color])

# Add labels, title, and legend
ax.set_xlabel('\nAccuracy Type', fontsize=14)
ax.set_ylabel('\nAccuracy', fontsize=14)
ax.set_title('Training, Validation, and Test Accuracies of Best Model\n', fontsize=16)
ax.set_xticks(x)
ax.set_xticklabels(acc_names, fontsize=12)
ax.tick_params(axis='y', which='major', labelsize=12)

# Add values above each bar
def autolabel(rects):
    for rect in rects:
        height = rect.get_height()
        ax.annotate('{:.3f}'.format(height),
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=12)

autolabel(rects)

# Show the plot
plt.show()


# ## Confusion Matrix

# In[74]:


import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# Define the confusion matrix
conf_matrix = CM

# Define the class labels
classes = ['Negative', 'Positive']

# Create the heatmap
fig, ax = plt.subplots(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, cmap='Blues', fmt='d', xticklabels=classes, yticklabels=classes)

# Set the axis labels and title
ax.set_xlabel('Predicted Labels')
ax.set_ylabel('True Labels')
ax.set_title('Confusion Matrix')

# Show the plot
plt.show()


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





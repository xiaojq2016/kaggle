#问题描述
#https://www.kaggle.com/c/titanic

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score
import pydotplus
from sklearn import tree

#load the data
pd.set_option('display.width', None)  # 设置字符显示宽度
pd.set_option('display.max_rows', None)  # 设置显示最大行
train_df = pd.read_csv("./data/0/train.csv")
test_df = pd.read_csv("./data/0/test.csv")

#Data Analysis
# print(train_df.describe())
# #print(train_df.head(5))
print(train_df.isnull().sum())

#Data preprocessing
dataset = [train_df,test_df]
for data in dataset:
    cabin_map = {"A":1,"B":2,"C":3,"D":4,"E":5,"F":6,"G":7,"T":8}
    data['Deck'] = data.Cabin.str.extract('([A-Za-z]+)', False)
    data['Deck'] = data['Deck'].map(cabin_map)
    data['Deck'] = data.Deck.fillna(0)
    data['Deck'] = data.Deck.astype(int)
train_df = train_df.drop(['Cabin'], 1)
test_df = test_df.drop(['Cabin'], 1)

dataset = [train_df, test_df]
for data in dataset:
    data['Title'] = data.Name.str.extract(' ([A-Za-z]+)\.', False)
    #train_data['Title']
    title_map = {"Mr":1,"Mrs":2,"Miss":3,"Ms":4,"Rare":5}
    data['Title'] = data['Title'].replace(['Master','Don','Rev','Dr',
                                           'Major','Lady','Sir', 'Col',
                                           'Capt','Countess','Jonkheer'], 'Rare')
    data['Title'] = data['Title'].replace(['Mme'],'Mrs')
    data['Title'] = data['Title'].replace(['Mlle'],'Miss')
    data['Title'] = data['Title'].map(title_map)
    data['Title'] = data['Title'].fillna(0)
    data['Title'] = data.Title.astype(int)

# for data in dataset:
#     data['Name'] = data['Name'].str.len()
#     data['Name'].astype(int)

train_df = train_df.drop(['Name'], 1)
test_df = test_df.drop(['Name'], 1)

ports = {"S":1,"C":2,"Q":3}
dataset = [train_df, test_df]

for data in dataset:
    data['Embarked'] = data['Embarked'].map(ports)
    data['Embarked'] = data.Embarked.fillna(0)
    data['Embarked'] = data.Embarked.astype(int)

gender_map = {"male":1 , "female":2}
dataset = [train_df,test_df]

for data in dataset:
    data['Sex'] = data['Sex'].map(gender_map)

dataset = [train_df,test_df]

for data in dataset:
    mean = data['Age'].mean()
    std = data['Age'].std()
    is_null = data['Age'].isnull().sum()
    rand_age = np.random.randint(mean-std,mean+std, is_null)
    #rand_age
    age_slice = data['Age'].copy()
    age_slice[np.isnan(age_slice)] = rand_age
    data['Age'] = age_slice
    data['Age'] = data.Age.astype(int)
    # train_data['Age'] = train_data.Age.astype(int)

dataset = [train_df,test_df]

# for data in dataset:
#     data.loc[ data['Age'] <= 11, 'Age'] = 0
#     data.loc[ (data['Age'] >11) & (data['Age']<=18),'Age'] = 1
#     data.loc[ (data['Age'] >18) & (data['Age']<=22),'Age'] = 2
#     data.loc[ (data['Age'] >22) & (data['Age']<=27),'Age'] = 3
#     data.loc[ (data['Age'] >27) & (data['Age']<=33),'Age'] = 4
#     data.loc[ (data['Age'] >33) & (data['Age']<=40),'Age'] = 5
#     data.loc[ (data['Age'] >40) & (data['Age']<=66),'Age'] = 6
#     data.loc[ data['Age'] >66 ,'Age'] = 7

dataset = [train_df,test_df]

for data in dataset:
    # data.loc[ data['Fare'] <= 7.91 , 'Fare'] = 0
    # data.loc[ (data['Fare'] > 7.91) & (data['Fare'] <= 14.454) , 'Fare'] = 1
    # data.loc[ (data['Fare'] > 14.454) & (data['Fare'] <= 31.00) , 'Fare'] = 2
    # data.loc[ (data['Fare'] > 31.00) & (data['Fare'] <= 100) , 'Fare'] = 3
    # data.loc[ (data['Fare'] > 100) & (data['Fare'] <= 250) , 'Fare'] = 4
    # data.loc[ data['Fare'] > 250 , 'Fare'] = 5
    data['Fare'] = data['Fare'].fillna(-1)
    data['Fare'] = data.Fare.astype(float)

train_df = train_df.drop(['Ticket'], 1)
test_df = test_df.drop(['Ticket'], 1)

train_df = train_df.drop(['PassengerId'], 1)
test_df = test_df.drop(['PassengerId'], 1)

dataset = [train_df,test_df]

for data in dataset:
    data['relatives'] = data['SibSp']+data['Parch']
    data.loc[data['relatives']>0,'not_alone'] = 0
    data.loc[data['relatives'] == 0,'not_alone'] = 1
    data['not_alone'] = data['not_alone'].astype(int)

y_train = train_df['Survived']
x_train = train_df.drop(['Survived'], 1)
x_test = test_df

#Random Forest
clf = RandomForestClassifier(min_samples_leaf=2, n_estimators=100)
print(cross_val_score(clf, x_train, y_train, cv=10).mean())
clf = clf.fit(x_train, y_train)
clf.predict(x_test)

#Decision tree
dec_clf = DecisionTreeClassifier()
print(cross_val_score(dec_clf, x_train, y_train, cv=10).mean())
dec_clf = dec_clf.fit(x_train, y_train)
dec_clf.predict(x_test)
dot_data = tree.export_graphviz(dec_clf, out_file=None,
                                filled=True, rounded=True,
                                special_characters=True)
graph = pydotplus.graph_from_dot_data(dot_data)
graph.write_pdf("0.pdf")
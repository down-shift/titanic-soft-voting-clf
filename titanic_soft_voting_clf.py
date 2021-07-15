import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import f1_score
from sklearn.model_selection import GridSearchCV

from sklearn.svm import LinearSVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, VotingClassifier

def replace_val(data, column, mapping):
    data = data.replace({column: mapping})
    return data

train_data = pd.read_csv(r'C:\Users\maxfe\Documents\datasets\titanic\train.csv')
test_data = pd.read_csv(r'C:\Users\maxfe\Documents\datasets\titanic\test.csv')
genders = {'male': 0, 'female': 1}
train_data, test_data = replace_val(train_data, 'Sex', genders), replace_val(test_data, 'Sex', genders)

ports = {'S': 0, 'C': 1, 'Q': 2}
train_data, test_data = replace_val(train_data, 'Embarked', ports), replace_val(test_data, 'Embarked', ports)

train_data['DivFareByClass'] = train_data['Fare'] / train_data['Pclass']
test_data['DivFareByClass'] = test_data['Fare'] / test_data['Pclass']
train_data['MultAgeAndSex'] = train_data['Age'] * train_data['Sex']
test_data['MultAgeAndSex'] = test_data['Age'] * test_data['Sex']

attributes = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'DivFareByClass', 'MultAgeAndSex', 'Embarked']
X_train, y_train = train_data[attributes], train_data['Survived']
X_test = test_data[attributes]

corr_matrix = train_data.corr() 
corr_matrix['Survived'].sort_values(ascending=False) # jupyter nb only

num_pipeline = Pipeline([
    ('imputer', SimpleImputer(missing_values=np.nan, strategy='mean')),
    ('scaler', StandardScaler())
])
X_train = pd.DataFrame(num_pipeline.fit_transform(X_train))
X_test = pd.DataFrame(num_pipeline.fit_transform(X_test))

lin_svm_clf = SVC(kernel='linear', C=2, probability=True)
#lin_svm_clf.fit(X_train, y_train)
#y_pred = pd.DataFrame(lin_svm_clf.predict(X_test))
#y_train_pred = lin_svm_clf.predict(X_train)

knn_clf = KNeighborsClassifier(n_neighbors=29)
#knn_clf.fit(X_train, y_train)
#y_pred = pd.DataFrame(knn_clf.predict(X_test))
#y_train_pred = knn_clf.predict(X_train)

svm_clf = SVC(probability=True)
#params = {'kernel': ['linear', 'rbf', 'poly'], 'C': [1, 5, 10, 100]}
#svm_clf = GridSearchCV(svm_clf, params)
#svm_clf.fit(X_train, y_train)
#svm_clf.best_params_
#y_pred = pd.DataFrame(svm_clf.predict(X_test))
#y_train_pred = svm_clf.predict(X_train)

rf_clf_1 = RandomForestClassifier(n_estimators=100, max_depth=15, min_samples_leaf=4, random_state=14)
#params = {'n_estimators': [100, 200, 500]}
#rf_clf = GridSearchCV(rf_clf, params)
#rf_clf.fit(X_train, y_train)
#y_pred = rf_clf.predict(X_test)
#y_train_pred = rf_clf.predict(X_train)
#rf_clf.best_params_

rf_clf_2 = RandomForestClassifier(n_estimators=200, max_depth=15, min_samples_leaf=4, random_state=14)

rf_clf_3 = RandomForestClassifier(n_estimators=300, max_depth=15, min_samples_leaf=4, random_state=14)

log_clf = LogisticRegression()
#log_clf.fit(X_train, y_train)
#y_train_pred = log_clf.predict(X_train)

voting_clf = VotingClassifier(estimators=[('lr', log_clf), ('rf1', rf_clf_1),
                                          ('svc', svm_clf), ('lin_svc', lin_svm_clf),
                                          ('knn', knn_clf), ('rf2', rf_clf_2),
                                          ('rf3', rf_clf_3)], voting='hard')
voting_clf.fit(X_train, y_train)
y_pred = pd.DataFrame(voting_clf.predict(X_test))
y_train_pred = voting_clf.predict(X_train)

print(f1_score(y_train, y_train_pred))

final_df = pd.DataFrame(np.c_[test_data['PassengerId'], y_pred])
final_df.reset_index(drop=True, inplace=True)
final_df.rename(columns={0: 'PassengerId', 1: 'Survived'}, inplace=True)

final_df = final_df.to_csv(r'C:\Users\maxfe\Documents\datasets\titanic\hard_voting_clf_1.csv', index=False)


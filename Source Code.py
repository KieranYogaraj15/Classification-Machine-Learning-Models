#Import the Diabetes Dataset
import pandas as pd

diabetes = pd.read_csv("diabetes.csv", sep=",")


#Data Exploration
diabetes.head()
diabetes.shape
diabetes.info() #No missing data

#What is the distribution of the Outcome variable?
import matplotlib.pyplot as plt
plt.bar(["0","1"], diabetes.Outcome.value_counts())

#What is the relationship between Outcome and the other variables?
import seaborn as sn
sn.heatmap(diabetes.corr(), annot=True) #View the correlation between all variables


#Feature Selection
X = diabetes[['Glucose','BMI','Age']]
y = diabetes.Outcome


#Split Dataset into Train and Test datasets
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X,y,train_size = 0.8,random_state=0)



#KNN Model
#Building the model
from sklearn.neighbors import KNeighborsClassifier

model = KNeighborsClassifier(3) #K=3
model.fit(X_train,y_train)

y_pred = model.predict(X_test)


#Evaluating the model
from sklearn.metrics import confusion_matrix
from sklearn import metrics
from sklearn.metrics import classification_report

cm_knn = confusion_matrix(y_test, y_pred)
print(cm_knn)
 
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
print("Precision:",metrics.precision_score(y_test, y_pred, average ='weighted'))
print("Recall:",metrics.recall_score(y_test, y_pred, average = 'weighted')) 
print("F1-score:",metrics.f1_score(y_test, y_pred, average = 'weighted'))  

report_knn = classification_report(y_test, y_pred)
print(report_knn)     
    

#Parameter Tunning using GridSearchCV
from sklearn.model_selection import GridSearchCV

model = KNeighborsClassifier()
params = {'n_neighbors': range(1,10)}
                                   
grs = GridSearchCV(model, param_grid=params, cv=5)
grs.fit(X_train, y_train)

print("Best Hyper Parameters:",grs.best_params_) #The best K is 8

y_pred=grs.predict(X_test)


#Evaluate model with new K
cm_knn = confusion_matrix(y_test, y_pred)
print(cm_knn)

print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
print("Precision:",metrics.precision_score(y_test, y_pred, average ='weighted'))
print("Recall:",metrics.recall_score(y_test, y_pred, average = 'weighted')) 
print("F1-score:",metrics.f1_score(y_test, y_pred, average = 'weighted'))       
                                                   
report_knn = classification_report(y_test, y_pred)
print(report_knn)                                                 



#Decision Tree Model
#Building the model
from sklearn.tree import DecisionTreeClassifier 
model = DecisionTreeClassifier(random_state=21)

model.fit(X_train,y_train)
y_pred= model.predict(X_test)


#Evaluating the model
from sklearn.metrics import confusion_matrix
from sklearn import metrics
from sklearn.metrics import classification_report

cm_tree = confusion_matrix(y_test, y_pred)
print(cm_tree)
 
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
print("Precision:",metrics.precision_score(y_test, y_pred, average ='weighted'))
print("Recall:",metrics.recall_score(y_test, y_pred, average = 'weighted')) 
print("F1-score:",metrics.f1_score(y_test, y_pred, average = 'weighted'))

report_tree = classification_report(y_test, y_pred)
print(report_tree)       


#Fine Tuning the Parameter
from sklearn.model_selection import GridSearchCV

model = DecisionTreeClassifier(random_state=0)

params = {'criterion': ['gini', 'entropy'], 'max_depth': range(1,10)}

grs = GridSearchCV(model, param_grid=params, cv=5)

grs.fit(X_train, y_train)

print("Best Hyper Parameters:",grs.best_params_) #max_depth = 4 and criterion = "gini"

model = grs.best_estimator_
y_pred=model.predict(X_test)


#Evaluating the model with the new parameters
cm_tree = confusion_matrix(y_test, y_pred)
print(cm_tree)
 
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
print("Precision:",metrics.precision_score(y_test, y_pred, average ='weighted'))
print("Recall:",metrics.recall_score(y_test, y_pred, average = 'weighted')) 
print("F1-score:",metrics.f1_score(y_test, y_pred, average = 'weighted'))

report_tree = classification_report(y_test, y_pred)
print(report_tree)       





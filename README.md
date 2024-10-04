# Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee

## AIM:
To write a program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import the libraries and read the data frame using pandas.

2.Calculate the null values present in the dataset and apply label encoder.

3.Determine test and training data set and apply decison tree regression in dataset.

4.Calculate Mean square error,data prediction and r2.

## Program:
```
/*
Program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee.
Developed by: YUVARAJ V
RegisterNumber:  212223230252
*/
```
```
import pandas as pd
from sklearn.tree import  plot_tree
data=pd.read_csv("/content/Salary.csv")
data.head()
data.info()
data.isnull().sum()
from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
data["Position"]=le.fit_transform(data["Position"])
data.head()
x=data[["Position","Level"]]
y=data["Salary"]
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=2)
from sklearn.tree import DecisionTreeRegressor
dt=DecisionTreeRegressor()
dt.fit(x_train,y_train)
y_pred=dt.predict(x_test)
from sklearn import metrics
mse=metrics.mean_squared_error(y_test,y_pred)
mse
r2=metrics.r2_score(y_test,y_pred)
r2
dt.predict([[5,6]])
plt.figure(figsize=(20,8))
plot_tree(dt,feature_names=x.columns,filled=True)
plt.show()
```

## Output:

## Head:

![371455811-93d06783-1b85-45f1-be51-540305ac568a](https://github.com/user-attachments/assets/edd7bdea-fc91-4c06-b272-16ffa6008a97)

## Mean Squared Error:

![371455923-99f11628-9360-4246-8f2e-8a64f2a9342f](https://github.com/user-attachments/assets/8e816d9a-7a6a-4183-b12d-7ac105d715a7)

## Predicted Value:

![371456086-0873c340-2c3b-4c90-a3f3-2a6018c532c4](https://github.com/user-attachments/assets/e2a41aba-6729-4df6-8103-c481a6dc8a01)

## Decision Tree:

![371456227-11902ca7-77a5-4970-bc1b-2108dc35b6ab](https://github.com/user-attachments/assets/deb9a7a0-74ef-408f-90be-d135c0a43d77)



## Result:
Thus the program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee is written and verified using python programming.

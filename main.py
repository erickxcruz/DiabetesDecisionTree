import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeRegressor
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import plot_tree
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import _plot
import matplotlib.pyplot as plt
import graphviz

df = pd.read_csv("data3.csv", skiprows=1, header=None)
df.columns = ['Diabetes_012', 'HighBP', 'HighChol', 'CholCheck', 'BMI', 'Smoker', 'Stroke', 'HeartDiseaseorAttack',
              'PhysActivity', 'Fruits', 'Veggies', 'HvyAlcoholConsump', 'AnyHealthcare', 'NoDocbcCost', 'GenHlth',
              'MentHlth', 'PhysHlth', 'DiffWalk', 'Sex', 'Age', 'Education', 'Income']
print(df.head())

X = df.drop('Diabetes_012', axis=1).copy()
y = df['Diabetes_012']
print(X)
print(y)

#print(X.dtypes)

# X_encoded = pd.get_dummies(X, columns=['HighBP', 'HighChol', 'CholCheck', 'Smoker', 'Stroke',
#                                      'HeartDiseaseorAttack', 'PhysActivity', 'Fruits', 'Veggies', 'HvyAlcoholConsump',
#                                      'AnyHealthcare', 'NoDocbcCost', 'GenHlth', 'MentHlth', 'PhysHlth', 'DiffWalk',
#                                  'Sex', 'Age', 'Education', 'Income'])
# print(X_encoded.head())
# print(X_encoded.dtypes)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
dtc = DecisionTreeClassifier(random_state=42)
dtc = dtc.fit(X_train, y_train)
plt.figure(figsize=(100, 100), dpi = 100)
plot_tree(dtc, filled=True, rounded=True, class_names=["No Diabetes", "Pre Diabetes", "Diabetes"],
          feature_names=X.columns
          )
plt.title("Decision Tree Trained On Diabetes Factors")
plt.show()
#export_graphviz(dtc, out_file='tree.dot', filled=True, rounded=True, class_names=["No Diabetes", "Pre Diabetes", "Diabetes"], feature_names=X.columns)



import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeRegressor
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import plot_tree
from sklearn.tree import export_graphviz
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import _plot
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import graphviz

df = pd.read_csv("data.csv", skiprows=1, header=None)
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
plt.figure(figsize=(25, 25), dpi = 100)
plot_tree(dtc, filled=True, rounded=True, class_names=["No Diabetes", "Pre Diabetes", "Diabetes"],
          feature_names=X.columns
          )
plt.title("Decision Tree Trained On Diabetes Factors")
##plt.show()
#export_graphviz(dtc, out_file='tree.dot', filled=True, rounded=True, class_names=["No Diabetes", "Pre Diabetes", "Diabetes"], feature_names=X.columns)

pred = dtc.predict(X_test)
print(classification_report(y_test, pred, zero_division=1))
print(confusion_matrix(y_test, pred))

rfc = RandomForestClassifier(n_estimators=100, random_state=42)
rfc.fit(X_train, y_train)
rfc_pred = rfc.predict(X_test)
plt.figure(figsize=(25, 25), dpi = 100)
plot_tree(rfc.estimators_[0], filled=True, rounded=True, class_names=["No Diabetes", "Pre Diabetes", "Diabetes"],
          feature_names=X.columns
          )
plt.title("Random Forest Tree Trained On Diabetes Factors")
print(classification_report(y_test, rfc_pred,  zero_division=1))
print(confusion_matrix(y_test, pred))

#Cost Complexity Pruning Visualize alphas
path = dtc.cost_complexity_pruning_path(X_train, y_train)
ccp_alphas = path.ccp_alphas
ccp_alphas = ccp_alphas[:-1]

dtca = []
for ccp_alpha in ccp_alphas:
    dtc = DecisionTreeClassifier(random_state=0, ccp_alpha=ccp_alpha)
    dtc.fit(X_train, y_train)
    dtca.append(dtc)

train_scores = [dtc.score(X_train, y_train) for dtc in dtca]
test_scores = [dtc.score(X_test, y_test) for dtc in dtca]
fig, ax = plt.subplots()
ax.set_xlabel("alpha")
ax.set_ylabel("accuracy")
ax.set_title("Accuracy vs alpha training and testing sets")
ax.plot(ccp_alphas, train_scores, marker='o', label="train", drawstyle="steps-post")
ax.plot(ccp_alphas, test_scores, marker='o', label="test", drawstyle="steps-post")
ax.legend()
plt.show()

#Cost Complexity Pruning: Cross Validation
dtc = DecisionTreeClassifier(random_state=42, ccp_alpha=0.016)
scores = cross_val_score(dtc, X_train, y_train, cv=5)
df = pd.DataFrame(data={'tree':range(5), 'accuracy': scores})
df.plot(x='tree', y='accuracy', marker='o', linestyle='--')
plt.show()

alpha_loop_values = []

for ccp_alpha in ccp_alphas:
    dtc = DecisionTreeClassifier(random_state=0, ccp_alpha=ccp_alpha)
    scores = cross_val_score(dtc, X_train, y_train, cv=5)
    alpha_loop_values.append([ccp_alpha, np.mean(scores), np.std(scores)])

alpha_results = pd.DataFrame(alpha_loop_values, columns=['alpha', 'mean_accuracy', 'std'])
alpha_results.plot(x='alpha', y='mean_accuracy', yerr ='std', marker='o', linestyle='--')
plt.show()
ideal_ccp_alpha = alpha_results[(alpha_results['alpha'] > 0.0) & (alpha_results['alpha'] < 0.012)]['alpha'].iloc[0]
ideal_ccp_alpha = float(ideal_ccp_alpha)
print(ideal_ccp_alpha)



dtc_pruned = DecisionTreeClassifier(random_state=42, ccp_alpha=ideal_ccp_alpha)
dtc_pruned = dtc_pruned.fit(X_train, y_train)
pruned_pred = dtc_pruned.predict(X_test)

print(classification_report(y_test, pruned_pred,  zero_division=1))
print(confusion_matrix(y_test, pruned_pred))
plt.figure(figsize=(25, 25), dpi = 100)
plot_tree(dtc_pruned, filled=True, rounded=True, class_names=["No Diabetes", "Pre Diabetes", "Diabetes"],
          feature_names=X.columns
          )
plt.title("Pruned Decision Tree Trained On Diabetes Factors")
plt.show()

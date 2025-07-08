import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from xgboost import XGBClassifier
from joblib import dump


df = pd.read_csv('diabetes.csv')
df.info()

# Plot Histogram
df.hist(bins = 30, figsize = (20,20), color = 'b');

# Plot the correlation matrix
correlations = df.corr()
f, ax = plt.subplots(figsize = (20, 20))
sns.heatmap(correlations, annot = True);

y = df['Outcome']
print(y)

X = df.drop(['Outcome'], axis = 1)
print(X)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)

print(X_train.shape)
print(X_test.shape)


# Train an XGBoost classifier model 

xgb_classifier = XGBClassifier(objective ='binary:logistic', eval_metric = 'error', learning_rate = 0.1, max_depth = 1, n_estimators = 10, use_label_encoder=False)
xgb_classifier.fit(X_train, y_train)




# Save the model to a file (use .joblib or .pkl extension)
dump(xgb_classifier, 'diabetes_model_v2.joblib', protocol=4)
# predict the score of the trained model using the testing dataset
result = xgb_classifier.score(X_test, y_test)
print("Accuracy : {}".format(result))

# make predictions on the test data
y_predict = xgb_classifier.predict(X_test)
y_predict

from sklearn.metrics import classification_report
print(classification_report(y_test, y_predict))
# precision is the ratio of TP/(TP+FP)
# recall is the ratio of TP/(TP+FN)
# F-beta score can be interpreted as a weighted harmonic mean of the precision and recall
# where an F-beta score reaches its best value at 1 and worst score at 0. 

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_predict)
plt.figure(figsize=(10,7))
sns.heatmap(cm, fmt = 'd', annot = True)
plt.show()
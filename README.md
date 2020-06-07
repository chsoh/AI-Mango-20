# AI-Mango-20

#h5
https://drive.google.com/drive/folders/1BmxsvSsKn9slMWs5EJqbqoqhEZPGOXL1?usp=sharing

#Adaboost
from sklearn.ensemble import AdaBoostClassifier
model = AdaBoostClassifier(random_state=100)
model.fit(y_train)
y_pred_ada = model.predict(X_test)
y_predprob_ada = model.predict_proba(X_test)
y_predprob_po_ada = y_predprob_ada[:,1]
print("Misclassified sample: %d" % (y_test != y_pred_ada).sum())
print('Accuracy: %.4f' % accuracy_score(y_test, y_pred_ada))
print('Precision: %f' % precision_score(y_test,y_pred_ada))
print('Recall: %f' % recall_score(y_test,y_pred_ada))
print('f1-score: %f' % f1_score(y_test, y_pred_ada))

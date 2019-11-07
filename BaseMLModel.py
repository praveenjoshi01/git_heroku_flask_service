import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report
from sklearn.externals import joblib

df = pd.read_csv('dataset.csv', encoding="latin-1")
X = df['log']
y = df['ErrorCode']
cv = CountVectorizer()
X = cv.fit_transform(X)  # Fit the Data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
# Naive Bayes Classifier
clf = MultinomialNB()
clf.fit(X_train, y_train)
clf.score(X_test, y_test)
y_pred = clf.predict(X_test)
print(classification_report(y_test, y_pred))
# Saved Model
joblib.dump(clf, 'NB_Err_model.pkl')
# Save Vocab
joblib.dump(cv, 'vectorized.pkl')

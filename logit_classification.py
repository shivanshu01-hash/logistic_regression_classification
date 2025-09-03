import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report

# Step 1: Load Data
df = pd.read_csv(r"logit classification.csv")
X = df.iloc[:, [2, 3]]     
y = df.iloc[:, -1]

# Step 2: Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=0)

# Step 3: Feature Scaling
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Step 4: Train Logistic Regression
classifier = LogisticRegression()
classifier.fit(X_train, y_train)

# Step 5: Predictions & Metrics
y_pred = classifier.predict(X_test)
cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:\n", cm)
ac = accuracy_score(y_test, y_pred)
print("Accuracy:", ac)
cr = classification_report(y_test, y_pred)
print("Classification Report:\n", cr)
bias = classifier.score(X_train, y_train)
variance = classifier.score(X_test, y_test)
print("Bias (Train Score):", bias)
print("Variance (Test Score):", variance)
print("Final Model Score (%):", variance * 100)
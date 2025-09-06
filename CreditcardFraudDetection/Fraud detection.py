print("🔍 Credit Card Fraud Detection Script Started")

# 📦 Import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE

#  Load dataset
df = pd.read_csv("D:\data analytics Projects\CreditCardFraudDetection\creditcard.csv")
print("✅ Data loaded:", df.shape)

#  Explore fraud ratio
fraud_ratio = df['Class'].value_counts(normalize=True)[1] * 100
print(f"⚠️ Fraudulent transactions: {fraud_ratio:.4f}%")

#  EDA: Transaction amount distribution
plt.figure(figsize=(6,4))
sns.histplot(df['Amount'], bins=50, kde=True)
plt.title("Transaction Amount Distribution")
plt.xlabel("Amount")
plt.ylabel("Frequency")
plt.tight_layout()
plt.show()

#  Correlation heatmap
plt.figure(figsize=(12,10))
sns.heatmap(df.corr(), cmap='coolwarm', annot=False)
plt.title("Feature Correlation Heatmap")
plt.tight_layout()
plt.show()

#  Feature scaling
scaler = StandardScaler()
df['scaled_amount'] = scaler.fit_transform(df[['Amount']])
df['scaled_time'] = scaler.fit_transform(df[['Time']])
df.drop(['Amount', 'Time'], axis=1, inplace=True)

#  Define features and target
X = df.drop('Class', axis=1)
y = df['Class']

#  Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, stratify=y, random_state=42)

#  Handle imbalance with SMOTE
sm = SMOTE(random_state=42)
X_train_res, y_train_res = sm.fit_resample(X_train, y_train)
print("📊 After SMOTE:", pd.Series(y_train_res).value_counts())

# Train logistic regression
model = LogisticRegression(max_iter=1000)
model.fit(X_train_res, y_train_res)

#  Predict and evaluate
y_pred = model.predict(X_test)
print("\n📋 Classification Report:\n")
print(classification_report(y_test, y_pred))

# Confusion matrix
cm = confusion_matrix(y_test, y_pred)
print("\n🧮 Confusion Matrix:\n", cm)

# Plot confusion matrix
plt.figure(figsize=(5,4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()
plt.show()

# Importing the necessary libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Importing the customer churn dataset
df= pd.read_csv("D:\data analytics Projects\Customer churn reducing\customer_churn_demo_v2.csv")

# Preview the dataset
df.head()

df.info()

# Churn rate by plan type
print(df.groupby("plan_type")["churned"].value_counts(normalize=True).mul(100).round(1))

# Average hours per week for Churned vs Non-Churned
df.groupby("churned")["hours_per_week"].mean()

# Satisfaction vs Churn
df.groupby("churned")["satisfaction_score"].mean()

# Top Cancellation Reasons
df.loc[df["churned"]=="Yes","cancellation_reason"].value_counts().plot(kind="bar")


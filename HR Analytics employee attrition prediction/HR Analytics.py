# 📦 Import Libraries
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier

# 📥 1. Data Collection
df = pd.read_excel(r'D:\data analytics Projects\HR attrition\HR-Employee-Attrition.xlsx')

# Ensure confidentiality: drop any personally identifiable columns if present
df.drop(columns=['EmployeeNumber', 'EmployeeCount', 'Over18', 'StandardHours'], errors='ignore', inplace=True)

# 🔍 2. Data Exploration
df['Attrition'] = df['Attrition'].map({'Yes': 1, 'No': 0})  # Convert to binary

# Attrition rate by Job Role
jobrole_attrition = df.groupby('JobRole')['Attrition'].mean().sort_values(ascending=False)
print("Attrition Rate by Job Role:\n", jobrole_attrition)

# Summary stats
print(df[['MonthlyIncome', 'TotalWorkingYears', 'JobRole']].describe())

# 🧹 3. Data Preprocessing
df['MonthlyIncome'] = df['MonthlyIncome'].fillna(df['MonthlyIncome'].median())
df['TotalWorkingYears'] = df['TotalWorkingYears'].fillna(df['TotalWorkingYears'].median())

# Encode categorical job roles
le = LabelEncoder()
df['JobRole_encoded'] = le.fit_transform(df['JobRole'])

# 📊 4. Exploratory Data Analysis (EDA)

# Heatmap of numeric correlations
numeric_df = df.select_dtypes(include='number')
plt.figure(figsize=(12, 8))
sns.heatmap(numeric_df.corr(), annot=True, cmap='coolwarm', fmt=".2f")
plt.title("Correlation Heatmap")
plt.tight_layout()
plt.show()

# Boxplots for key factors vs Attrition
eda_features = ['WorkLifeBalance', 'JobSatisfaction', 'YearsSinceLastPromotion']
for feature in eda_features:
    plt.figure(figsize=(6, 4))
    sns.boxplot(
        x='Attrition',
        y=feature,
        data=df,
        hue='Attrition',         # Explicitly assign hue
        palette='magma',
        legend=False             # Suppress legend
    )
    plt.title(f"{feature} vs Attrition")
    plt.tight_layout()
    plt.show()
# 📌 5. Feature Selection
X = numeric_df.drop(columns=['Attrition'])
y = df['Attrition']

model = RandomForestClassifier(random_state=42)
model.fit(X, y)

# Feature importance
importances = pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=False)
print("Top Features Influencing Attrition:\n", importances.head(10))

# Plot top features
plt.figure(figsize=(10, 6))
sns.barplot(
    x=importances.values[:10],
    y=importances.index[:10],
    hue=importances.index[:10],
    palette='magma',
    legend=False
)
plt.title("Top 10 Features Influencing Attrition")
plt.xlabel("Importance Score")
plt.tight_layout()
plt.show()

# 📝 6. Conclusion & Recommendations
print("\n📌 Recommendations to Improve Retention:")
print("- Enhance work-life balance through flexible scheduling")
print("- Create clear promotion pathways and career development plans")
print("- Monitor job satisfaction regularly and act on feedback")
print("- Offer competitive compensation, especially in high-attrition roles")
print("- Provide mentorship and recognition programs for long-tenured employees")

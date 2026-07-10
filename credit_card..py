# ============================================
# CREDIT CARD APPROVAL PREDICTION - PART 1
# ============================================

# Install Required Library
!pip install -q xgboost

# ==========================
# Import Libraries
# ==========================

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings("ignore")

from google.colab import files

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

from xgboost import XGBClassifier

import joblib

# ==========================
# Upload Dataset
# ==========================

print("Upload credit_card.csv")

uploaded = files.upload()

filename = list(uploaded.keys())[0]

df = pd.read_csv(filename)

print("\nDataset Loaded Successfully")

print(df.head())

# ==========================
# Dataset Information
# ==========================

print("\nDataset Shape :", df.shape)

print("\nDataset Info")

print(df.info())

print("\nMissing Values")

print(df.isnull().sum())

# ==========================
# Handle Missing Values
# ==========================

for col in df.columns:

    if df[col].dtype == "object":

        df[col].fillna(df[col].mode()[0], inplace=True)

    else:

        df[col].fillna(df[col].median(), inplace=True)

print("\nMissing Values Removed Successfully")

# ==========================
# Label Encoding
# ==========================

encoders = {}

for col in df.columns:

    if df[col].dtype == "object":

        le = LabelEncoder()

        df[col] = le.fit_transform(df[col])

        encoders[col] = le

print("\nCategorical Columns Encoded")

# ==========================
# Features and Target
# ==========================

X = df.drop("Approved", axis=1)

y = df["Approved"]

print("\nFeatures Shape :", X.shape)

print("Target Shape :", y.shape)

# ==========================
# Train Test Split
# ==========================

X_train, X_test, y_train, y_test = train_test_split(

    X,

    y,

    test_size=0.20,

    random_state=42

)

print("\nTraining Samples :", len(X_train))

print("Testing Samples :", len(X_test))

# ==========================
# Feature Scaling
# ==========================

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)

X_test = scaler.transform(X_test)

print("\nData Preprocessing Completed Successfully")
# ============================================
# CREDIT CARD APPROVAL PREDICTION - PART 2
# Model Training
# ============================================

models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=200, random_state=42),
    "XGBoost": XGBClassifier(eval_metric='logloss', random_state=42)
}

accuracy = {}

best_model = None
best_accuracy = 0
best_model_name = ""

print("="*50)
print("TRAINING MODELS")
print("="*50)

for name, model in models.items():

    print("\nTraining :", name)

    model.fit(X_train, y_train)

    pred = model.predict(X_test)

    acc = accuracy_score(y_test, pred)

    accuracy[name] = acc

    print("Accuracy :", round(acc*100,2), "%")

    print("\nConfusion Matrix")
    print(confusion_matrix(y_test, pred))

    print("\nClassification Report")
    print(classification_report(y_test, pred))

    if acc > best_accuracy:
        best_accuracy = acc
        best_model = model
        best_model_name = name

print("\n" + "="*50)
print("MODEL COMPARISON")
print("="*50)

results = pd.DataFrame({
    "Model": list(accuracy.keys()),
    "Accuracy": list(accuracy.values())
})

results = results.sort_values(by="Accuracy", ascending=False)

print(results)

plt.figure(figsize=(8,5))
sns.barplot(x="Model", y="Accuracy", data=results)
plt.title("Model Accuracy Comparison")
plt.xticks(rotation=15)
plt.show()

print("\nBest Model :", best_model_name)
print("Best Accuracy :", round(best_accuracy*100,2), "%")
# ============================================
# CREDIT CARD APPROVAL PREDICTION - PART 3
# Save Best Model & Prediction
# ============================================

import joblib

# Save Best Model
joblib.dump(best_model, "best_model.pkl")

# Save Label Encoders
joblib.dump(encoders, "label_encoders.pkl")

# Save Scaler
joblib.dump(scaler, "scaler.pkl")

print("="*50)
print("FILES SAVED SUCCESSFULLY")
print("="*50)

print("best_model.pkl")
print("label_encoders.pkl")
print("scaler.pkl")


# ============================================
# SAMPLE PREDICTION
# ============================================

sample = X.iloc[[0]]

sample_scaled = scaler.transform(sample)

prediction = best_model.predict(sample_scaled)

print("\n==============================")

if prediction[0] == 1:
    print("Prediction : CREDIT CARD APPROVED")
else:
    print("Prediction : CREDIT CARD REJECTED")

print("==============================")


# ============================================
# FEATURE IMPORTANCE (Random Forest/XGBoost)
# ============================================

try:

    importance = best_model.feature_importances_

    feature_df = pd.DataFrame({

        "Feature": X.columns,

        "Importance": importance

    })

    feature_df = feature_df.sort_values(

        by="Importance",

        ascending=False

    )

    print("\nTop Important Features")

    print(feature_df)

    plt.figure(figsize=(10,6))

    sns.barplot(

        x="Importance",

        y="Feature",

        data=feature_df

    )

    plt.title("Feature Importance")

    plt.show()

except:

    print("\nFeature Importance Not Available For This Model")


# ============================================
# DOWNLOAD MODEL FILES
# ============================================

from google.colab import files

files.download("best_model.pkl")

files.download("label_encoders.pkl")

files.download("scaler.pkl")

print("\nPROJECT COMPLETED SUCCESSFULLY")

print("Best Model :", best_model_name)

print("Accuracy :", round(best_accuracy*100,2), "%")
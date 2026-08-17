import pandas as pd
from python.backend.database import get_connection

connection = get_connection()

query = "SELECT * FROM dbo.CustomerChurn"

df = pd.read_sql(query, connection)

print("Data loaded successfully!")
connection.close()

print(df.head())
print("*"*50)

print("Shape:", df.shape)
print("*"*50)

print(df.columns)
print("*"*50)

print(df.dtypes)
print("*"*50)

print("Missing Values:")
print(df.isnull().sum())
print("*"*50)

print("\nDuplicates:", df.duplicated().sum())
print("*"*50)

print("\nStatistics:")
print(df.describe())
print("*"*50)

# Checking the categorical values for conversion
print("\nChurn distribution:")
print(df["Churn"].value_counts())
print("*"*50)


print("\nGender distribution:")
print(df['Gender'].value_counts())
print("*"*50)

print("\nContract distribution:")
print(df['Contract'].value_counts())
print("*"*50)

print("\nPayment method distribution:")
print(df['PaymentMethod'].value_counts())
print("*"*50)

#Seperating the features and target
#Deleting the customer_id
X = df.drop(columns=["CustomerID", "Churn"])
y = df["Churn"]

print("X columns:")
print(X.columns)
print("*"*50)
#Coverting the yeas and no with 1 and 0
y = y.map({
    "No": 0,
    "Yes": 1
})
print(y.head())
print("*"*50)

numerical_columns = X.select_dtypes(include=["int64", "float64"]).columns
categorical_columns = X.select_dtypes(include=["str"]).columns

print("\nNumerical columns:")
print(numerical_columns)

print("\nCategorical columns:")
print(categorical_columns)
print("*"*50)

#one hot encoding of gender sice they don't have order
from sklearn.preprocessing import OneHotEncoder

encoder = OneHotEncoder(sparse_output=False, handle_unknown="ignore")

gender_encoded = encoder.fit_transform(X[["Gender"]])

print("\nEncoded Gender:")
print(gender_encoded[:5])

print("\nGender columns:")
print(encoder.get_feature_names_out(["Gender"]))

#ordinal encoding of contract sice they have order

from sklearn.preprocessing import OrdinalEncoder

contract_encoder = OrdinalEncoder(
    categories=[["Month-to-month", "One year", "Two year"]]
)

contract_encoded = contract_encoder.fit_transform(X[["Contract"]])

print("\nContract encoded:")
print(contract_encoded[:5])
print("*"*50)

#one hot encoding for payment 

payment_encoder = OneHotEncoder(
    sparse_output=False,
    handle_unknown="ignore"
)

payment_encoded = payment_encoder.fit_transform(
    X[["PaymentMethod"]]
)

print("\nPaymentMethod encoded:")
print(payment_encoded[:5])

print("\nPaymentMethod columns:")
print(payment_encoder.get_feature_names_out(["PaymentMethod"]))

#handing the numerical data for standardscaler

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

numerical_scaled = scaler.fit_transform(
    X[["Age", "Tenure", "MonthlyCharges", "TotalCharges"]]
)

print("\nScaled numerical values:")
print(numerical_scaled[:5])

#combining all the values which were standard scalered and encoded

import numpy as np

# Get transformed column names
gender_columns = encoder.get_feature_names_out(["Gender"])
payment_columns = payment_encoder.get_feature_names_out(["PaymentMethod"])

all_columns = (
    list(numerical_columns)
    + list(gender_columns)
    + ["Contract"]
    + list(payment_columns)
)

# Combine all transformed data
X_processed = np.hstack([
    numerical_scaled,
    gender_encoded,
    contract_encoded,
    payment_encoded
])

# Convert to DataFrame
X_processed_df = pd.DataFrame(
    X_processed,
    columns=all_columns,
    index=X.index
)

# Print first 10 rows
print("\nFirst 10 rows after transformation:")
print(X_processed_df.head(10))
print(X_processed_df.shape)

print("*"*50)

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X_processed_df,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print( X_train.shape)
print( X_test.shape)
#Churn distribution 
print(y_train.value_counts())
print(y_test.value_counts())

print("*"*50)

print("\nTraining class distribution:")
print(y_train.value_counts())

print("\nTraining class percentages:")
print(y_train.value_counts(normalize=True) * 100)


print("*"*50)

from sklearn.linear_model import LogisticRegression

model = LogisticRegression(
    class_weight="balanced",
    random_state=42,
    max_iter=1000
)

model.fit(X_train, y_train)

print("*"*50)

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

# Make predictions
y_pred = model.predict(X_test)

# Evaluation
print("\n--- Model Evaluation ---")

print("Accuracy :", accuracy_score(y_test, y_pred))
print("Precision:", precision_score(y_test, y_pred))
print("Recall   :", recall_score(y_test, y_pred))
print("F1 Score :", f1_score(y_test, y_pred))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_pred))


print("*"*50)

from sklearn.tree import DecisionTreeClassifier

dt_model = DecisionTreeClassifier(
    class_weight="balanced",
    random_state=42,
    max_depth=5
)

dt_model.fit(X_train, y_train)

y_pred_dt = dt_model.predict(X_test)

print("\n--- Decision Tree Evaluation ---")

print("Accuracy :", accuracy_score(y_test, y_pred_dt))
print("Precision:", precision_score(y_test, y_pred_dt))
print("Recall   :", recall_score(y_test, y_pred_dt))
print("F1 Score :", f1_score(y_test, y_pred_dt))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred_dt))

print("\nClassification Report:")
print(classification_report(y_test, y_pred_dt))

print("*"*50)

from sklearn.model_selection import GridSearchCV

param_grid = {
    "max_depth": [3, 5, 7, 10, 15],
    "min_samples_split": [2, 5, 10],
    "min_samples_leaf": [1, 2, 5]
}

grid_search = GridSearchCV(
    DecisionTreeClassifier(
        class_weight="balanced",
        random_state=42
    ),
    param_grid,
    cv=5,
    scoring="f1",
    n_jobs=-1
)

grid_search.fit(X_train, y_train)

print("\nBest parameters:")
print(grid_search.best_params_)

print("\nBest cross-validation F1 score:")
print(grid_search.best_score_)

print("*"*50)

best_dt = grid_search.best_estimator_

y_pred_best_dt = best_dt.predict(X_test)

print("\n--- Tuned Decision Tree Evaluation ---")

print("Accuracy :", accuracy_score(y_test, y_pred_best_dt))
print("Precision:", precision_score(y_test, y_pred_best_dt))
print("Recall   :", recall_score(y_test, y_pred_best_dt))
print("F1 Score :", f1_score(y_test, y_pred_best_dt))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred_best_dt))

print("\nClassification Report:")
print(classification_report(y_test, y_pred_best_dt))


from sklearn.metrics import roc_auc_score

y_prob = best_dt.predict_proba(X_test)[:, 1]

roc_auc = roc_auc_score(y_test, y_prob)

print("\nROC-AUC:", roc_auc)



y_prob = best_dt.predict_proba(X_test)[:, 1]

print(y_prob[:10])


#using random forest

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    roc_auc_score
)

# Create Random Forest model
rf_model = RandomForestClassifier(
    n_estimators=100,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1
)

# Train
rf_model.fit(X_train, y_train)

# Predictions
y_pred_rf = rf_model.predict(X_test)

# Probabilities for ROC-AUC
y_prob_rf = rf_model.predict_proba(X_test)[:, 1]

# Evaluation
print("\n--- Random Forest Evaluation ---")

print("Accuracy :", accuracy_score(y_test, y_pred_rf))
print("Precision:", precision_score(y_test, y_pred_rf))
print("Recall   :", recall_score(y_test, y_pred_rf))
print("F1 Score :", f1_score(y_test, y_pred_rf))
print("ROC-AUC  :", roc_auc_score(y_test, y_prob_rf))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred_rf))

print("\nClassification Report:")
print(classification_report(y_test, y_pred_rf))

print("*"*50)

from xgboost import XGBClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    roc_auc_score
)

scale_pos_weight=53485/26515
xgb_model=XGBClassifier(
    scale_pos_weight=1.75
)

xgb_model.fit(X_train, y_train)

# Predictions
y_pred_xgb = xgb_model.predict(X_test)

# Probability of churn
y_prob_xgb = xgb_model.predict_proba(X_test)[:, 1]

# Evaluation
print("\n--- XGBoost Evaluation ---")

print("Accuracy :", accuracy_score(y_test, y_pred_xgb))
print("Precision:", precision_score(y_test, y_pred_xgb))
print("Recall   :", recall_score(y_test, y_pred_xgb))
print("F1 Score :", f1_score(y_test, y_pred_xgb))
print("ROC-AUC  :", roc_auc_score(y_test, y_prob_xgb))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred_xgb))

print("\nClassification Report:")
print(classification_report(y_test, y_pred_xgb))


final_model = best_dt

print("\nFinal model selected:")
print(final_model)

import joblib

joblib.dump(best_dt, "churn_model.pkl")

print("Model saved successfully!")

#These are for performing the data preprocessing on new data
joblib.dump(scaler, "scaler.pkl")
joblib.dump(encoder, "gender_encoder.pkl")
joblib.dump(contract_encoder, "contract_encoder.pkl")
joblib.dump(payment_encoder, "payment_encoder.pkl")

print("Preprocessing objects saved successfully!")




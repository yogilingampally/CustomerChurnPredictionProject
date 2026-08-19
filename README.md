# Customer Churn Prediction System

An end-to-end **Machine Learning project** that predicts whether a customer is likely to churn based on demographic, account, service, and billing information.

The project uses **SQL Server for data storage**, **Python for data processing and machine learning**, and **Scikit-learn/XGBoost for model development**. The trained model and preprocessing objects are saved using Joblib so they can be used later in the deployed application.

---

## 🚀 Project Overview

Customer churn prediction helps businesses identify customers who are likely to stop using their services.

In this project, customer data is stored in a **SQL Server database**. The data is extracted using Python, cleaned and analyzed, transformed into machine-learning-ready features, and used to train multiple classification models.

Several models are compared:

* Logistic Regression
* Decision Tree
* Tuned Decision Tree
* Random Forest
* XGBoost

After comparing the models using appropriate classification metrics, the best-performing model is selected and saved for deployment.

---

## 🎯 Project Objectives

The main objectives of this project are:

* Extract customer data directly from SQL Server.
* Perform exploratory data analysis and data-quality checks.
* Identify numerical and categorical features.
* Encode categorical variables appropriately.
* Standardize numerical features.
* Handle class imbalance.
* Train multiple machine learning classification models.
* Tune the Decision Tree using GridSearchCV.
* Evaluate models using multiple performance metrics.
* Select the best-performing model.
* Save the trained model and preprocessing objects.
* Use the saved model for future customer churn predictions.

---

# 🏗️ Project Workflow

The complete machine learning workflow is:

```text
SQL Server Database
        ↓
Data Extraction
        ↓
Data Quality Checks
        ↓
Exploratory Data Analysis
        ↓
Feature & Target Separation
        ↓
Categorical Encoding
        ↓
Numerical Feature Scaling
        ↓
Feature Combination
        ↓
Train-Test Split
        ↓
Model Training
        ↓
Model Comparison
        ↓
Hyperparameter Tuning
        ↓
Model Evaluation
        ↓
Best Model Selection
        ↓
Save Model & Preprocessing Objects
        ↓
Deployment
```

---

# 📊 1. Data Extraction from SQL Server

The customer churn data is stored in a SQL Server database.

A custom database connection function is used to establish the connection:

```python
from python.backend.database import get_connection

connection = get_connection()
```

The data is retrieved using SQL:

```python
query = "SELECT * FROM dbo.CustomerChurn"

df = pd.read_sql(query, connection)
```

After loading the data, the database connection is closed:

```python
connection.close()
```

This allows the machine learning pipeline to work directly with the database rather than manually importing a CSV file.

---

# 🔍 2. Data Inspection

After loading the dataset, several checks are performed to understand the structure and quality of the data.

### Dataset preview

```python
print(df.head())
```

### Dataset shape

```python
print(df.shape)
```

This tells us the number of rows and columns.

### Column names

```python
print(df.columns)
```

### Data types

```python
print(df.dtypes)
```

Understanding data types is important because numerical and categorical columns require different preprocessing techniques.

---

# 🧹 3. Data Quality Checks

Before building the model, the dataset is checked for missing values and duplicate records.

### Missing values

```python
df.isnull().sum()
```

This helps identify columns containing missing data.

### Duplicate records

```python
df.duplicated().sum()
```

Duplicate records can potentially affect model training and should be investigated before modeling.

---

# 📈 4. Statistical Analysis

Basic statistical information is obtained using:

```python
df.describe()
```

This provides information such as:

* Mean
* Standard deviation
* Minimum
* Maximum
* Quartiles

This is useful for understanding numerical features and detecting unusual values.

---

# 🎯 5. Target Variable Analysis

The target variable in this project is:

```text
Churn
```

The distribution of the target is checked:

```python
df["Churn"].value_counts()
```

The target contains two classes:

```text
No  → Customer does not churn
Yes → Customer churns
```

The target is converted into numerical values:

```python
y = y.map({
    "No": 0,
    "Yes": 1
})
```

Therefore:

```text
0 → No Churn
1 → Churn
```

---

# 👥 6. Categorical Feature Analysis

The distributions of categorical features are also examined.

### Gender

```python
df["Gender"].value_counts()
```

### Contract

```python
df["Contract"].value_counts()
```

### Payment Method

```python
df["PaymentMethod"].value_counts()
```

This helps understand the categories present in the dataset before applying encoding.

---

# 🧩 7. Feature and Target Separation

The `CustomerID` column is removed because it is an identifier and does not provide meaningful predictive information.

```python
X = df.drop(columns=["CustomerID", "Churn"])
y = df["Churn"]
```

Therefore:

```text
X → Input features
y → Target variable
```

---

# 🔢 8. Identifying Numerical and Categorical Features

The numerical columns are identified using:

```python
numerical_columns = X.select_dtypes(
    include=["int64", "float64"]
).columns
```

Categorical columns are identified separately.

The project contains features such as:

### Numerical Features

* Age
* Tenure
* MonthlyCharges
* TotalCharges

### Categorical Features

* Gender
* Contract
* PaymentMethod

---

# 🔠 9. Encoding Categorical Features

Machine learning algorithms generally require numerical input.

Therefore, categorical features are converted into numerical representations.

Different encoding techniques are used depending on the nature of the feature.

---

## Gender — One-Hot Encoding

Gender does not have a meaningful order, so **One-Hot Encoding** is used.

```python
from sklearn.preprocessing import OneHotEncoder

encoder = OneHotEncoder(
    sparse_output=False,
    handle_unknown="ignore"
)

gender_encoded = encoder.fit_transform(
    X[["Gender"]]
)
```

This creates separate binary columns for the different gender categories.

`handle_unknown="ignore"` is particularly useful during deployment because a new category will not cause the preprocessing pipeline to fail.

---

# 📋 10. Contract — Ordinal Encoding

Contract type has an inherent progression based on contract duration:

```text
Month-to-month
One year
Two year
```

Therefore, an ordinal encoding approach is used.

```python
from sklearn.preprocessing import OrdinalEncoder

contract_encoder = OrdinalEncoder(
    categories=[
        ["Month-to-month", "One year", "Two year"]
    ]
)
```

The resulting values represent the predefined ordering.

---

# 💳 11. Payment Method — One-Hot Encoding

Payment method is a nominal categorical feature, meaning its categories do not have an inherent ranking.

Therefore, One-Hot Encoding is used:

```python
payment_encoder = OneHotEncoder(
    sparse_output=False,
    handle_unknown="ignore"
)

payment_encoded = payment_encoder.fit_transform(
    X[["PaymentMethod"]]
)
```

---

# 📏 12. Numerical Feature Scaling

The numerical variables can have very different ranges.

For example:

```text
Age             → relatively small values
Tenure          → smaller values
MonthlyCharges  → larger values
TotalCharges    → potentially much larger values
```

To bring the numerical features to a comparable scale, `StandardScaler` is used.

```python
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

numerical_scaled = scaler.fit_transform(
    X[
        [
            "Age",
            "Tenure",
            "MonthlyCharges",
            "TotalCharges"
        ]
    ]
)
```

Standardization transforms the numerical features approximately to:

```text
Mean = 0
Standard Deviation = 1
```

---

# 🧱 13. Combining the Processed Features

After encoding and scaling, the transformed features are combined into one dataset.

```python
X_processed = np.hstack([
    numerical_scaled,
    gender_encoded,
    contract_encoded,
    payment_encoded
])
```

The resulting NumPy array is converted back into a Pandas DataFrame:

```python
X_processed_df = pd.DataFrame(
    X_processed,
    columns=all_columns,
    index=X.index
)
```

This produces the final machine-learning-ready feature matrix.

---

# ✂️ 14. Train-Test Split

The processed data is divided into training and testing datasets.

```python
X_train, X_test, y_train, y_test = train_test_split(
    X_processed_df,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)
```

### Split ratio

```text
80% → Training data
20% → Testing data
```

### Why `stratify=y`?

The churn dataset contains an imbalance between churned and non-churned customers.

`stratify=y` ensures that the training and testing datasets maintain approximately the same class distribution.

---

# ⚖️ 15. Handling Class Imbalance

Customer churn datasets commonly contain more non-churned customers than churned customers.

To reduce the impact of class imbalance, class weighting is used in several models:

```python
class_weight="balanced"
```

This gives greater importance to the minority class during training.

This is especially useful when the business objective is to identify customers who are likely to churn.

---

# 🤖 16. Model 1 — Logistic Regression

The first baseline model is Logistic Regression.

```python
model = LogisticRegression(
    class_weight="balanced",
    random_state=42,
    max_iter=1000
)
```

The model is trained using:

```python
model.fit(X_train, y_train)
```

Predictions are generated using:

```python
y_pred = model.predict(X_test)
```

The model is evaluated using:

* Accuracy
* Precision
* Recall
* F1 Score
* Confusion Matrix

---

# 🌳 17. Model 2 — Decision Tree

A Decision Tree classifier is then trained.

```python
dt_model = DecisionTreeClassifier(
    class_weight="balanced",
    random_state=42,
    max_depth=5
)
```

The `max_depth` parameter limits the depth of the tree and helps control overfitting.

---

# 🔧 18. Hyperparameter Tuning with GridSearchCV

Instead of manually selecting Decision Tree parameters, GridSearchCV is used to find a better combination.

The parameter grid contains:

```python
param_grid = {
    "max_depth": [3, 5, 7, 10, 15],
    "min_samples_split": [2, 5, 10],
    "min_samples_leaf": [1, 2, 5]
}
```

GridSearchCV evaluates different combinations using 5-fold cross-validation:

```python
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
```

### Why F1 Score?

Since churn is the important minority class, F1 Score provides a balance between:

```text
Precision + Recall
```

This makes it more useful than relying only on accuracy.

The best model is obtained using:

```python
best_dt = grid_search.best_estimator_
```

---

# 🌲 19. Model 3 — Random Forest

Random Forest is another tree-based ensemble model used for comparison.

```python
rf_model = RandomForestClassifier(
    n_estimators=100,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1
)
```

Random Forest combines multiple decision trees to improve predictive performance and reduce the risk of relying on a single tree.

---

# 🚀 20. Model 4 — XGBoost

XGBoost is also evaluated as a powerful gradient-boosting algorithm.

```python
xgb_model = XGBClassifier(
    scale_pos_weight=1.75
)
```

The `scale_pos_weight` parameter is used to give additional importance to the positive churn class.

The model produces:

* Predictions
* Churn probabilities
* Accuracy
* Precision
* Recall
* F1 Score
* ROC-AUC
* Confusion Matrix

---

# 📊 21. Model Evaluation

The models are evaluated using multiple metrics rather than relying only on accuracy.

### Accuracy

Measures the overall percentage of correct predictions.

```text
Accuracy =
Correct Predictions / Total Predictions
```

### Precision

Among customers predicted as churners, precision measures how many actually churned.

This is useful when the business wants to avoid unnecessarily targeting customers who are unlikely to churn.

### Recall

Recall measures how many actual churners were correctly identified.

For a churn prediction problem, recall can be particularly important because missing a customer who is actually going to churn can result in lost revenue.

### F1 Score

F1 Score balances precision and recall.

```text
F1 = 2 × Precision × Recall
          -------------------
          Precision + Recall
```

### ROC-AUC

ROC-AUC measures the model's ability to distinguish between churn and non-churn customers across different classification thresholds.

---

# 📉 22. Confusion Matrix

A confusion matrix is generated for the models:

```python
confusion_matrix(y_test, y_pred)
```

It contains:

```text
                    Predicted
                  No       Yes

Actual No        TN        FP

Actual Yes       FN        TP
```

For churn prediction:

* **True Negative** → Correctly predicted non-churn customer
* **False Positive** → Predicted churn but customer does not churn
* **False Negative** → Customer churns but model fails to identify them
* **True Positive** → Correctly predicted churn customer

---

# 🏆 23. Model Selection

After evaluating multiple algorithms, the tuned Decision Tree is selected as the final model:

```python
final_model = best_dt
```

The selection is based on the model's performance on the relevant evaluation metrics, particularly F1 Score and ROC-AUC.

The final model is:

```text
Tuned Decision Tree Classifier
```

---

# 💾 24. Saving the Machine Learning Model

The final trained model is saved using Joblib.

```python
import joblib

joblib.dump(
    best_dt,
    "churn_model.pkl"
)
```

This allows the trained model to be loaded later without retraining it.

---

# 📦 25. Saving Preprocessing Objects

The preprocessing objects are also saved:

```python
joblib.dump(
    scaler,
    "scaler.pkl"
)

joblib.dump(
    encoder,
    "gender_encoder.pkl"
)

joblib.dump(
    contract_encoder,
    "contract_encoder.pkl"
)

joblib.dump(
    payment_encoder,
    "payment_encoder.pkl"
)
```

This is important because new customer data must go through the **same preprocessing steps** used during model training.

The saved objects are:

| File                   | Purpose                         |
| ---------------------- | ------------------------------- |
| `churn_model.pkl`      | Trained Decision Tree model     |
| `scaler.pkl`           | Standardizes numerical features |
| `gender_encoder.pkl`   | Encodes Gender                  |
| `contract_encoder.pkl` | Encodes Contract                |
| `payment_encoder.pkl`  | Encodes Payment Method          |

---

# 🔄 Prediction Pipeline for New Customers

When a new customer is entered into the application, the data follows the same preprocessing pipeline:

```text
New Customer Data
       ↓
Separate Numerical & Categorical Features
       ↓
Gender → One-Hot Encoding
       ↓
Contract → Ordinal Encoding
       ↓
Payment Method → One-Hot Encoding
       ↓
Numerical Features → StandardScaler
       ↓
Combine Features
       ↓
Load churn_model.pkl
       ↓
Generate Prediction
       ↓
Churn / No Churn
       ↓
Churn Probability
```

This ensures that the model receives data in the same format that it saw during training.

---

# 🖥️ Deployment

The trained model and preprocessing objects can be integrated into a Streamlit application.

The application can provide an interactive interface where users enter customer information and receive:

```text
Prediction:
Customer will churn

Churn Probability:
82.66%
```

The application can be used as a business-oriented customer churn prediction dashboard.

---

# 🗂️ Suggested Project Structure

```text
Customer-Churn-Prediction/
│
├── python/
│   ├── backend/
│   │   └── database.py
│   │
│   └── model/
│       └── model_training.py
│
├── app.py
│
├── churn_model.pkl
├── scaler.pkl
├── gender_encoder.pkl
├── contract_encoder.pkl
├── payment_encoder.pkl
│
├── requirements.txt
├── .gitignore
└── README.md
```

---

# 🛠️ Technologies Used

### Programming Language

* Python

### Data Analysis

* Pandas
* NumPy

### Machine Learning

* Scikit-learn
* XGBoost

### Database

* Microsoft SQL Server
* SQL

### Model Persistence

* Joblib

### Deployment / Interface

* Streamlit

### Development Tools

* VS Code
* Jupyter Notebook

---

# 📚 Machine Learning Concepts Demonstrated

This project demonstrates practical knowledge of:

* SQL data extraction
* Exploratory Data Analysis
* Data quality checking
* Missing-value analysis
* Duplicate detection
* Feature engineering
* One-Hot Encoding
* Ordinal Encoding
* Standardization
* Train-Test Split
* Stratified Sampling
* Class Imbalance
* Logistic Regression
* Decision Trees
* Random Forest
* XGBoost
* Cross-Validation
* GridSearchCV
* Hyperparameter Tuning
* Confusion Matrix
* Precision
* Recall
* F1 Score
* ROC-AUC
* Model Selection
* Model Serialization
* ML Deployment

---

# ⚠️ Important Machine Learning Consideration

For a production-grade ML pipeline, preprocessing transformations such as scaling and encoding should ideally be **fitted only on the training data**, rather than fitting them on the complete dataset before the train-test split.

A production implementation can use Scikit-learn's `Pipeline` and `ColumnTransformer` to keep preprocessing and model training together and prevent data leakage.

Example architecture:

```text
Raw Data
   ↓
ColumnTransformer
   ├── Numerical → StandardScaler
   └── Categorical → Encoder
   ↓
Machine Learning Model
   ↓
Prediction
```

This approach also makes deployment cleaner because the entire preprocessing + model pipeline can be saved as a single object.

---

# 🔮 Future Improvements

Possible improvements for this project include:

* Build a complete `Pipeline` using `ColumnTransformer`.
* Perform preprocessing after the train-test split to avoid preprocessing leakage.
* Perform extensive EDA with visualizations.
* Add ROC curves and Precision-Recall curves.
* Add feature importance visualization.
* Perform cross-validation for all candidate models.
* Tune Random Forest and XGBoost hyperparameters.
* Optimize the classification threshold based on business requirements.
* Add explainable AI using SHAP.
* Connect the Streamlit application directly to SQL Server.
* Add real-time customer prediction.
* Add an interactive churn analytics dashboard.
* Deploy the application using a cloud platform.

---

# 💡 Business Value

The system can help businesses identify customers who are at a higher risk of leaving.

For example:

```text
Customer
   ↓
Churn Prediction
   ↓
High-Risk Customer
   ↓
Retention Strategy
   ↓
Discount / Offer / Support / Personalized Service
```

Instead of waiting for customers to leave, businesses can proactively identify high-risk customers and take retention actions.

---

# 👨‍💻 Author

**Yogi Lingampally**

B.Tech — Computer Science Engineering
Specialization: Artificial Intelligence & Machine Learning

---

# ⭐ Project Highlights

> **End-to-end Customer Churn Prediction System using SQL Server, Python, Machine Learning and Streamlit.**

The project demonstrates the complete journey from **database → preprocessing → model development → evaluation → model persistence → deployment**.


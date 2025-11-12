# Titanic
I trained a Logistic Regression model for discrete classification using the Titanic dataset.

The content of the file **Titanic Disaster.ipynb** is a Jupyter Notebook that outlines a full machine learning pipeline for Titanic survival prediction.
The notebook proceeds through the following steps: importing libraries, loading and exploring the data, performing data cleaning (handling missing values), feature engineering (encoding categorical features), splitting the data, defining an evaluation function, and finally training and evaluating multiple classification models.

-----

## Titanic Disaster.ipynb Content

### 1. Import Libraries

```python
# ===========================================
# Titanic Survival Prediction - Full Pipeline
# ===========================================

# --- 1. IMPORT LIBRARIES ---
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn import metrics
```

### 2. Load Data

```python
# --- 2. LOAD DATA ---
data_df = pd.read_csv('data/Titanic.csv')
print("\nInitial Data Head:")
data_df.head(10)
```

**Output: Initial Data Head (First 10 Rows)**

| | PassengerId | Survived | Pclass | Name | Sex | Age | SibSp | Parch | Ticket | Fare | Cabin | Embarked |
|---:|---:|---:|---:|:---|:---|---:|---:|---:|:---|---:|:---|:---|
| 0 | 1 | 0 | 3 | Braund, Mr. Owen Harris | male | 22.0 | 1 | 0 | A/5 21171 | 7.2500 | NaN | S |
| 1 | 2 | 1 | 1 | Cumings, Mrs. John Bradley (Florence Briggs Th... | female | 38.0 | 1 | 0 | PC 17599 | 71.2833 | C85 | C |
| 2 | 3 | 1 | 3 | Heikkinen, Miss. Laina | female | 26.0 | 0 | 0 | STON/O2. 3101282 | 7.9250 | NaN | S |
| 3 | 4 | 1 | 1 | Futrelle, Mrs. Jacques Heath (Lily May Peel) | female | 35.0 | 1 | 0 | 113803 | 53.1000 | C123 | S |
| 4 | 5 | 0 | 3 | Allen, Mr. William Henry | male | 35.0 | 0 | 0 | 373450 | 8.0500 | NaN | S |
| 5 | 6 | 0 | 3 | Moran, Mr. James | male | NaN | 0 | 0 | 330877 | 8.4583 | NaN | Q |
| 6 | 7 | 0 | 1 | McCarthy, Mr. Timothy J | male | 54.0 | 0 | 0 | 17463 | 51.8625 | E46 | S |
| 7 | 8 | 0 | 3 | Palsson, Master. Gosta Leonard | male | 2.0 | 3 | 1 | 349909 | 21.0750 | NaN | S |
| 8 | 9 | 1 | 3 | Johnson, Mrs. Oscar W (Elisabeth Vilhelmina Berg) | female | 27.0 | 0 | 2 | 347742 | 11.1333 | NaN | S |
| 9 | 10 | 1 | 2 | Nasser, Mrs. Nicholas (Adele Achem) | female | 14.0 | 1 | 0 | 237736 | 30.0708 | NaN | C |

```python
data_df.tail(10)
```

**Output: Tail of Data (Last 10 Rows)**

| | PassengerId | Survived | Pclass | Name | Sex | Age | SibSp | Parch | Ticket | Fare | Cabin | Embarked |
|---:|---:|---:|---:|:---|:---|---:|---:|---:|:---|---:|:---|:---|
| 881 | 882 | 0 | 3 | Markun, Mr. Johann | male | 33.0 | 0 | 0 | 349257 | 7.8958 | NaN | S |
| 882 | 883 | 0 | 3 | Dahlberg, Miss. Gerda Ulrika | female | 22.0 | 0 | 0 | 7552 | 10.5167 | NaN | S |
| 883 | 884 | 0 | 2 | Banfield, Mr. Frederick James | male | 28.0 | 0 | 0 | C.A./SOTON 34068 | 10.5000 | NaN | S |
| 884 | 885 | 0 | 3 | Sutehall, Mr. Henry Jr | male | 25.0 | 0 | 0 | SOTON/OQ 392076 | 7.0500 | NaN | S |
| 885 | 886 | 0 | 3 | Rice, Mrs. William (Margaret Norton) | female | 39.0 | 0 | 5 | 382652 | 29.1250 | NaN | Q |
| 886 | 887 | 0 | 2 | Montvila, Rev. Juozas | male | 27.0 | 0 | 0 | 211536 | 13.0000 | NaN | S |
| 887 | 888 | 1 | 1 | Graham, Miss. Margaret Edith | female | 19.0 | 0 | 0 | 112053 | 30.0000 | B42 | S |
| 888 | 889 | 0 | 3 | Johnston, Miss. Catherine Helen "Carrie" | female | NaN | 1 | 2 | W./C. 6607 | 23.4500 | NaN | S |
| 889 | 890 | 1 | 1 | Behr, Mr. Karl Howell | male | 26.0 | 0 | 0 | 111369 | 30.0000 | C148 | C |
| 890 | 891 | 0 | 3 | Dooley, Mr. Patrick | male | 32.0 | 0 | 0 | 370376 | 7.7500 | NaN | Q |

### 3. Basic Info

```python
# --- 3. BASIC INFO ---
print("\nDataset Shape:", data_df.shape)
print("\nInfo:")
print(data_df.info())
print("\nDescribe:")
data_df.describe()
```

**Output:**

```
Dataset Shape: (891, 12)

Info:
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 891 entries, 0 to 890
Data columns (total 12 columns):
 #   Column       Non-Null Count  Dtype  
---  ------       --------------  -----  
 0   PassengerId  891 non-null    int64  
 1   Survived     891 non-null    int64  
 2   Pclass       891 non-null    int64  
 3   Name         891 non-null    object 
 4   Sex          891 non-null    object 
 5   Age          714 non-null    float64
 6   SibSp        891 non-null    int64  
 7   Parch        891 non-null    int64  
 8   Ticket       891 non-null    object 
 9   Fare         891 non-null    float64
 10  Cabin        204 non-null    object 
 11  Embarked     889 non-null    object 
dtypes: float64(2), int64(5), object(5)
memory usage: 83.7+ KB
None

Describe:
```

| | PassengerId | Survived | Pclass | Age | SibSp | Parch | Fare |
|---:|---:|---:|---:|---:|---:|---:|---:|
| **count** | 891.000000 | 891.000000 | 891.000000 | 714.000000 | 891.000000 | 891.000000 | 891.000000 |
| **mean** | 446.000000 | 0.383838 | 2.308642 | 29.699118 | 0.523008 | 0.381594 | 32.204208 |
| **std** | 257.353842 | 0.486592 | 0.836071 | 14.526497 | 1.102743 | 0.806057 | 49.693429 |
| **min** | 1.000000 | 0.000000 | 1.000000 | 0.420000 | 0.000000 | 0.000000 | 0.000000 |
| **25%** | 223.500000 | 0.000000 | 2.000000 | 20.125000 | 0.000000 | 0.000000 | 7.910400 |
| **50%** | 446.000000 | 0.000000 | 3.000000 | 28.000000 | 0.000000 | 0.000000 | 14.454200 |
| **75%** | 668.500000 | 1.000000 | 3.000000 | 38.000000 | 1.000000 | 0.000000 | 31.000000 |
| **max** | 891.000000 | 1.000000 | 3.000000 | 80.000000 | 8.000000 | 6.000000 | 512.329200 |

-----

### 4. Basic Visualizations

```python
# --- 4. BASIC VISUALIZATIONS ---
fig, ax = plt.subplots(1, 3, figsize=(22, 10))

data_df["Survived"].value_counts().plot(kind="bar", ax=ax[0], color=['red', 'green'])
ax[0].set_title('Survival Count')
ax[0].set_xlabel('Survived (0=No, 1=Yes)')
ax[0].set_ylabel('Count')

data_df["Sex"].value_counts().plot(kind="bar", ax=ax[1], color=['blue', 'orange'])
ax[1].set_title('Sex Count')

data_df["Embarked"].value_counts().plot(kind="bar", ax=ax[2], color=['purple', 'cyan', 'yellow'])
ax[2].set_title('Embarked Count')

plt.tight_layout()
plt.show()
```

**Output: Figure of 3 Bar Plots**

*(This output is a combined figure displaying three bar plots: 'Survival Count', 'Sex Count', and 'Embarked Count'.)*

```python
data_df.corr(numeric_only=True).style.background_gradient(cmap='coolwarm')
```

**Output: Correlation Matrix**

*(This output is a stylized correlation matrix heatmap for the numerical columns.)*

-----

### 5. Missing Values and Data Preprocessing

```python
# --- 5. DATA CLEANING AND PREPROCESSING ---

# Check for missing values
data_df.isnull().sum()
```

**Output: Missing Value Counts**

```
PassengerId      0
Survived         0
Pclass           0
Name             0
Sex              0
Age            177
SibSp            0
Parch            0
Ticket           0
Fare             0
Cabin          687
Embarked         2
dtype: int64
```

```python
# Fill missing 'Age' with mean
data_df["Age"].fillna(data_df["Age"].mean(), inplace=True)

# Fill missing 'Fare' with mean
data_df["Fare"].fillna(data_df["Fare"].mean(), inplace=True)

# Fill missing 'Embarked' with mode (Note: includes a warning about chained assignment)
data_df["Embarked"].fillna(data_df["Embarked"].mode()[0], inplace=True)
```

```python
# Re-check missing values
data_df.isnull().sum()
```

**Output: Missing Value Counts after Imputation**

```
PassengerId      0
Survived         0
Pclass           0
Name             0
Sex              0
Age              0
SibSp            0
Parch            0
Ticket           0
Fare             0
Cabin          687
Embarked         0
dtype: int64
```

```python
# --- 6. FEATURE ENGINEERING / DROPPING ---
# Drop irrelevant features
data_df.drop(['PassengerId', 'Name', 'Ticket', 'Cabin'], axis=1, inplace=True)
```

### 6. Encode Categorical Features

```python
# --- 7. ENCODING CATEGORICAL FEATURES ---
# Convert 'Sex' to numeric
LEC_Sex = LabelEncoder()
data_df['Sex'] = LEC_Sex.fit_transform(data_df['Sex'])
data_df.head()
```

**Output: Head after Sex Encoding**

| | Survived | Pclass | Sex | Age | SibSp | Parch | Fare | Embarked |
|---:|---:|---:|---:|---:|---:|---:|---:|:---|
| 0 | 0 | 3 | 1 | 22.0 | 1 | 0 | 7.2500 | S |
| 1 | 1 | 1 | 0 | 38.0 | 1 | 0 | 71.2833 | C |
| 2 | 1 | 3 | 0 | 26.0 | 0 | 0 | 7.9250 | S |
| 3 | 1 | 1 | 0 | 35.0 | 1 | 0 | 53.1000 | S |
| 4 | 0 | 3 | 1 | 35.0 | 0 | 0 | 8.0500 | S |

```python
# Convert 'Embarked' to numeric
LEC_Embarked = LabelEncoder()
data_df['Embarked'] = LEC_Embarked.fit_transform(data_df['Embarked'])
data_df.head()
```

**Output: Head after Embarked Encoding**

| | Survived | Pclass | Sex | Age | SibSp | Parch | Fare | Embarked |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 0 | 3 | 1 | 22.0 | 1 | 0 | 7.2500 | 2 |
| 1 | 1 | 1 | 0 | 38.0 | 1 | 0 | 71.2833 | 0 |
| 2 | 1 | 3 | 0 | 26.0 | 0 | 0 | 7.9250 | 2 |
| 3 | 1 | 1 | 0 | 35.0 | 1 | 0 | 53.1000 | 2 |
| 4 | 0 | 3 | 1 | 35.0 | 0 | 0 | 8.0500 | 2 |

-----

### 7. Model Training and Evaluation

```python
# --- 8. DEFINE X AND Y ---
y = data_df['Survived']
X = data_df.drop(['Survived'], axis=1)
X.head()
```

**Output: Features (X) Head**

| | Pclass | Sex | Age | SibSp | Parch | Fare | Embarked |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 3 | 1 | 22.0 | 1 | 0 | 7.2500 | 2 |
| 1 | 1 | 0 | 38.0 | 1 | 0 | 71.2833 | 0 |
| 2 | 3 | 0 | 26.0 | 0 | 0 | 7.9250 | 2 |
| 3 | 1 | 0 | 35.0 | 1 | 0 | 53.1000 | 2 |
| 4 | 3 | 1 | 35.0 | 0 | 0 | 8.0500 | 2 |

```python
print(y)
```

**Output: Target (y)**

```
0      0
1      1
2      1
3      1
4      0
      ..
886    0
887    1
888    0
889    1
890    0
Name: Survived, Length: 891, dtype: int64
```

```python
# --- 11. SPLIT DATA ---
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
```

```python
print("\nTrain shape:", X_train.shape, " Test shape:", X_test.shape)
```

**Output:**

```
Train shape: (623, 7) Test shape: (268, 7)
```

```python
# --- 12. EVALUATION FUNCTION ---
def evaluate_model(model, model_name):
    # ... (function body for fitting and printing evaluation metrics)
    print(f"\n--- {model_name} Evaluation ---")
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    print(f"Accuracy: {metrics.accuracy_score(y_test, y_pred):.4f}")
    print("\nClassification Report:\n", metrics.classification_report(y_test, y_pred))
    return model
```

#### Logistic Regression

```python
# --- 13. LOGISTIC REGRESSION ---
from sklearn.linear_model import LogisticRegression
LogisticRegression = evaluate_model(LogisticRegression(random_state=42), "Logistic Regression")
```

**Output:**

```
--- Logistic Regression Evaluation ---
Accuracy: 0.8172

Classification Report:
               precision    recall  f1-score   support

           0       0.83      0.88      0.85       157
           1       0.79      0.72      0.75       111

    accuracy                           0.82       268
   macro avg       0.81      0.80      0.80       268
weighted avg       0.81      0.82      0.81       268
```

#### K-Nearest Neighbors

```python
# --- 14. K-NEAREST NEIGHBORS ---
from sklearn.neighbors import KNeighborsClassifier
evaluate_model(KNeighborsClassifier(), "K-Nearest Neighbors Classifier")
```

**Output:**

```
--- K-Nearest Neighbors Classifier Evaluation ---
Accuracy: 0.6978

Classification Report:
               precision    recall  f1-score   support

           0       0.74      0.78      0.76       157
           1       0.63      0.58      0.60       111

    accuracy                           0.70       268
   macro avg       0.68      0.68      0.68       268
weighted avg       0.70      0.70      0.70       268
```

#### Support Vector Machine (SVC)

```python
# --- 15. SUPPORT VECTOR MACHINE ---
from sklearn.svm import SVC
evaluate_model(SVC(random_state=42), "Support Vector Machine (SVC)")
```

**Output:**

```
--- Support Vector Machine (SVC) Evaluation ---
Accuracy: 0.6828

Classification Report:
               precision    recall  f1-score   support

           0       0.69      0.93      0.79       157
           1       0.60      0.24      0.35       111

    accuracy                           0.68       268
   macro avg       0.65      0.58      0.57       268
weighted avg       0.66      0.68      0.64       268
```

#### Gaussian Naive Bayes

```python
# --- 16. GAUSSIAN NAIVE BAYES ---
from sklearn.naive_bayes import GaussianNB
evaluate_model(GaussianNB(), "Gaussian Naive Bayes")
```

**Output:**

```
--- Gaussian Naive Bayes Evaluation ---
Accuracy: 0.7724

Classification Report:
               precision    recall  f1-score   support

           0       0.83      0.78      0.81       157
           1       0.69      0.77      0.73       111

    accuracy                           0.77       268
   macro avg       0.76      0.77      0.77       268
weighted avg       0.78      0.77      0.77       268
```

#### Decision Tree

```python
# --- 17. DECISION TREE ---
from sklearn.tree import DecisionTreeClassifier
evaluate_model(DecisionTreeClassifier(random_state=42), "Decision Tree Classifier")
```

**Output:**

```
--- Decision Tree Classifier Evaluation ---
Accuracy: 0.7687

Classification Report:
               precision    recall  f1-score   support

           0       0.80      0.82      0.81       157
           1       0.72      0.70      0.71       111

    accuracy                           0.77       268
   macro avg       0.76      0.76      0.76       268
weighted avg       0.77      0.77      0.77       268
```

#### Random Forest

```python
# --- 16. RANDOM FOREST ---
from sklearn.ensemble import RandomForestClassifier
evaluate_model(RandomForestClassifier(random_state=42), "Random Forest Classifier")
```

**Output:**

```
--- Random Forest Classifier Evaluation ---
Accuracy: 0.7910

Classification Report:
               precision    recall  f1-score   support

           0       0.80      0.85      0.83       157
           1       0.77      0.70      0.74       111

    accuracy                           0.79       268
   macro avg       0.79      0.78      0.78       268
weighted avg       0.79      0.79      0.79       268
```

#### Gradient Boosting

```python
# --- 19. GRADIENT BOOSTING ---
from sklearn.ensemble import GradientBoostingClassifier
evaluate_model(GradientBoostingClassifier(random_state=42), "Gradient Boosting Classifier")
```

**Output:**

```
--- Gradient Boosting Classifier Evaluation ---
Accuracy: 0.8134

Classification Report:
               precision    recall  f1-score   support

           0       0.82      0.88      0.85       157
           1       0.79      0.71      0.75       111

    accuracy                           0.81       268
   macro avg       0.80      0.80      0.80       268
weighted avg       0.81      0.81      0.81       268
```

#### AdaBoost

```python
# --- 20. ADABOOST ---
from sklearn.ensemble import AdaBoostClassifier
evaluate_model(AdaBoostClassifier(random_state=42), "AdaBoost Classifier")
```

**Output:**

```
--- AdaBoost Classifier Evaluation ---
Accuracy: 0.8134

Classification Report:
               precision    recall  f1-score   support

           0       0.81      0.90      0.85       157
           1       0.81      0.69      0.75       111

    accuracy                           0.81       268
   macro avg       0.81      0.80      0.80       268
weighted avg       0.81      0.81      0.81       268
```

#### XGBoost

```python
# --- 21. XGBOOST ---
# (Installation code and checks omitted for brevity)
try:
    from xgboost import XGBClassifier
    evaluate_model(XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='logloss'), "XGBoost Classifier")
except ImportError:
    print("XGBoost not installed. Skipping.")

print("\n✅ All models evaluated.")
```

**Output:**

```
--- XGBoost Classifier Evaluation ---
Accuracy: 0.8172

Classification Report:
               precision    recall  f1-score   support

           0       0.83      0.87      0.85       157
           1       0.79      0.74      0.76       111

    accuracy                           0.82       268
   macro avg       0.81      0.80      0.81       268
weighted avg       0.82      0.82      0.82       268

 All models evaluated.
```

### 8. Save Model and Encoders

```python
# 7. Save model and encoders
import joblib
joblib.dump(LogisticRegression, "best_model.pkl")
joblib.dump(LEC_Sex, "LEC_Sex.pkl")
joblib.dump(LEC_Embarked, "LEC_Embarked.pkl")

print("✅ Model and encoders saved successfully!")

## 👨🏽‍💻 Author

*Japhet Ujile*
📧 [assistant.rawlings@gmail.com](mailto:assistant.rawlings@gmail.com)
🌐 [GitHub](https://github.com/assistantrawlings-lgtm) | [LinkedIn](https://www.linkedin.com/in/japhet-ujile-838442148?utm_source=share&utm_campaign=share_via&utm_content=profile&utm_medium=android_app])

```

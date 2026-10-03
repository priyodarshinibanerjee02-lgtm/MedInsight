import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import GridSearchCV
from imblearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold, cross_val_score
import shap
from sklearn.metrics import roc_auc_score, roc_curve
import seaborn as sns
import numpy as np
from sklearn.metrics import confusion_matrix


# df = pd.read_csv('Stroke_prediction.csv')

df = pd.read_csv('MedInsight/Stroke_prediction.csv')

print(df.head())
print(df.shape)
print(df.info())

#missing values

print(df.isnull().sum())
print(df['bmi'].describe())
df['bmi'] = df['bmi'].fillna(df['bmi'].median())
print(df.isnull().sum())
print(df['stroke'].value_counts())

# =========================================
# DATASET VISUALIZATION
# 1. Stroke Class Distribution
# =========================================

import matplotlib.pyplot as plt

stroke_counts = df['stroke'].value_counts().sort_index()
plt.figure(figsize=(7, 5))
bars = plt.bar(
    ['No Stroke', 'Stroke'],
    stroke_counts.values
)
plt.title(
    'Stroke Class Distribution',
    fontsize=14,
    fontweight='bold'
)
plt.xlabel('Stroke Status')
plt.ylabel('Number of Patients')

# Add values above bars
for bar in bars:
    height = bar.get_height()
    plt.text(
        bar.get_x() + bar.get_width() / 2,
        height,
        f'{int(height)}',
        ha='center',
        va='bottom',
        fontsize=10
    )
plt.tight_layout()
plt.savefig(
    'stroke_class_distribution.png', dpi=300, bbox_inches='tight')
plt.show()

# =========================================
# 2. Age Distribution
# =========================================

plt.figure(figsize=(8, 5))
plt.hist(
    df['age'],
    bins=20
)
plt.title(
    'Age Distribution of Patients',
    fontsize=14,
    fontweight='bold'
)
plt.xlabel('Age')
plt.ylabel('Number of Patients')
plt.tight_layout()
plt.savefig(
    'age_distribution.png',
    dpi=300,
    bbox_inches='tight'
)
plt.show()

# =========================================
# 3. Gender vs Stroke
# =========================================

gender_stroke = pd.crosstab(
    df['gender'],
    df['stroke']
)

gender_stroke.columns = ['No Stroke', 'Stroke']

gender_stroke.plot(
    kind='bar',
    figsize=(8, 5)
)

plt.title(
    'Stroke Distribution by Gender',
    fontsize=14,
    fontweight='bold'
)

plt.xlabel('Gender')
plt.ylabel('Number of Patients')

plt.xticks(rotation=0)
plt.legend(title='Stroke Status')

plt.tight_layout()

plt.savefig(
    'gender_vs_stroke.png',
    dpi=300,
    bbox_inches='tight'
)

plt.show()
# =========================================
# 4. Hypertension vs Stroke
# =========================================

hypertension_stroke = pd.crosstab(
    df['hypertension'],
    df['stroke']
)
hypertension_stroke.columns = ['No Stroke', 'Stroke']
hypertension_stroke.index = ['No Hypertension', 'Hypertension']
hypertension_stroke.plot(
    kind='bar',
    figsize=(8, 5)
)
plt.title(
    'Stroke Distribution by Hypertension',
    fontsize=14,
    fontweight='bold'
)

plt.xlabel('Hypertension Status')
plt.ylabel('Number of Patients')
plt.xticks(rotation=0)
plt.legend(title='Stroke Status')
plt.tight_layout()
plt.savefig(
    'hypertension_vs_stroke.png',
    dpi=300,
    bbox_inches='tight'
)
plt.show()

# =========================================
# 5. Heart Disease vs Stroke
# =========================================

heart_stroke = pd.crosstab(
    df['heart_disease'],
    df['stroke']
)

heart_stroke.columns = ['No Stroke', 'Stroke']

heart_stroke.index = ['No Heart Disease', 'Heart Disease']

heart_stroke.plot(
    kind='bar',
    figsize=(8, 5)
)
plt.title(
    'Stroke Distribution by Heart Disease',
    fontsize=14,
    fontweight='bold'
)
plt.xlabel('Heart Disease Status')
plt.ylabel('Number of Patients')
plt.xticks(rotation=0)
plt.legend(title='Stroke Status')
plt.tight_layout()
plt.savefig(
    'heart_disease_vs_stroke.png',
    dpi=300,
    bbox_inches='tight'
)
plt.show()
# =========================================
# 6. BMI Category vs Stroke
# =========================================


bmi_df = df.copy()
bmi_df['bmi_category'] = pd.cut(
    bmi_df['bmi'],
    bins=[0, 18.5, 25, 30, float('inf')],
    labels=['Underweight', 'Healthy', 'Overweight', 'Obese']
)

bmi_stroke = pd.crosstab(
    bmi_df['bmi_category'],
    bmi_df['stroke']
)

bmi_stroke.columns = ['No Stroke', 'Stroke']

bmi_stroke.plot(
    kind='bar',
    figsize=(8, 5)
)

plt.title(
    'Stroke Distribution by BMI Category',
    fontsize=14,
    fontweight='bold'
)

plt.xlabel('BMI Category')
plt.ylabel('Number of Patients')

plt.xticks(rotation=0)
plt.legend(title='Stroke Status')

plt.tight_layout()

plt.savefig(
    'bmi_vs_stroke.png',
    dpi=300,
    bbox_inches='tight'
)
plt.show()


# Age vs Average Glucose Level Scatter Plot

plt.figure(figsize=(8, 6))
for stroke_value in [0, 1]:
    subset = df[df["stroke"] == stroke_value]
    plt.scatter(
        subset["age"],
        subset["avg_glucose_level"],
        label=f"Stroke = {stroke_value}",
        alpha=0.5
    )
plt.xlabel("Age")
plt.ylabel("Average Glucose Level")
plt.title("Age vs Average Glucose Level by Stroke Status")
plt.legend()
plt.tight_layout()
plt.savefig("age_vs_glucose_scatter.png", dpi=300)
plt.show()

# =========================================
# 7. Average Glucose Level vs Stroke
# =========================================

glucose_df = df.copy()
glucose_df['glucose_category'] = pd.cut(
    glucose_df['avg_glucose_level'],
    bins=[0, 100, 140, float('inf')],
    labels=['Normal', 'Elevated', 'High']
)
glucose_stroke = pd.crosstab(
    glucose_df['glucose_category'],
    glucose_df['stroke']
)
glucose_stroke.columns = ['No Stroke', 'Stroke']
glucose_stroke.plot(
    kind='bar',
    figsize=(8, 5)
)
plt.title(
    'Stroke Distribution by Average Glucose Level',
    fontsize=14,
    fontweight='bold'
)
plt.xlabel('Average Glucose Level')
plt.ylabel('Number of Patients')
plt.xticks(rotation=0)
plt.legend(title='Stroke Status')
plt.tight_layout()
plt.savefig(
    'glucose_vs_stroke.png',
    dpi=300,
    bbox_inches='tight'
)
plt.show()

# =========================================
# 8. Smoking Status vs Stroke
# =========================================

smoking_df = df.copy()

smoking_stroke = pd.crosstab(
    smoking_df['smoking_status'],
    smoking_df['stroke']
)

smoking_stroke.columns = ['No Stroke', 'Stroke']

smoking_stroke.plot(
    kind='bar',
    figsize=(9, 5)
)

plt.title(
    'Stroke Distribution by Smoking Status',
    fontsize=14,
    fontweight='bold'
)

plt.xlabel('Smoking Status')
plt.ylabel('Number of Patients')

plt.xticks(rotation=15)
plt.legend(title='Stroke Status')

plt.tight_layout()

plt.savefig(
    'smoking_status_vs_stroke.png',
    dpi=300,
    bbox_inches='tight'
)

plt.show()

# =========================================
# 9. Correlation Heatmap
# =========================================

import seaborn as sns

# Select numerical features only
correlation_data = df.select_dtypes(include=['number']).drop(columns = ['id'])

plt.figure(figsize=(10, 7))
sns.heatmap(
    correlation_data.corr(),
    annot=True,
    cmap='coolwarm',
    fmt='.2f',
    linewidths=0.5
)
plt.title(
    'Correlation Heatmap of Numerical Features',
    fontsize=14,
    fontweight='bold'
)

plt.tight_layout()
plt.savefig(
    'correlation_heatmap.png',
    dpi=300,
    bbox_inches='tight'
)
plt.show()

print(df['gender'].unique())
print(df['ever_married'].unique())
print(df['work_type'].unique())
print(df['Residence_type'].unique())
print(df['smoking_status'].unique())

le = LabelEncoder()
df['gender'] = le.fit_transform(df['gender'])
print(df['gender'].unique())

df['ever_married'] = df['ever_married'].map({'No': 0 , 'Yes': 1})
print(df['ever_married'].unique())
df['Residence_type']= df['Residence_type'].map({'Rural':0, 'Urban': 1})
print(df['Residence_type'].unique())

#one hot 
df = pd.get_dummies(df, columns = ['work_type'] , dtype = int)
print(df.head())
df = pd.get_dummies(df , columns = ['smoking_status'] , dtype = int)
print(df.head())
print(df.info())
#remove patient identifier as not a predictive feature
df = df.drop('id' , axis = 1)
print(df.head())
print(df.shape)

# seperate inputs and target
X = df.drop('stroke' , axis=1)
y = df['stroke']
print(X.shape)
print(y.shape)

#train - test - split

X_train , X_test , y_train , y_test = train_test_split(X,y, test_size = 0.2 , random_state = 42, stratify = y)

import joblib
joblib.dump(X_train , "X_train.pkl")
joblib.dump(X_test, "X_test.pkl")
joblib.dump(y_test, "y_test.pkl")

#stratify coz data highly imbalanced ; ensures the train and test sets maintain approximately the same stroke/non-stroke proportion.
print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)

#scaling

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

#Create LR model
model_lr = LogisticRegression(max_iter = 1000)
#train model
model_lr.fit(X_train_scaled , y_train)
#max_iter=1000 simply gives the model enough iterations to converge.

#Predict
y_pred_lr = model_lr.predict(X_test_scaled)
print(y_pred_lr[:10])

#Evaluate
accuracy_lr = accuracy_score(y_test , y_pred_lr)
print("Logistic Regression Accuracy:",accuracy_lr)

recall_lr = recall_score(y_test ,  y_pred_lr)
print("Logistic Regression Recall:", recall_lr)

precision_lr = precision_score(y_test , y_pred_lr)
print("Logistic Regression Precision" , precision_lr)

f1_lr = f1_score(y_test , y_pred_lr)
print("Logistic Regression F1: ", f1_lr)

#Confusion Matrix

cm = confusion_matrix(y_test , y_pred_lr)
print(cm)

#importance to minority class (stroke = 1)
model_lr_balanced = LogisticRegression(
    class_weight = 'balanced',
    max_iter = 1000
)
#train
model_lr_balanced.fit(X_train_scaled , y_train)
#predict
y_pred_lr_balanced = model_lr_balanced.predict(X_test_scaled)
print(y_pred_lr_balanced[:10])

recall_lr_balanced = recall_score(y_test, y_pred_lr_balanced)

print("Balanced Logistic Regression Recall:", recall_lr_balanced)

accuracy_lr_balanced = accuracy_score(y_test , y_pred_lr_balanced)
print("Balanced Logistic Regression Accuracy:" , accuracy_lr_balanced)

precision_lr_balanced = precision_score(y_test , y_pred_lr_balanced)
print("Balanced Logistic Regression  Precision:" , precision_lr_balanced)

f1_lr_balanced = f1_score(y_test , y_pred_lr_balanced)
print("Balanced Logistic Regression F1:" , f1_lr_balanced)

#C_Matrix again
cm_balanced = confusion_matrix(y_test , y_pred_lr_balanced)
print(cm_balanced)

k = 5 
# create
knn = KNeighborsClassifier(n_neighbors = 5)
knn.fit(X_train_scaled , y_train)

#predict
y_pred_knn = knn.predict(X_test_scaled)
print(y_pred_knn[:10])
#recall
recall_knn = recall_score(y_test, y_pred_knn)
print("KNN Recall:", recall_knn)
accuracy_knn = accuracy_score(y_test, y_pred_knn)
precision_knn = precision_score(y_test, y_pred_knn)
f1_knn = f1_score(y_test, y_pred_knn)

print("KNN Accuracy:", accuracy_knn)
print("KNN Precision:", precision_knn)
print("KNN F1:", f1_knn)

#dec tree
# Create Decision Tree model
dt = DecisionTreeClassifier(random_state=42)
# Train the Decision Tree
dt.fit(X_train, y_train)
# Predict using Decision Tree
y_pred_dt = dt.predict(X_test)

print(y_pred_dt[:10])

recall_dt = recall_score(y_test, y_pred_dt)
print("Decision Tree Recall:", recall_dt)
accuracy_dt = accuracy_score(y_test, y_pred_dt)
precision_dt = precision_score(y_test, y_pred_dt)
f1_dt = f1_score(y_test, y_pred_dt)

print("Decision Tree Accuracy:", accuracy_dt)
print("Decision Tree Precision:", precision_dt)
print("Decision Tree F1:", f1_dt)

# Create the Random Forest model
# n_estimators = number of decision trees in the forest
# random_state = gives reproducible results
rf = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

# Train the Random Forest
# Decision Tree/Random Forest do NOT require scaled data
rf.fit(X_train, y_train)

# Make predictions on unseen test data
y_pred_rf = rf.predict(X_test)

# Evaluate the model
accuracy_rf = accuracy_score(y_test, y_pred_rf)
recall_rf = recall_score(y_test, y_pred_rf)
precision_rf = precision_score(y_test, y_pred_rf)
f1_rf = f1_score(y_test, y_pred_rf)

# Display results
print("Random Forest Accuracy:", accuracy_rf)
print("Random Forest Recall:", recall_rf)
print("Random Forest Precision:", precision_rf)
print("Random Forest F1:", f1_rf)

# Confusion Matrix
cm_rf = confusion_matrix(y_test, y_pred_rf)
print("Random Forest Confusion Matrix:")
print(cm_rf)

#SMOTE
smote = SMOTE(random_state=42)
X_train_smote, y_train_smote = smote.fit_resample(
    X_train,
    y_train
)
print(y_train_smote.value_counts())

# Scale the SMOTE-balanced training data
X_train_smote_scaled = scaler.fit_transform(X_train_smote)

# Keep the test data scaling unchanged
X_test_scaled = scaler.transform(X_test)
# Logistic Regression trained on SMOTE-balanced data
lr_smote = LogisticRegression(max_iter=1000)

lr_smote.fit(X_train_smote_scaled, y_train_smote)
# Predict on the ORIGINAL untouched test set
y_pred_smote = lr_smote.predict(X_test_scaled)
print(y_pred_smote[:10])

#RECALL
recall_smote = recall_score(y_test, y_pred_smote)
print("SMOTE Logistic Regression Recall:", recall_smote)
#ACCURACY
accuracy_smote = accuracy_score(y_test, y_pred_smote)
print("SMOTE Accuracy:", accuracy_smote)
#PRECISION
precision_smote = precision_score(y_test, y_pred_smote)
print("SMOTE Precision:", precision_smote)
#F1
f1_smote = f1_score(y_test, y_pred_smote)
print("SMOTE F1:", f1_smote)

#CONFUSION MATRIX
cm_smote = confusion_matrix(y_test, y_pred_smote)
print("SMOTE Confusion Matrix:")
print(cm_smote)

# Create Random Forest
rf_smote = RandomForestClassifier(
    n_estimators=100,      # Number of trees
    random_state=42
)

# Train Random Forest on SMOTE-balanced training data
rf_smote.fit(X_train_smote, y_train_smote)

# Predict on the ORIGINAL test data
y_pred_rf_smote = rf_smote.predict(X_test)

# Evaluate
accuracy_rf_smote = accuracy_score(y_test, y_pred_rf_smote)
recall_rf_smote = recall_score(y_test, y_pred_rf_smote)
precision_rf_smote = precision_score(y_test, y_pred_rf_smote)
f1_rf_smote = f1_score(y_test, y_pred_rf_smote)

# print("SMOTE Random Forest Accuracy:", accuracy_rf_smote)
# print("SMOTE Random Forest Recall:", recall_rf_smote)
# print("SMOTE Random Forest Precision:", precision_rf_smote)
# print("SMOTE Random Forest F1:", f1_rf_smote)

# Create Logistic Regression
lr_tune = LogisticRegression(max_iter=2000)

# Values we want to test
param_grid_lr = {
    'C': [0.01, 0.1, 1, 10],
    'class_weight': [None, 'balanced']
}
# Try every combination using 5-fold cross-validation
grid_lr = GridSearchCV(
    lr_tune,
    param_grid_lr,
    cv=5,
    scoring='f1',
    n_jobs=-1
)
# Train using the SMOTE-balanced training data
grid_lr.fit(X_train_smote_scaled, y_train_smote)# Best hyperparameter combination found
print("Best Parameters:", grid_lr.best_params_)
# Best average F1-score from 5-fold cross-validation
print("Best Cross-Validation F1:", grid_lr.best_score_)

# Get the best Logistic Regression model
best_lr = grid_lr.best_estimator_

# Predict on completely unseen test data
y_pred_best_lr = best_lr.predict(X_test_scaled)
print(y_pred_best_lr[:10])
# Evaluate the tuned Logistic Regression on unseen test data

accuracy_best = accuracy_score(y_test, y_pred_best_lr)
recall_best = recall_score(y_test, y_pred_best_lr)
precision_best = precision_score(y_test, y_pred_best_lr)
f1_best = f1_score(y_test, y_pred_best_lr)

print("Tuned LR Accuracy:", accuracy_best)
print("Tuned LR Recall:", recall_best)
print("Tuned LR Precision:", precision_best)
print("Tuned LR F1:", f1_best)

# Get probability of Stroke (class = 1)
y_prob = best_lr.predict_proba(X_test_scaled)[:, 1]
print(y_prob[:10])


# # Change the decision threshold from 0.50 to 0.30
# threshold = 0.30
# # Convert probabilities into predictions
# y_pred_threshold = (y_prob >= threshold).astype(int)

# # Evaluate
# print("Recall:", recall_score(y_test, y_pred_threshold))
# print("Precision:", precision_score(y_test, y_pred_threshold))
# print("F1:", f1_score(y_test, y_pred_threshold))

# Test different probability thresholds
thresholds = [0.20, 0.30, 0.40, 0.50, 0.60]

for threshold in thresholds:

    # Convert probability into 0/1 prediction
    y_pred = (y_prob >= threshold).astype(int)

    # Calculate metrics
    recall = recall_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    print(
        "Threshold:", threshold,
        "| Recall:", round(recall, 3),
        "| Precision:", round(precision, 3),
        "| F1:", round(f1, 3)
    )

# =========================================
# ROC CURVE AND AUC
# =========================================

# Calculate AUC
auc_score = roc_auc_score(y_test, y_prob)

print("ROC-AUC Score:", auc_score)

# Calculate ROC curve
fpr, tpr, thresholds_roc = roc_curve(y_test, y_prob)

# Plot ROC Curve
plt.figure(figsize=(7, 5))

plt.plot(
    fpr,
    tpr,
    label=f"ROC Curve (AUC = {auc_score:.3f})"
)

plt.plot(
    [0, 1],
    [0, 1],
    linestyle='--',
    label='Random Classifier'
)

plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend()
plt.grid(alpha=0.3)

plt.tight_layout()

plt.savefig(
    'roc_curve.png',
    dpi=300,
    bbox_inches='tight'
)

plt.show()

# SMOTE happens only inside each training fold
pipeline = Pipeline([
    ('smote', SMOTE(random_state=42)),
    ('model', LogisticRegression(max_iter=2000))
])
# Keeps the stroke/non-stroke ratio similar in every fold
cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)
# Evaluate using F1-score
cv_scores = cross_val_score(
    pipeline,
    X_train,
    y_train,
    cv=cv,
    scoring='f1'
)
print("CV F1 Scores:", cv_scores)
print("Mean CV F1:", cv_scores.mean())

cv_recall_scores = cross_val_score(
    pipeline,
    X_train,
    y_train,
    cv=cv,
    scoring='recall'
)
print("CV Recall Scores:", cv_recall_scores)
print("Mean CV Recall:", cv_recall_scores.mean())

# Decision Tree
dt_cv = DecisionTreeClassifier(random_state=42)

# 5-fold cross-validation using F1
dt_cv_scores = cross_val_score(
    dt_cv,
    X_train,
    y_train,
    cv=cv,
    scoring='f1'
)

print("Decision Tree CV F1 Scores:", dt_cv_scores)
print("Decision Tree Mean CV F1:", dt_cv_scores.mean())

rf_cv = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)
rf_cv_scores = cross_val_score(
    rf_cv,
    X_train,
    y_train,
    cv=cv,
    scoring='f1'
)
print("Random Forest CV F1 Scores:", rf_cv_scores)
print("Random Forest Mean CV F1:", rf_cv_scores.mean())

knn_cv = KNeighborsClassifier(n_neighbors=5)

knn_cv_scores = cross_val_score(
    knn_cv,
    X_train_scaled,
    y_train,
    cv=cv,
    scoring='f1'
)
print("KNN CV F1 Scores:", knn_cv_scores)
print("KNN Mean CV F1:", knn_cv_scores.mean())


# Balanced Logistic Regression with Cross-Validation

pipeline_balanced = Pipeline([
    ('model', LogisticRegression(
        class_weight='balanced',
        max_iter=2000
    ))
])

cv_balanced_f1 = cross_val_score(
    pipeline_balanced,
    X_train,
    y_train,
    cv=cv,
    scoring='f1'
)

cv_balanced_recall = cross_val_score(
    pipeline_balanced,
    X_train,
    y_train,
    cv=cv,
    scoring='recall'
)

print("Balanced LR CV F1:", cv_balanced_f1)
print("Mean Balanced LR CV F1:", cv_balanced_f1.mean())

print("Balanced LR CV Recall:", cv_balanced_recall)
print("Mean Balanced LR CV Recall:", cv_balanced_recall.mean())

# Hyperparameter tuning for Balanced Logistic Regression

lr_balanced_tune = LogisticRegression(
    class_weight='balanced',
    max_iter=2000
)
param_grid_balanced = {
    'C': [0.01, 0.1, 1, 10, 100]
}
grid_balanced = GridSearchCV(
    lr_balanced_tune,
    param_grid_balanced,
    cv=cv,
    scoring='f1',
    n_jobs=-1
)
grid_balanced.fit(X_train, y_train)

print("Best C:", grid_balanced.best_params_)
print("Best CV F1:", grid_balanced.best_score_)

# Best tuned Balanced Logistic Regression
best_balanced_lr = grid_balanced.best_estimator_

import joblib
joblib.dump(best_balanced_lr, "balanced_lr_model.pkl")
# joblib.dump(X_train , "x_train.pkl")

#trained model saved to a file

# Predict using the same type of data used during training
y_pred_best_balanced = best_balanced_lr.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred_best_balanced))
print("Recall:", recall_score(y_test, y_pred_best_balanced))
print("Precision:", precision_score(y_test, y_pred_best_balanced))
print("F1:", f1_score(y_test, y_pred_best_balanced))

# =========================================
# TRAINING VS TEST ACCURACY
# =========================================

train_accuracy_lr = accuracy_score(
    y_train,
    model_lr.predict(X_train_scaled)
)

test_accuracy_lr = accuracy_score(
    y_test,
    y_pred_lr
)

train_accuracy_knn = accuracy_score(
    y_train,
    knn.predict(X_train_scaled)
)

test_accuracy_knn = accuracy_score(
    y_test,
    y_pred_knn
)

train_accuracy_dt = accuracy_score(
    y_train,
    dt.predict(X_train)
)

test_accuracy_dt = accuracy_score(
    y_test,
    y_pred_dt
)

train_accuracy_rf = accuracy_score(
    y_train,
    rf.predict(X_train)
)

test_accuracy_rf = accuracy_score(
    y_test,
    y_pred_rf
)

train_accuracy_smote = accuracy_score(
    y_train,
    lr_smote.predict(scaler.transform(X_train))
)

test_accuracy_smote = accuracy_score(
    y_test,
    y_pred_smote
)

train_accuracy_balanced = accuracy_score(
    y_train,
    best_balanced_lr.predict(X_train)
)

test_accuracy_balanced = accuracy_score(
    y_test,
    y_pred_best_balanced
)

print("\nTraining vs Test Accuracy")

print("Logistic Regression:",
      train_accuracy_lr,
      test_accuracy_lr)

print("KNN:",
      train_accuracy_knn,
      test_accuracy_knn)

print("Decision Tree:",
      train_accuracy_dt,
      test_accuracy_dt)

print("Random Forest:",
      train_accuracy_rf,
      test_accuracy_rf)

print("SMOTE + Logistic Regression:",
      train_accuracy_smote,
      test_accuracy_smote)

print("Tuned Balanced Logistic Regression:",
      train_accuracy_balanced,
      test_accuracy_balanced)

# =========================================
# TRAINING VS TEST ACCURACY GRAPH
# =========================================

models = [
    "Logistic Regression",
    "KNN",
    "Decision Tree",
    "Random Forest",
    "SMOTE + Logistic Regression",
    "Tuned Balanced Logistic Regression"
]

train_accuracies = [
    train_accuracy_lr,
    train_accuracy_knn,
    train_accuracy_dt,
    train_accuracy_rf,
    train_accuracy_smote,
    train_accuracy_balanced
]

test_accuracies = [
    test_accuracy_lr,
    test_accuracy_knn,
    test_accuracy_dt,
    test_accuracy_rf,
    test_accuracy_smote,
    test_accuracy_balanced
]

x = np.arange(len(models))
width = 0.35

plt.figure(figsize=(12, 6))

bars1 = plt.bar(
    x - width/2,
    np.array(train_accuracies) * 100,
    width,
    label="Training Accuracy"
)

bars2 = plt.bar(
    x + width/2,
    np.array(test_accuracies) * 100,
    width,
    label="Test Accuracy"
)

plt.xlabel("Models")
plt.ylabel("Accuracy (%)")
plt.title(
    "Training vs Test Accuracy of Evaluated Models",
    fontsize=14,
    fontweight="bold"
)

plt.xticks(
    x,
    models,
    rotation=25,
    ha="right"
)

plt.ylim(0, 105)
plt.legend()

# Add percentage values above bars
for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            height,
            f"{height:.2f}%",
            ha="center",
            va="bottom",
            fontsize=9
        )

plt.tight_layout()

plt.savefig(
    "training_vs_test_accuracy.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()

# =========================================
# FINAL MODEL PERFORMANCE GRAPH
# =========================================

metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
values = [
    accuracy_score(y_test, y_pred_best_balanced) * 100,
    precision_score(y_test, y_pred_best_balanced) * 100,
    recall_score(y_test, y_pred_best_balanced) * 100,
    f1_score(y_test, y_pred_best_balanced) * 100
]

plt.figure(figsize=(7, 5))
bars = plt.bar(metrics, values)
plt.title(
    'Performance of Tuned Balanced Logistic Regression',
    fontsize=14,
    fontweight='bold'
)

plt.xlabel('Evaluation Metrics')
plt.ylabel('Score (%)')
for bar, value in zip(bars, values):
    plt.text(
        bar.get_x() + bar.get_width() / 2,
        value,
        f'{value:.2f}%',
        ha='center',
        va='bottom'
    )
plt.ylim(0, 100)
plt.tight_layout()

plt.savefig(
    'final_model_performance.png',
    dpi=300,
    bbox_inches='tight'
)
plt.show()

# =========================================
# FINAL CONFUSION MATRIX
# Tuned Balanced Logistic Regression
# =========================================

cm_final = confusion_matrix(y_test, y_pred_best_balanced)

plt.figure(figsize=(7, 5))

sns.heatmap(
    cm_final,
    annot=True,
    fmt='d',
    cmap='Blues',
    xticklabels=['No Stroke', 'Stroke'],
    yticklabels=['No Stroke', 'Stroke']
)

plt.title(
    'Confusion Matrix - Tuned Balanced Logistic Regression',
    fontsize=14,
    fontweight='bold'
)

plt.xlabel('Predicted Outcome')
plt.ylabel('Actual Outcome')

plt.tight_layout()

plt.savefig(
    'final_confusion_matrix.png',
    dpi=300,
    bbox_inches='tight'
)

plt.show()

comparison = pd.DataFrame({
    "Model": [
        "Logistic Regression",
        "KNN",
        "Decision Tree",
        "Random Forest",
        "SMOTE + Logistic Regression",
        "Balanced Logistic Regression"
    ],
    
    "Accuracy": [
        accuracy_lr,
        accuracy_knn,
        accuracy_dt,
        accuracy_rf,
        accuracy_smote,
        accuracy_lr_balanced
    ],
    
    "Recall": [
        recall_lr,
        recall_knn,
        recall_dt,
        recall_rf,
        recall_smote,
        recall_lr_balanced
    ],
    
    "Precision": [
        precision_lr,
        precision_knn,
        precision_dt,
        precision_rf,
        precision_smote,
        precision_lr_balanced
    ],
    
    "F1": [
        f1_lr,
        f1_knn,
        f1_dt,
        f1_rf,
        f1_smote,
        f1_lr_balanced
    ]
})
print(comparison.round(3))

# =========================================
# MODEL PERFORMANCE COMPARISON GRAPH
# =========================================

models = comparison["Model"]

accuracy = comparison["Accuracy"] * 100
precision = comparison["Precision"] * 100
recall = comparison["Recall"] * 100
f1 = comparison["F1"] * 100

x = np.arange(len(models))
width = 0.2

plt.figure(figsize=(12, 6))

plt.bar(x - 1.5*width, accuracy, width, label="Accuracy")
plt.bar(x - 0.5*width, precision, width, label="Precision")
plt.bar(x + 0.5*width, recall, width, label="Recall")
plt.bar(x + 1.5*width, f1, width, label="F1-Score")

plt.xlabel("Models")
plt.ylabel("Performance (%)")
plt.title("Performance Comparison of Evaluated Models")

plt.xticks(
    x,
    models,
    rotation=25,
    ha="right"
)

plt.legend()
plt.tight_layout()

plt.savefig(
    "model_performance_comparison.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()

#create shap explainer
explainer = shap.LinearExplainer(best_balanced_lr , X_train)
# #calculate shap values
# shap_values = explainer(X_test)

#which features are important overall

# shap.summary_plot(shap_values , X_test)

# Explain one patient
# patient_index = 0
# shap.plots.waterfall(shap_values[patient_index])

print(X_train.columns.tolist())
# shap.force_plot(
#     explainer.expected_value,
#     shap_values[patient_index],
#     X_test.iloc[patient_index],
#     matplotlib=True
# )

#create one sample patient

sample_patient = pd.DataFrame({
    'gender': [1],
    'age':[65],
    'hypertension':[1],
    'heart_disease':[0],
    'ever_married':[1],
    'Residence_type':[1],
    'avg_glucose_level': [180],
    'bmi':[30],
    'work_type_Govt_job': [0],
    'work_type_Never_worked':[0],
    'work_type_Private':[1],
    'work_type_Self-employed':[0],
    'work_type_children': [0],
    'smoking_status_Unknown':[0],
    'smoking_status_formerly smoked':[0],
    'smoking_status_never smoked':[0],
    'smoking_status_never smoked': [1],
    'smoking_status_smokes':[0]
})
# Predict stroke for the sample patient
sample_prediction = best_balanced_lr.predict(sample_patient)

# Get stroke probability
sample_probability = best_balanced_lr.predict_proba(sample_patient)[:, 1]

print("Prediction:", sample_prediction[0])
print("Stroke Probability:", sample_probability[0])

# Explain the sample patient
sample_shap = explainer(sample_patient)

# Show which features influenced this prediction
shap.plots.waterfall(sample_shap[0])
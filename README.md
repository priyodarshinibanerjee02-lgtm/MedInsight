# MedInsight – Stroke Risk Screening and Explainable Prediction System

MedInsight is an explainable machine learning-based stroke risk screening system developed using Python and Streamlit.

The system predicts the risk of stroke from patient health and lifestyle factors and provides explanations for the prediction using SHAP (SHapley Additive exPlanations).

## Project Overview

Stroke is a serious medical condition where early identification of risk factors can support timely screening and awareness.

MedInsight combines:

- Machine Learning-based risk prediction
- Class imbalance handling
- Model evaluation and comparison
- Hyperparameter tuning
- SHAP-based explainability
- Interactive Streamlit interface
- Automated PDF reporting

> **Note:** MedInsight is a screening and prediction-support system, not a medical diagnosis tool.

## Dataset

The project uses a stroke prediction dataset containing:

- 5,110 patient records
- 10 input features
- 4,861 non-stroke cases
- 249 stroke cases

The dataset contains demographic, health and lifestyle-related information.

## Machine Learning Workflow

The project follows these major steps:

1. Data loading
2. Data preprocessing
3. Missing-value handling
4. Categorical feature encoding
5. Patient ID removal
6. Feature-target separation
7. Train-test splitting
8. Feature scaling
9. Class imbalance handling
10. Model training
11. Model comparison
12. Hyperparameter tuning
13. Final model selection
14. SHAP-based explainability
15. Risk assessment and reporting

## Models Evaluated

The following approaches were evaluated:

- Logistic Regression
- K-Nearest Neighbors (KNN)
- Decision Tree
- Random Forest
- SMOTE + Logistic Regression
- Balanced Logistic Regression

## Final Model

The final model used in MedInsight is a **Tuned Balanced Logistic Regression** model.

Hyperparameter tuning was performed using **GridSearchCV with 5-fold cross-validation**, using F1-score as the scoring metric.

## Model Performance

The final model achieved:

- Recall: **80%**
- ROC-AUC: **82.80%**

The final confusion matrix contained:

- True Positive (TP): 40
- True Negative (TN): 723
- False Positive (FP): 249
- False Negative (FN): 10

Recall was given particular importance because the project focuses on stroke-risk screening, where identifying actual stroke cases is important.

## Explainable AI with SHAP

MedInsight uses **SHAP (SHapley Additive exPlanations)** to make model predictions more understandable.

SHAP provides:

### Individual Explanation

Shows which features contributed to a particular patient's prediction.

### Global Feature Importance

Shows which features generally have greater influence on the model's predictions.

The project uses `LinearExplainer` for the final logistic regression model.

## Streamlit Application

The MedInsight application provides an interactive workflow:

```text
User Login
     ↓
Patient Input
     ↓
Risk Assessment
     ↓
Explainable Prediction
     ↓
Report Generation
```

## Automated Reporting

The system generates a PDF report containing:

- Patient information
- Health factors
- Lifestyle factors
- Estimated stroke probability
- Model prediction
- Top contributing factors
- SHAP contribution details
- Personalised health guidance

## Model Analytics

The Model Analytics section provides insights into the performance and behaviour of the final model.

It includes:

- Final Model Performance
- Reason for Hyperparameter Tuning
- Importance of Recall
- Confusion Matrix
- Individual SHAP Explanation
- SHAP Feature Importance

## Advantages

- Handles class imbalance
- Focuses on stroke-case recall
- Provides explainable predictions
- Includes model analytics
- Provides automated PDF reporting
- Offers an interactive screening interface

## Limitations

- Lower precision results in a relatively high number of false positives
- Model performance depends on the quality and representativeness of the dataset
- The system is intended for screening support and not medical diagnosis
- Further external and clinical validation would be required before real-world clinical use

## Future Scope

Possible future improvements include:

- Larger and more diverse datasets
- External and clinical validation
- Improved precision-recall balance
- Advanced machine learning and deep learning approaches
- Real-world clinical integration

## Project Structure

```text
MedInsight/
│
├── app.py
├── stroke.py
├── balanced_lr_model.pkl
├── X_train.pkl
├── X_test.pkl
├── y_test.pkl
├── Stroke_prediction.csv
│
├── LOGO.png
├── age_distribution.png
├── bmi_vs_stroke.png
├── correlation_heatmap.png
├── final_confusion_matrix.png
├── final_model_performance.png
├── gender_vs_stroke.png
├── glucose_vs_stroke.png
├── heart_disease_vs_stroke.png
├── hypertension_vs_stroke.png
├── model_performance_comparison.png
├── roc_curve.png
├── smoking_status_vs_stroke.png
├── stroke_class_distribution.png
├── training_vs_test_accuracy.png
│
├── MedInsight_PPT_.pptx
├── Stroke_Risk_Prediction_and_Screening.pdf
├── requirements.txt
└── .gitignore
```

## Technologies Used

- Python
- Streamlit
- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- Seaborn
- SHAP
- Joblib
- ReportLab


## Author

Developed as an academic and internship project in the field of Machine Learning and Explainable AI.
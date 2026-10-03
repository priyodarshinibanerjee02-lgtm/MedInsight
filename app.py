import streamlit as st
import pandas as pd
import os
import joblib
import shap
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap,PowerNorm
from io import BytesIO
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle
)
from reportlab.platypus import Flowable
from reportlab.graphics.shapes import Drawing, Rect, String, Line
from reportlab.platypus import KeepTogether
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix
import seaborn as sns
import numpy as np
from PIL import Image
import base64


st.set_page_config(
    page_title="MedInsight",
    layout="wide",
    initial_sidebar_state="expanded"
)

#===========================
#MEDINSIGHT LOGO
#===========================

logo_path = os.path.join(
    os.path.dirname(__file__),
    "LOGO.png"
)
logo = Image.open(logo_path)

# =========================
# LOGIN SYSTEM
# =========================

if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

# =========================
# LOGIN PAGE
# =========================

if not st.session_state.logged_in:

    # st.title("MedInsight")
    # st.markdown(
    # """
    # <div style="
    #     display: flex;
    #     justify-content: center;
    #     align-items: center;
    #     margin: 10px 0 20px 0;
    # ">
    # """,
    # unsafe_allow_html = True
    # )
    # col1, col2, col3 = st.columns([1, 2, 1])
    # with col2:
    #     st.image(
    #         logo_path,
    #         width=200
    #     )
    with open(logo_path, "rb") as f:
        logo_bytes = f.read()

    logo_base64 = base64.b64encode(logo_bytes).decode()

    st.markdown(
        f"""
        <div style="
            display: flex;
            justify-content: center;
            align-items: center;
            width: 100%;
            margin: 5px 0 18px 0;
        ">
            <img
                src="data:image/png;base64,{logo_base64}"
                style="
                    width: 180px;
                    height: 180px;
                    object-fit: contain;
                    display: block;
                "
            >
        </div>
        """,
        unsafe_allow_html=True
    )
    # LOGIN DESCRIPTION
    st.markdown(
        """
        <div style="
            text-align: center;
            font-size: 18px;
            font-weight: 500;
            margin-bottom: 20px;
        ">
            Login to access the Stroke Risk Screening System
        </div>
        """,
        unsafe_allow_html=True
    )

    # st.write("Login to access the Stroke Risk Screening System")
    username = st.text_input(
        "👤 Username",
        placeholder="Enter username"
    )

    password = st.text_input(
        " 🔐Password",
        type="password",
        placeholder="Enter password"
    )

    if st.button("Login"):
        if username == "admin" and password == "admin123":
            st.session_state.logged_in = True
            st.success("Login Successful!")
            st.rerun()

        else:
            st.error("Invalid username or password.")

    st.info(
        "🔑 Default Login\n\n"
        "Username: admin\n\n"
        "Password: admin123"
    )
    st.stop()

# =========================
# SIDEBAR
# =========================
    
with st.sidebar:
    # st.title("🩺 MedInsight")

    # st.image(
    #     logo,
    #     width=180
    # )
    # ==============================
    # MEDINSIGHT SIDEBAR BRANDING
    # ==============================
    with open(logo_path, "rb") as f:
        sidebar_logo_bytes = f.read()
    sidebar_logo_base64 = base64.b64encode(sidebar_logo_bytes).decode()
    st.markdown(
         f"""
        <div style="
            text-align: center;
            padding: 10px 0 15px 0;
        ">
            <img
                src="data:image/png;base64,{sidebar_logo_base64}"
                style="
                    width: 145px;
                    height: 145px;
                    object-fit: contain;
                    display: block;
                    margin: 0 auto;
                "
            >
        </div>
        """,
        unsafe_allow_html=True
    )
    st.write("Logged in as: admin")
    st.divider()
    page = st.radio(
        "Navigation",
        ["🩺 Patient Screening", "📊 Model Analytics"]
    )
    st.divider()

    if st.button("🚪 Logout", use_container_width=True):
        st.session_state.logged_in = False
        st.rerun()


    
# base_path = os.path.dirname(os.path.dirname(__file__))
base_path = os.path.dirname(__file__)
model_path = os.path.join(base_path, "balanced_lr_model.pkl")
X_train_path = os.path.join(base_path, "X_train.pkl")
X_test_path = os.path.join(base_path, "X_test.pkl")
y_test_path = os.path.join(base_path, "y_test.pkl")
best_balanced_lr = joblib.load(model_path)
X_train = joblib.load(X_train_path)
# X_test = joblib.load(os.path.join(base_path, "X_test.pkl"))
# y_test = joblib.load(os.path.join(base_path, "y_test.pkl"))
X_test = joblib.load(X_test_path)
y_test = joblib.load(y_test_path)

#SHAP Explainer

explainer = shap.LinearExplainer(
        best_balanced_lr ,
        X_train
    )

# =========================
# PAGE NAVIGATION
# =========================

if page == "📊 Model Analytics":
    st.title("📊 Model Analytics")
    # st.write("Model performance and analytics will be shown here.")
    st.subheader("Final Model")
    st.info(
        "Tuned Balanced Logistic Regression\n\n"
        "Selected as the final model after model comparison, "
        "class balancing and hyperparameter tuning."
    )
    # st.stop()
    #=============================
    #MODEL PERFORMANCE
    #=============================
    st.subheader("Model Performance")
    #Predictions from final tuned balanced LR model
    y_pred_analytics = best_balanced_lr.predict(X_test)

    # Calculate performance metrics
    accuracy = accuracy_score(y_test, y_pred_analytics)
    precision = precision_score(y_test, y_pred_analytics, zero_division=0)
    recall = recall_score(y_test, y_pred_analytics, zero_division=0)
    f1 = f1_score(y_test, y_pred_analytics, zero_division=0)

    # Display metrics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Accuracy", f"{accuracy * 100:.2f}%")
    with col2:
        st.metric("Precision", f"{precision * 100:.2f}%")
    with col3:
        st.metric("Recall", f"{recall * 100:.2f}%")
    with col4:
        st.metric("F1 Score", f"{f1 * 100:.2f}%")

# =========================
# MODEL COMPARISON
# =========================

    st.subheader("Model Comparison")
    comparison_display = pd.DataFrame({
        "Model": [
            "Logistic Regression",
            "KNN",
            "Decision Tree",
            "Random Forest",
            "SMOTE + Logistic Regression",
            "Balanced Logistic Regression"
        ],
        "Accuracy": [
            95.21, 94.81, 91.10, 94.91, 93.25, 74.66
        ],
        "Recall": [
            2.00, 2.00, 20.00, 2.00, 38.00, 80.00
        ],
        "Precision": [
            100.00, 20.00, 16.39, 25.00, 33.33, 13.84
        ],
        "F1 Score": [
            3.92, 3.64, 18.02, 3.70, 35.51, 23.60
        ]
    })
    st.dataframe(
        comparison_display,
        hide_index=True,
        use_container_width=True
    )
# =================================
# WHY ACCURACY ALONE IS NOT ENOUGH
# =================================
    st.info(
        "**Why Accuracy Alone Isn't Enough?**\n\n"
        "The stroke dataset is highly imbalanced with many more "
        "non-stroke cases than stroke cases. Therefore accuracy alone "
        "can be misleading. For stroke screening identifying actual "
        "stroke cases is especially important, so Recall is an important "
        "metric when evaluating the models."
    )
# ===================================
# WHY BALANCED LOGISTIC REGRESSION
# ===================================
    st.success(
        "**Why Balanced Logistic Regression?**\n\n"
        "Standard Logistic Regression achieved high accuracy but detected "
        "only a small proportion of actual stroke cases. By using class "
        "balancing, Balanced Logistic Regression improved Recall to "
        "**80%**, making it more suitable for a stroke screening task "
        "where detecting potential stroke cases is important."
    )
# =========================
# WHY HYPERPARAMETER TUNING
# =========================

    st.warning(
        "**Why Hyperparameter Tuning?**\n\n"
        "The balanced Logistic Regression model was further tuned to "
        "find better model settings. Hyperparameter tuning helps improve "
        "the model's performance while maintaining its focus on detecting "
        "the minority stroke class."
    )
    # =========================
# RECALL COMPARISON
# =========================

    st.subheader("Recall Comparison")

    recall_models = [
        "Logistic Regression",
        "KNN",
        "Decision Tree",
        "Random Forest",
        "SMOTE + Logistic Regression",
        "Balanced Logistic Regression"
    ]
    recall_values = [
        2.00,
        2.00,
        20.00,
        2.00,
        38.00,
        80.00
    ]
    fig, ax = plt.subplots(
        figsize=(6, 3.2),
        dpi=160
    )
    # ax.barh(
    #     recall_models,
    #     recall_values,
    #     color="#2F6FA3"
    # )
    bars = ax.barh(
    recall_models,
    recall_values,
    color="#2F6FA3"
    )
    # Display exact recall values
    for bar, value in zip(bars, recall_values):
        ax.text(
            value + 1,
            bar.get_y() + bar.get_height() / 2,
            f"{value:.0f}%",
            va="center",
            fontsize=8,
            fontweight="bold"
        )
    ax.set_xlabel(
        "Recall (%)",
        fontsize=9
    )
    ax.set_ylabel(
        "Model",
        fontsize=9
    )
    ax.tick_params(
        axis="both",
        labelsize=7
    )
    ax.set_xlim(0, 100)
    plt.tight_layout()
    st.pyplot(
        fig,
        use_container_width=False
    )
    plt.close(fig)
    st.markdown(
        "Balanced Logistic Regression achieves the highest recall (80%), "
        "meaning it identifies more actual stroke cases than the other evaluated models."
    )   

# =========================
# FINAL MODEL SELECTION
# =========================

    st.success(
        "**Final Model Selection**\n\n"
        "Balanced Logistic Regression was selected as the final model "
        "because it achieved the highest recall (80%) among the evaluated models. "
        "Although some models achieved higher accuracy, they detected very few "
        "actual stroke cases. Since the primary objective of this project is "
        "stroke-risk screening, identifying potential stroke cases is more "
        "important than maximizing accuracy alone."
    )

# =========================
# CONFUSION MATRIX
# =========================
    st.subheader("Confusion Matrix")
    cm = confusion_matrix(y_test, y_pred_analytics)
    # st.write("The confusion matrix shows how the final model's predictions compare with the actual outcomes.")
    # st.dataframe(
    #     {
    #         "Actual No Stroke": [cm[0][0], cm[1][0]],
    #         "Actual Stroke": [cm[0][1], cm[1][1]]
    #     },
    #     use_container_width=True 
    # ) 
    st.write(
    "The confusion matrix shows how the final model's predictions "
    "compare with the actual outcomes."
    )


# =========================
# CONFUSION MATRIX HEATMAP
# =========================

    blue_cmap = LinearSegmentedColormap.from_list(
        "MedInsightBlues",
        [
            "#DCEBF7",   # light blue
            "#8FC4E8",   # soft blue
            "#4A90C2",   # medium blue
            "#164A7A"    # dark navy
        ]
    )
    fig, ax = plt.subplots(figsize=(4.2, 3.0), dpi=160)
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap=blue_cmap,

        # Makes lower values like 10 and 40 more visually distinct
        norm=PowerNorm(gamma=0.35, vmin=0, vmax=cm.max()),
        xticklabels=["No Stroke", "Stroke"],
        yticklabels=["No Stroke", "Stroke"],
        cbar=True,
        linewidths=1,
        linecolor="white",
        annot_kws={
            "size": 11,
            "weight": "bold"
        },
        ax=ax
    )

# =========================
# ANNOTATION COLOURS
# =========================

    for text in ax.texts:
        value = int(text.get_text())
        if value >= 200:
            text.set_color("white")
        else:
            text.set_color("#123B5D")

# =========================
# AXIS LABELS
# =========================
    ax.set_xlabel(
        "Predicted Outcome",
        fontsize=9
    )
    ax.set_ylabel(
        "Actual Outcome",
        fontsize=9
    )
    ax.tick_params(
        axis="both",
        labelsize=8,
        length=0
    )
    # =========================
    # COLORBAR
    # =========================

    cbar = ax.collections[0].colorbar
    cbar.ax.tick_params(
        labelsize=7,
        length=3
    )
    plt.tight_layout()
    st.pyplot(
        fig,
        use_container_width=False
    )
    plt.close(fig)
# =========================
# CONFUSION MATRIX INTERPRETATION
# =========================

    st.info(
        "**Confusion Matrix Interpretation**\n\n"
        "The model correctly identified 723 non-stroke cases and 40 stroke cases. "
        "It missed 10 actual stroke cases and incorrectly predicted stroke for "
        "249 non-stroke cases. This reflects the model's high recall, while also "
        "showing the trade-off with precision."
    )

    #SHAP learns how the model's features affect predictions
    # explainer = shap.LinearExplainer(
    #     best_balanced_lr ,
    #     X_train
    # )
    shap_values = explainer(X_test)
    # Select one patient for explanation
    patient_index = 0
    patient_shap = shap_values[patient_index]

    # =========================
    # INDIVIDUAL PATIENT SHAP
    # =========================

    # patient_shap = shap_values[patient_index]
    # Presentation-friendly labels for waterfall chart
    display_names = {
        "gender": "Gender",
        "age": "Age",
        "hypertension": "Hypertension",
        "heart_disease": "Heart Disease",
        "ever_married": "Ever Married",
        "Residence_type": "Residence Type",
        "avg_glucose_level": "Avg Glucose Level",
        "bmi": "BMI",
        "work_type_Govt_job": "Work Type: Govt Job",
        "work_type_Never_worked": "Work Type: Never Worked",
        "work_type_Private": "Work Type: Private",
        "work_type_Self-employed": "Work Type: Self-Employed",
        "work_type_children": "Work Type: Children",
        "smoking_status_Unknown": "Smoking Status: Unknown",
        "smoking_status_formerly smoked": "Smoking Status: Formerly Smoked",
        "smoking_status_never smoked": "Smoking Status: Never Smoked",
        "smoking_status_smokes": "Smoking Status: Smokes"
    }
    waterfall_shap = shap.Explanation(
        values=patient_shap.values,
        base_values=patient_shap.base_values,
        data=patient_shap.data,
        feature_names=[
            display_names.get(name, name)
            for name in patient_shap.feature_names
        ]
    )
    st.subheader("Example Patient SHAP Explanation")
    fig = plt.figure(figsize=(5, 3), dpi=160)
    shap.plots.waterfall(
        waterfall_shap,
        max_display=8,
        show=False
    )
    plt.tight_layout()
    st.pyplot(
        fig,
        use_container_width=False
    )
    plt.close(fig)

    feature_importance = np.abs(
        shap_values.values
    ).mean(axis=0)
    feature_names = X_test.columns
    importance_df = pd.DataFrame({
        "Feature": feature_names,
        "Importance": feature_importance
    })
    
    importance_df = importance_df.sort_values(
        "Importance",
        ascending=True
    )
    # Make feature names presentation-friendly
    # importance_df["Feature"] = (
    #     importance_df["Feature"]
    #     .str.replace("_", " ", regex=False)
    #     .str.title()
    # )
    # Create presentation-friendly feature names
    importance_df["Display Feature"] = (
        importance_df["Feature"]
        .str.replace("_", " ", regex=False)
        .str.title()
    )
    st.subheader("SHAP Feature Importance")
    fig, ax = plt.subplots(
        figsize=(6, 4),
        dpi=160
    )
    ax.barh(
        importance_df["Display Feature"],
        importance_df["Importance"],
        color="#2F6FA3"
    )
    ax.set_xlabel(
        "Mean Absolute SHAP Value",
        fontsize=9
    )
    ax.set_ylabel(
        "Feature",
        fontsize=9
    )
    ax.tick_params(
        axis="both",
        labelsize=7
    )
    plt.tight_layout()
    st.pyplot(
        fig,
        use_container_width=False
    )
    plt.close(fig)
    #organize → sort → visualize → display


    st.stop()
# best_balanced_lr = the model we want to explain
# X_train = the reference data SHAP uses to explain it


#PDF Create

class ECGLine(Flowable):
    def __init__(self, width=470, height=25):
        Flowable.__init__(self)
        self.width = width
        self.height = height
        self.hAlign = "CENTER"
    def draw(self):
        canvas = self.canv
        canvas.setStrokeColor(colors.HexColor("#0D47A1"))
        canvas.setLineWidth(1)
        path = canvas.beginPath()
        y = self.height / 2

        # Start of ECG line
        path.moveTo(0, y)

        # Flat line
        path.lineTo(175, y)

        # ECG waveform
        path.lineTo(225, y)
        path.lineTo(230, y + 4)
        path.lineTo(235, y - 3)
        path.lineTo(240, y + 16)   # sharp peak
        path.lineTo(245, y - 10)   # sharp drop
        path.lineTo(250, y + 3)
        path.lineTo(255, y)

        # Continue flat line
        path.lineTo(self.width, y)
        canvas.drawPath(path)

class PersonIcon(Flowable):

    def __init__(self, size=22):
        Flowable.__init__(self)
        self.width = size
        self.height = size
        self.size = size

    def draw(self):
        canvas = self.canv

        canvas.setStrokeColor(colors.HexColor("#1E4FA3"))
        canvas.setFillColor(colors.HexColor("#1E4FA3"))

        # Head
        canvas.circle(
            self.size / 2,
            self.size * 0.68,
            self.size * 0.16,
            fill=1,
            stroke=0
        )

        # Shoulders/body
        canvas.roundRect(
            self.size * 0.20,
            self.size * 0.15,
            self.size * 0.60,
            self.size * 0.35,
            self.size * 0.15,
            fill=1,
            stroke=0
        )

class HealthIcon(Flowable):
    def __init__(self, size=22):
        Flowable.__init__(self)
        self.width = size
        self.height = size
        self.size = size
    def draw(self):
        canvas = self.canv

        canvas.setStrokeColor(colors.HexColor("#238B57"))
        canvas.setLineWidth(2)

        # Heart outline
        path = canvas.beginPath()
        s = self.size
        path.moveTo(s * 0.50, s * 0.15)
        path.curveTo(
            s * 0.10, s * 0.45,
            s * 0.05, s * 0.75,
            s * 0.50, s * 0.90
        )
        path.curveTo(
            s * 0.95, s * 0.75,
            s * 0.90, s * 0.45,
            s * 0.50, s * 0.15
        )
        canvas.drawPath(path)
        # ECG line inside heart
        ecg = canvas.beginPath()
        ecg.moveTo(s * 0.20, s * 0.53)
        ecg.lineTo(s * 0.37, s * 0.53)
        ecg.lineTo(s * 0.43, s * 0.40)
        ecg.lineTo(s * 0.50, s * 0.67)
        ecg.lineTo(s * 0.57, s * 0.47)
        ecg.lineTo(s * 0.64, s * 0.53)
        ecg.lineTo(s * 0.80, s * 0.53)
        canvas.drawPath(ecg)

class LifestyleIcon(Flowable):

    def __init__(self, size=22):
        Flowable.__init__(self)
        self.width = size
        self.height = size
        self.size = size
    def draw(self):
        canvas = self.canv
        s = self.size
        canvas.setFillColor(colors.HexColor("#6A1B9A"))
        # Three people
        canvas.circle(s * 0.30, s * 0.68, s * 0.12, fill=1, stroke=0)
        canvas.circle(s * 0.70, s * 0.68, s * 0.12, fill=1, stroke=0)
        canvas.circle(s * 0.50, s * 0.78, s * 0.13, fill=1, stroke=0)
        canvas.roundRect(
            s * 0.10, s * 0.20,
            s * 0.35, s * 0.30,
            s * 0.10,
            fill=1, stroke=0
        )
        canvas.roundRect(
            s * 0.55, s * 0.20,
            s * 0.35, s * 0.30,
            s * 0.10,
            fill=1, stroke=0
        )



class RiskShieldIcon(Flowable):
    def __init__(self, size=22):
        Flowable.__init__(self)
        self.width = size
        self.height = size
        self.size = size
    def draw(self):
        canvas = self.canv
        s = self.size
        canvas.setFillColor(colors.HexColor("#123F7A"))
        path = canvas.beginPath()
        path.moveTo(s * 0.50, s * 0.92)
        path.lineTo(s * 0.85, s * 0.78)
        path.lineTo(s * 0.80, s * 0.42)
        path.curveTo(
            s * 0.75, s * 0.22,
            s * 0.60, s * 0.10,
            s * 0.50, s * 0.05
        )
        path.curveTo(
            s * 0.40, s * 0.10,
            s * 0.25, s * 0.22,
            s * 0.20, s * 0.42
        )
        path.lineTo(s * 0.15, s * 0.78)
        path.close()

        canvas.drawPath(path, fill=1, stroke=0)

        # Small white check mark
        canvas.setStrokeColor(colors.white)
        canvas.setLineWidth(1.5)

        check = canvas.beginPath()
        check.moveTo(s * 0.34, s * 0.48)
        check.lineTo(s * 0.46, s * 0.35)
        check.lineTo(s * 0.68, s * 0.62)
        canvas.drawPath(check)

class SHAPChart(Flowable):

    def __init__(self, contribution_data, width=230, height=150):
        Flowable.__init__(self)
        self.width = width
        self.height = height
        self.data = contribution_data[:5]

    def draw(self):

        canvas = self.canv

        # Maximum absolute SHAP value
        max_value = max(
            abs(item["SHAP Contribution"])
            for item in self.data
        )

        # Chart area
        # chart_left = 80
        # chart_right = 220
        label_x = 95
        zero_x = 145
        chart_top = self.height - 15

        canvas.setStrokeColor(
            colors.HexColor("#999999")
        )
        canvas.setLineWidth(0.7)
        # Zero line
        canvas.line(
            zero_x,
            10,
            zero_x,
            chart_top
        )

        # Draw each factor
        for i, item in enumerate(self.data):

            factor = item["Factor"]
            value = item["SHAP Contribution"]
            y = chart_top - (i * 25)

            # Short display name
            # if len(factor) > 21:
            #     factor = factor[:21] + "..."
            if factor == "Smoking Status: Never Smoked":
                factor = "Smoking: Never Smoked"
            elif len(factor) > 21:
                factor = factor[:21] + "..."

            # canvas.setFont("Helvetica", 7)
            # Feature label
            canvas.setFont("Helvetica", 6.5)
            canvas.setFillColor(
                colors.HexColor("#333333")
            )
            canvas.drawRightString(
            label_x,
            y - 3,
            factor
            )

            # Bar length
            bar_length = (
                abs(value) / max_value
            ) * 40

            if value >= 0:
                    canvas.setFillColor(
                        colors.HexColor("#C62828")
                    )

                    canvas.rect(
                        zero_x,
                        y - 5,
                        bar_length,
                        9,
                        fill=1,
                        stroke=0
                    )

                    canvas.setFont("Helvetica", 6)
                    canvas.setFillColor(
                        colors.HexColor("#444444")
                    )

                    canvas.drawString(
                        zero_x + bar_length + 3,
                        y - 3,
                        f"{value:.3f}"
                    )

            else:
                    canvas.setFillColor(
                        colors.HexColor("#1765B0")
                    )

                    canvas.rect(
                        zero_x - bar_length,
                        y - 5,
                        bar_length,
                        9,
                        fill=1,
                        stroke=0
                    )

                    canvas.setFont("Helvetica", 6)
                    canvas.setFillColor(
                        colors.HexColor("#444444")
                    )

                    canvas.drawRightString(
                        zero_x - bar_length - 5,
                        y - 3,
                        f"{value:.3f}"
                    )

def create_pdf_report(
    age,
    gender,
    ever_married,
    Residence_type,
    hypertension,
    heart_disease,
    avg_glucose_level,
    bmi,
    work_type,
    smoking_status,
    probability,
    prediction,
    guidance
):
    buffer = BytesIO()

    doc = SimpleDocTemplate(
        buffer,
        pagesize=A4,
        rightMargin=40,
        leftMargin=40,
        topMargin=40,
        bottomMargin=40
    )

    styles = getSampleStyleSheet()

    title_style = ParagraphStyle(
        "TitleStyle",
        parent=styles["Title"],
        alignment=TA_CENTER,
        fontSize=24,
        textColor = colors.HexColor("#042962"),
        spaceAfter = 5,
        fontName = "Helvetica-Bold"
    )
    subtitle_style = ParagraphStyle(
        "SubtitleStyle",
        parent=styles["Normal"],
        alignment=TA_CENTER,
        fontSize=15,
        textColor = colors.HexColor("#042962"),
        spaceAfter=18,
        fontName="Helvetica-Bold"
    )
    heading_style = ParagraphStyle(
        "HeadingStyle",
        parent=styles["Heading2"],
        fontSize=14,
        spaceBefore=12,
        spaceAfter=8
    )
    normal_style = ParagraphStyle(
        "NormalStyle",
        parent=styles["Normal"],
        fontSize=10,
        leading=15
    )

    story = []
    # Header
    story.append(
        Paragraph("MEDINSIGHT", title_style)
    )
    story.append(
        Paragraph(
            "Stroke Risk Screening & Explainable ML Report",
            subtitle_style
        )
    )
    story.append(ECGLine())
    story.append(Spacer(1, 12))
    
    
    patient_table = Table([
        ["Age", str(age)],
        ["Gender", gender],
        ["Ever Married", ever_married],
        ["Residence Type", Residence_type]
    ], colWidths=[105, 120])

    patient_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (0, -1), colors.HexColor("#EEF5FF")),
        ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#C7D7EA")),
        # ("FONTNAME", (0, 0), (-1, -1), "Helvetica"),
        ("FONTNAME", (0, 0), (0, -1), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 8),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING", (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
    ]))
    patient_heading = Table(
    [[PersonIcon(22), Paragraph("PATIENT INFORMATION", heading_style)]],colWidths=[30, 210]
    )
    patient_section = Table(
    [[patient_heading],
     [patient_table]],
    colWidths=[250]
    )


    health_table = Table([
        ["Hypertension", hypertension],
        ["Heart Disease", heart_disease],
        ["Average Glucose Level", str(avg_glucose_level)],
        ["BMI", str(bmi)]
    ], colWidths=[125, 100])

    health_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (0, -1), colors.HexColor("#F0F8F1")),
        ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#C8DCCB")),
        ("FONTNAME", (0, 0), (0, -1), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 8),
        ("TOPPADDING", (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
    ]))
    
    health_heading = Table(
    [[HealthIcon(22), Paragraph("HEALTH FACTORS", heading_style)]], colWidths=[30, 210])

    health_section = Table(
        [[health_heading],
        [health_table]],
        colWidths=[250]
    )
    combined_sections = Table(
        [[patient_section, health_section]],
        colWidths=[250, 250]
    )
    combined_sections.setStyle(TableStyle([
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING", (0, 0), (-1, -1), 0),
        ("RIGHTPADDING", (0, 0), (-1, -1), 8),
        ("TOPPADDING", (0, 0), (-1, -1), 0),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 0),
    ]))
    story.append(combined_sections)
    story.append(Spacer(1, 12))


    # Lifestyle Factors
    # story.append(
    #         Paragraph("Lifestyle & Social Factors", heading_style)
    #     )
    
    lifestyle_heading = Table(
        [[LifestyleIcon(22),
          Paragraph("LIFESTYLE & SOCIAL FACTORS", heading_style)]], colWidths = [30,250]
    )
    story.append(lifestyle_heading)
    work_type_display = work_type.replace("_", " ").title()
    smoking_status_display = smoking_status.replace("_", " ").title()

    lifestyle_table = Table([
        ["Work Type", work_type_display],
        ["Smoking Status", smoking_status_display]
    ], colWidths=[170, 300])

    lifestyle_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (0, -1), colors.HexColor("#F3E8F8")),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#D8C5E5")),
        ("FONTNAME", (0, 0), (0, -1), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 10),
        ("TOPPADDING", (0, 0), (-1, -1), 7),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 7),
    ]))
    story.append(lifestyle_table)
    story.append(Spacer(1, 18))

    # =========================
    # RISK ASSESSMENT
    # =========================

    risk_heading = Table(
        [[
            RiskShieldIcon(22),
            Paragraph("RISK ASSESSMENT", heading_style)
        ]],
        colWidths=[30, 180]
    )

    risk_heading.setStyle(TableStyle([
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("LEFTPADDING", (0, 0), (-1, -1), 0),
        ("RIGHTPADDING", (0, 0), (-1, -1), 4),
        ("TOPPADDING", (0, 0), (-1, -1), 0),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
    ]))
    story.append(risk_heading)

    # risk_result = (
    #     "Higher Predicted Risk"
    #     if prediction == 1
    #     else "Lower Predicted Risk"
    # )

    # risk_color = (
    # "#C62828"
    # if prediction == 1
    # else "#247A45"
    # )
    # risk_explanation = (
    #     "Based on the patient's information, "
    #     "the machine learning model predicts a higher "
    #     "likelihood of stroke."
    #     if prediction == 1
    #     else
    #     "Based on the patient's information, "
    #     "the machine learning model predicts a lower "
    #     "likelihood of stroke."
    # )
    # ======================================================
# PDF RISK CLASSIFICATION
# ======================================================

    if probability >= 0.50:

        risk_result = "Higher Predicted Risk"
        risk_color = "#C62828"

        risk_explanation = (
            "Based on the patient's information, "
            "the machine learning model predicts a higher "
            "likelihood of stroke."
        )

    elif probability >= 0.20:

        risk_result = "Moderate Predicted Risk"
        risk_color = "#B8860B"

        risk_explanation = (
            "Based on the patient's information, "
            "the machine learning model predicts a moderate "
            "likelihood of stroke."
        )

    else:

        risk_result = "Lower Predicted Risk"
        risk_color = "#247A45"

        risk_explanation = (
            "Based on the patient's information, "
            "the machine learning model predicts a lower "
            "likelihood of stroke."
    )

    # LEFT SIDE
    probability_label = Paragraph(
        "<para align='center'>"
        "<font size='12'><b>ESTIMATED STROKE PROBABILITY</b></font>"
        # "<br/>"
        # "<font size='23' color='#1765B0'><b>"
        # f"{probability * 100:.2f}%"
        # "</b></font>"
        "</para>",
        normal_style
    )
    probability_value = Paragraph(
        "<para align='center'>"
        f"<font size='25' color='#1765B0'><b>"
        f"{probability * 100:.2f}%"
        f"</b></font>"
        "</para>",
        normal_style
    )
    left_risk = Table(
        [
            [probability_label],
            [probability_value],
        ],
        colWidths=[250],
        rowHeights=[35,65]
    )   
    left_risk.setStyle(TableStyle([
        ("ALIGN", (0, 0), (-1, -1), "CENTER"),

        ("VALIGN", (0, 0), (-1, 0), "TOP"),
        ("VALIGN", (0, 1), (-1, 1), "MIDDLE"),

        ("LEFTPADDING", (0, 0), (-1, -1), 8),
        ("RIGHTPADDING", (0, 0), (-1, -1), 8),

        ("TOPPADDING", (0, 0), (-1, 0), 8),
        ("BOTTOMPADDING", (0, 0), (-1, 0), 0),

        ("TOPPADDING", (0, 1), (-1, 1), 0),
        ("BOTTOMPADDING", (0, 1), (-1, 1), 8),
    ]))

    # RIGHT SIDE

    model_label = Paragraph(
        "<para align='center'>"
        "<font size='12'><b>MODEL PREDICTION</b></font>"
        "</para>",
        normal_style
    )


    prediction_value = Paragraph(
        "<para align='center'>"
        f"<font size='16' color='{risk_color}'><b>"
        f"{risk_result}"
        f"</b></font>"
        "</para>",
        normal_style
    )

    prediction_description = Paragraph(
        "<para align='center'>"
        f"<font size='8.5'>{risk_explanation}</font>"
        "</para>",
        normal_style
    )

    prediction_text = Table(
    [
        [model_label],
        [prediction_value],
        [prediction_description]
    ],
    colWidths=[250],
    rowHeights=[25, 35, 45]
)

    prediction_text.setStyle(TableStyle([
        ("ALIGN", (0, 0), (-1, -1), "CENTER"),

        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),

        ("LEFTPADDING", (0, 0), (-1, -1), 10),
        ("RIGHTPADDING", (0, 0), (-1, -1), 10),

        ("TOPPADDING", (0, 0), (-1, -1), 0),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 0),
    ]))

    # FINAL RISK CARD
    risk_table = Table(
        [[
            left_risk,
            prediction_text
        ]],
        colWidths=[250, 250],
        rowHeights=[105]
    )
    risk_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1),
        colors.HexColor("#F3F8FF")),

        ("BOX", (0, 0), (-1, -1),
        0.8, colors.HexColor("#B8D3F2")),

        ("LINEAFTER", (0, 0), (0, 0),
        0.6, colors.HexColor("#D5E2F2")),

        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),

        ("LEFTPADDING", (0, 0), (-1, -1), 8),
        ("RIGHTPADDING", (0, 0), (-1, -1), 8),

        ("TOPPADDING", (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
    ]))
    story.append(risk_table)
    story.append(Spacer(1, 18)) 

# ===================================
# TOP 5 CONTRIBUTING FACTORS  + SHAP
# ===================================

    factors_heading = Paragraph(
        "<font color='#F57C00'><b>★</b></font> "
        "<font color='#F57C00'><b>TOP 5 CONTRIBUTING FACTORS</b></font>",
        normal_style
    )
    shap_heading = Paragraph(
        "<font color='#174A8B'><b>▥</b></font> "
        "<font color='#174A8B'><b>SHAP CONTRIBUTION (IMPACT ON RISK)</b></font>",
        normal_style
    )  
    #2 headings side by side
    two_headings = Table(
        [[
            factors_heading,
            shap_heading
        ]],
        colWidths=[250, 250]
    )

    two_headings.setStyle(TableStyle([
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),

        ("LEFTPADDING", (0, 0), (-1, -1), 5),
        ("RIGHTPADDING", (0, 0), (-1, -1), 5),

        ("TOPPADDING", (0, 0), (-1, -1), 0),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
    ]))

    factor_rows = []

    for i, item in enumerate(contribution_data[:5], start=1):

        # feature = item["Feature"]
        # shap_value = item["SHAP Value"]
        feature = item["Factor"]
        shap_value = item["SHAP Contribution"]

        if shap_value >= 0:
            value_color = "#C62828"
            sign = "+"
        else:
            value_color = "#1765B0"
            sign = ""

        factor_rows.append([
            Paragraph(
                f"<b>{i}</b>",
                normal_style
            ),
            Paragraph(
                f"<b>{feature}</b>",
                normal_style
            ),
            Paragraph(
                f"<font color='{value_color}'><b>"
                f"{sign}{shap_value:.3f}"
                f"</b></font>",
                normal_style
            )
        ])

    factors_table = Table(
        factor_rows,
        colWidths=[25, 155, 55]
    )

    factors_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1),
        colors.HexColor("#FFF8F0")),

        ("BOX", (0, 0), (-1, -1),
        0.6, colors.HexColor("#F4D6B5")),

        ("GRID", (0, 0), (-1, -1),
        0.3, colors.HexColor("#F1E1D0")),

        ("ALIGN", (0, 0), (0, -1), "CENTER"),
        ("ALIGN", (2, 0), (2, -1), "RIGHT"),

        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),

        ("FONTSIZE", (0, 0), (-1, -1), 7.5),

        ("TOPPADDING", (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),

        ("LEFTPADDING", (0, 0), (-1, -1), 4),
        ("RIGHTPADDING", (0, 0), (-1, -1), 4),
    ]))
    shap_chart = SHAPChart(
    contribution_data,
    width=230,
    height=140
    )
    shap_box = Table(
    [[shap_chart]],
    colWidths=[250],
    rowHeights=[150]
    )

    shap_box.setStyle(TableStyle([
        ("BOX", (0, 0), (-1, -1), 0.6, colors.HexColor("#D8E2EF")),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("LEFTPADDING", (0, 0), (-1, -1), 5),
        ("RIGHTPADDING", (0, 0), (-1, -1), 5),
        ("TOPPADDING", (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
    ]))
    
# =========================
# FACTORS + SHAP SIDE BY SIDE
# =========================
    factors_and_shap = Table(
        [[
            factors_table,
            shap_box
        ]],
        colWidths=[250, 250]
    )
    factors_and_shap.setStyle(TableStyle([
        ("VALIGN", (0, 0), (-1, -1), "TOP"),

        ("LEFTPADDING", (0, 0), (-1, -1), 5),
        ("RIGHTPADDING", (0, 0), (-1, -1), 5),

        ("TOPPADDING", (0, 0), (-1, -1), 0),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 0),
    ]))
    story.append(two_headings)
    story.append(factors_and_shap)
    story.append(Spacer(1, 15))
    
# =========================
# DETAILED SHAP CONTRIBUTION
# =========================

    detailed_heading = Paragraph(
        "<font color='#174A8B'><b>▣ DETAILED SHAP CONTRIBUTION</b></font>",
        normal_style
    )

    # story.append(detailed_heading)
    # story.append(Spacer(1, 6))

    detailed_rows = [
        [
            Paragraph("<b>Feature</b>", normal_style),
            Paragraph("<b>Patient Value</b>", normal_style),
            Paragraph("<b>SHAP Value</b>", normal_style),
            Paragraph("<b>Impact on Risk</b>", normal_style)
        ]
    ]

    for item in contribution_data:

        shap_value = item["SHAP Contribution"]

        if shap_value >= 0:
            impact = "Increases Risk"
            impact_color = "#C62828"
            sign = "+"
        else:
            impact = "Decreases Risk"
            impact_color = "#1765B0"
            sign = ""

        detailed_rows.append([
            Paragraph(
                f"<b>{item['Factor']}</b>",
                normal_style
            ),

            Paragraph(
                str(item["Patient Value"]),
                normal_style
            ),

            Paragraph(
                f"<font color='{impact_color}'><b>"
                f"{sign}{shap_value:.3f}"
                f"</b></font>",
                normal_style
            ),

            Paragraph(
                f"<font color='{impact_color}'><b>"
                f"{impact}"
                f"</b></font>",
                normal_style
            )
        ])

    detailed_table = Table(
        detailed_rows,
        colWidths=[150, 100, 80, 120],
        repeatRows=1
    )

    detailed_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0),
        colors.HexColor("#6EB5FC")),

        ("TEXTCOLOR", (0, 0), (-1, 0),
        colors.black),

        ("GRID", (0, 0), (-1, -1),
        0.4, colors.HexColor("#D5DDE8")),

        ("BACKGROUND", (0, 1), (-1, -1),
        colors.white),

        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),

        ("FONTSIZE", (0, 0), (-1, -1), 8),

        ("TOPPADDING", (0, 0), (-1, -1), 6),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 6),

        ("LEFTPADDING", (0, 0), (-1, -1), 5),
        ("RIGHTPADDING", (0, 0), (-1, -1), 5),
    ]))

    # story.append(detailed_table)
    # story.append(Spacer(1, 15))
    detailed_section = KeepTogether([
        detailed_heading,
        Spacer(1, 6),
        detailed_table,
        Spacer(1, 15)
    ])
    story.append(detailed_section)
    # shap_chart = Drawing(230, 150)


    # =========================================
# PERSONALIZED HEALTH GUIDANCE
# =========================================

    # ---------- Premium color palette ----------
    guide_blue = colors.HexColor("#174A8B")
    guide_blue_light = colors.HexColor("#EAF2FB")
    guide_teal = colors.HexColor("#1599A8")
    guide_dark = colors.HexColor("#263238")
    guide_grey = colors.HexColor("#68727D")

    # ---------- Risk-based styling ----------
    if probability >= 0.50:

        guidance_intro = (
            "<b>Higher estimated stroke risk:</b> "
            "The model indicates a higher estimated risk. "
            "Pay particular attention to modifiable risk factors "
            "and arrange appropriate medical follow-up."
        )

        risk_bg = colors.HexColor("#FDECEC")
        risk_border = colors.HexColor("#C62828")
        risk_title = colors.HexColor("#C62828")

    elif probability >= 0.20:

        guidance_intro = (
            "<b>Moderate estimated stroke risk:</b> "
            "The model indicates an intermediate level of estimated risk. "
            "Pay particular attention to modifiable risk factors "
            "and maintain regular health checkups."
        )

        risk_bg = colors.HexColor("#FFF7E6")
        risk_border = colors.HexColor("#D39A00")
        risk_title = colors.HexColor("#B47D00")

    else:

        guidance_intro = (
            "<b>Lower estimated stroke risk:</b> "
            "The model indicates a lower estimated risk based on the "
            "information provided. Continue maintaining healthy habits "
            "and regular preventive checkups."
        )

        risk_bg = colors.HexColor("#EAF7F0")
        risk_border = colors.HexColor("#247A45")
        risk_title = colors.HexColor("#247A45")


    # =========================================
    # HEADING
    # =========================================

    guidance_heading = Paragraph(
        f"""
        <font color="#174A8B"><b>PERSONALIZED HEALTH GUIDANCE</b></font>
        """,
        ParagraphStyle(
            "GuidanceHeading",
            parent=normal_style,
            fontSize=15,
            leading=19,
            spaceAfter=5
        )
    )

    # Small premium teal accent
    guidance_accent = Table(
        [[""]],
        colWidths=[75],
        rowHeights=[3]
    )

    guidance_accent.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), guide_teal),
        ("BOX", (0, 0), (-1, -1), 0, guide_teal),
    ]))


    # =========================================
    # RISK SUMMARY CARD
    # =========================================

    risk_para = Paragraph(
        guidance_intro,
        ParagraphStyle(
            "GuidanceIntro",
            parent=normal_style,
            fontSize=9.5,
            leading=14,
            textColor=guide_dark
        )
    )

    risk_card = Table(
        [[risk_para]],
        colWidths=[470]
    )

    risk_card.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), risk_bg),

        ("BOX", (0, 0), (-1, -1), 0.8, risk_border),

        ("LINEBEFORE", (0, 0), (0, -1), 4, risk_title),

        ("LEFTPADDING", (0, 0), (-1, -1), 12),
        ("RIGHTPADDING", (0, 0), (-1, -1), 12),
        ("TOPPADDING", (0, 0), (-1, -1), 9),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 9),

        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
    ]))


    # =========================================
    # GUIDANCE ITEMS
    # =========================================

    guidance_rows = []

    for item in guidance:

        # Remove emojis for reliable PDF rendering
        clean_item = (
            item
            .replace("❤️", "")
            .replace("🫀", "")
            .replace("🩸", "")
            .replace("⚖️", "")
            .replace("🚭", "")
            .replace("🏃", "")
            .replace("🩺", "")
            .replace("💚", "")
            .replace("⚠️", "")
            .replace("🎯", "")
            .strip()
        )

        # Remove markdown bold markers
        clean_item = clean_item.replace("**", "")

        if ":" in clean_item:

            heading, description = clean_item.split(":", 1)

            bullet_text = (
                f'<font color="#174A8B"><b>{heading.strip()}</b></font>'
                f'<font color="#263238">: {description.strip()}</font>'
            )

        else:

            bullet_text = (
                f'<font color="#263238">{clean_item}</font>'
            )
        bullet = Paragraph(
            f'<font color="#1599A8"><b>●</b></font>&nbsp;&nbsp;'
            f'{bullet_text}',
            ParagraphStyle(
                "GuidanceBullet",
                parent=normal_style,
                fontSize=9,
                leading=13.5,
                textColor=guide_dark,
            )
        )
        guidance_rows.append([bullet])
    guidance_table = Table(
        guidance_rows,
        colWidths=[470]
    )
    guidance_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1),
         colors.HexColor("#F8FAFD")),

        ("BOX", (0, 0), (-1, -1),
         0.6, colors.HexColor("#D7E1EC")),

        ("INNERGRID", (0, 0), (-1, -1),
         0.35, colors.HexColor("#E7EDF4")),

        ("LEFTPADDING", (0, 0), (-1, -1), 11),
        ("RIGHTPADDING", (0, 0), (-1, -1), 11),
        ("TOPPADDING", (0, 0), (-1, -1), 7),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 7),

        ("VALIGN", (0, 0), (-1, -1), "TOP"),
    ]))
    # =========================================
    # FOLLOW-UP NOTE
    # =========================================

    # followup_text = (
    #     "Because the estimated risk is not low, arrange consistent health "
    #     "checkups and review important risk factors such as blood pressure, "
    #     "glucose, heart health, weight, and smoking status with your "
    #     "healthcare professional."
    #     if probability >= 0.20
    #     else
    #     "Continue routine preventive health checkups and monitor important "
    #     "factors such as blood pressure, glucose, weight, and heart health."
    # )

    # followup = Paragraph(
    #     f"""
    #     <font color="#1599A8"><b>Regular Follow-up:</b></font>
    #     <font color="#263238"> {followup_text}</font>
    #     """,
    #     ParagraphStyle(
    #         "Followup",
    #         parent=normal_style,
    #         fontSize=9,
    #         leading=13,
    #         leftIndent=3,
    #         spaceBefore=7
    #     )
    # )
    # =========================================
    # PROFESSIONAL NOTE
    # =========================================

    guidance_note = Paragraph(
        """
        <font color="#68727D">
        This guidance is based on the information entered into the screening
        system and is intended for general risk-factor awareness. It does not
        constitute a medical diagnosis or replace advice from a qualified
        healthcare professional.
        </font>
        """,
        ParagraphStyle(
            "GuidanceNote",
            parent=normal_style,
            fontSize=7.5,
            leading=10.5,
            spaceBefore=7
        )
    )
    # =========================================
    # FINAL GUIDANCE SECTION
    # =========================================

    guidance_section = KeepTogether([
        guidance_heading,
        guidance_accent,
        Spacer(1, 8),
        risk_card,
        Spacer(1, 9),
        guidance_table,
        guidance_note,

        Spacer(1, 15)
    ])
    story.append(guidance_section)
    story.append(
                Paragraph(
                    "Disclaimer : This report is generated by MedInsight for screening "
                    "and decision-support purposes.",
                    normal_style
                )
            )
    doc.build(story)
    buffer.seek(0)
    return buffer

    
st.markdown(
    """<div style="text-align:center;margin:8px 0 34px 0;"><div style="font-size:50px;font-weight:750;letter-spacing:-1.5px;line-height:1.1;background:linear-gradient(90deg,#1557A6 0%,#1769B0 30%,#1599A8 65%,#12B8B0 100%);-webkit-background-clip:text;-webkit-text-fill-color:transparent;">MedInsight</div><div style="display:flex;align-items:center;justify-content:center;margin-top:6px;"><div style="width:125px;height:2px;background:linear-gradient(90deg,#1557A6, 50%, #12B8B0 100%);"></div><svg width="55" height="22" viewBox="0 0 55 22" style="margin:0;"><polyline points="0,11 15,11 20,11 24,3 28,19 32,11 55,11" fill="none" stroke="#12AEB0" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round"></polyline></svg><div style="width:125px;height:2px;background:linear-gradient(90deg,#12B8B0,#1557A6);"></div></div></div>""",
    unsafe_allow_html=True
    
)

st.write("Explainable Stroke Risk Prediction & Screening Support System")
st.caption(
    "An explainable machine-learning system for stroke-risk screening "
    "and decision support."
)

# Patient inputs
st.subheader("👤Patient Information")
col1, col2 = st.columns(2)
with col1:
    age = st.number_input("Age", min_value=1, max_value=100, value=50, step = 1)

    ever_married = st.selectbox(
        "Ever Married",
        ["No" , "Yes"]
    )
    ever_married = 1 if ever_married == "Yes" else 0

with col2:
    gender = st.selectbox(
        "Gender",
        ["Female" , "Male" , "Other"]
    )
    gender = 0 if gender == "Female" else 1 if gender == "Male" else 2


    Residence_type = st.selectbox(
        "Residence Type",
        ["Urban" , "Rural"]
    )
    Residence_type = 1 if Residence_type == "Urban" else 0


st.subheader("❤️ Health Factors")

col1 , col2 = st.columns(2)
with col1: 
    hypertension = st.selectbox(
        "Hypertension",
        ["No", "Yes"]
    )
    hypertension = 1 if hypertension == "Yes" else 0
    avg_glucose_level = st.number_input(
        "Average Glucose Level",
        min_value=0.0,
        max_value=300.0,
        value = 100.0,
        step = 0.1
    )

with col2: 
    heart_disease = st.selectbox(
        "Heart Disease",
        ["No" , "Yes"]
    )
    heart_disease = 1 if heart_disease == "Yes" else 0

    bmi = st.number_input(
        "BMI",
        min_value=10.0,
        max_value=60.0,
        value=25.0,
        step=0.1
    )

st.subheader("🏢 Lifestyle & Social Factors")

col1 , col2 = st.columns(2)
with col1:
    work_type_display = st.selectbox(
        "Work Type",
            [
                "Private",
                "Self-employed",
                "Govt job",
                "Children",
                "Never worked"
            ]
        )
    
    work_type_map = {
        "Private": "Private",
        "Self-employed": "Self-employed",
        "Govt job": "Govt_job",
        "Children": "children",
        "Never worked": "Never_worked"
    }

work_type = work_type_map[work_type_display]    

with col2: 
    smoking_status = st.selectbox(
        "Smoking Status",
        [

            "never smoked",
            "formerly smoked",
            "smokes",
            "Unknown"
        ]
    )

# if st.button("Predict Stroke Risk"):

if st.button("Assess Stroke Risk", use_container_width=True):
    # Convert Work Type into one-hot encoded columns
    work_type_Govt_job = 1 if work_type == "Govt_job" else 0
    work_type_Never_worked = 1 if work_type == "Never_worked" else 0
    work_type_Private = 1 if work_type == "Private" else 0
    work_type_Self_employed = 1 if work_type == "Self-employed" else 0
    work_type_children = 1 if work_type == "children" else 0

    # Convert Smoking Status into one-hot encoded columns
    smoking_status_Unknown = 1 if smoking_status == "Unknown" else 0
    smoking_status_formerly_smoked = 1 if smoking_status == "formerly smoked" else 0
    smoking_status_never_smoked = 1 if smoking_status == "never smoked" else 0
    smoking_status_smokes = 1 if smoking_status == "smokes" else 0


    patient_data = pd.DataFrame({
        'gender': [gender],
        'age': [age],
        'hypertension': [hypertension],
        'heart_disease': [heart_disease],
        'ever_married': [ever_married],
        'Residence_type': [Residence_type],
        'avg_glucose_level': [avg_glucose_level],
        'bmi': [bmi],

        'work_type_Govt_job': [work_type_Govt_job],
        'work_type_Never_worked': [work_type_Never_worked],
        'work_type_Private': [work_type_Private],
        'work_type_Self-employed': [work_type_Self_employed],
        'work_type_children': [work_type_children],

        'smoking_status_Unknown': [smoking_status_Unknown],
        'smoking_status_formerly smoked': [smoking_status_formerly_smoked],
        'smoking_status_never smoked': [smoking_status_never_smoked],
        'smoking_status_smokes': [smoking_status_smokes]
    })

    # Ensure patient feature order exactly matches the trained model
    patient_data = patient_data.reindex(
        columns=X_train.columns,
        fill_value=0
    )


    #Display label dict

    display_names = {
        "gender": "Gender",
        "age": "Age",
        "hypertension": "Hypertension",
        "heart_disease": "Heart Disease",
        "ever_married": "Ever Married",
        "Residence_type": "Residence Type",
        "avg_glucose_level": "Average Glucose Level",
        "bmi": "BMI",

        "work_type_Govt_job": "Work Type: Govt job",
        "work_type_Never_worked": "Work Type: Never worked",
        "work_type_Private": "Work Type: Private",
        "work_type_Self-employed": "Work Type: Self-employed",
        "work_type_children": "Work Type: Children",

        "smoking_status_Unknown": "Smoking Status: Unknown",
        "smoking_status_formerly smoked": "Smoking Status: Formerly smoked",
        "smoking_status_never smoked": "Smoking Status: Never smoked",
        "smoking_status_smokes": "Smoking Status: Smokes"
    }


    #connect patient data to trained Balanced LR model

    # st.write("Patient columns:", patient_data.columns.tolist())
    # st.write("Model columns:", best_balanced_lr.feature_names_in_.tolist())
    #Make Prediction
    prediction = best_balanced_lr.predict(patient_data)[0] #stroke / no stroke

    #Get Probability
    probability = best_balanced_lr.predict_proba(patient_data)[0][1] #estimate stroke probability
    # st.write("Prediction:" , prediction)
    # st.write("Stroke Probability:", probability)

    # st.subheader("🩺Risk Assessment")
    # if prediction == 1:
    #     st.warning("Higher Stroke Risk Detected")
    # else:
    #     st.success("Lower Stroke Risk Detected")
    # st.metric(
    #     "Estimated Stroke Probability",
    #     f"{probability * 100:.2f}%"

    st.subheader("🩺Risk Assessment")
    if probability >= 0.50:
        st.error("🔴 Higher Stroke Risk Detected")
    elif probability >= 0.20:
        st.warning("🟡 Moderate Stroke Risk Detected")
    else:
        st.success("🟢 Lower Stroke Risk Detected")
    st.metric(
        "Estimated Stroke Probability",
        f"{probability * 100:.2f}%"
    )
    # st.write(f"Estimated Stroke Probability: {probability * 100:.2f}%")
    st.info(
    "This tool is intended for screening and decision-support purposes only "
    "and is not a substitute for professional medical advice or diagnosis."
    )
    

    #Create SHAP explainer 
    #calculates how much each feature contributed to this patient's prediction.
    sample_shap = explainer(patient_data)
    #Display
    st.subheader("Prediction Rationale")
    st.caption(
        "The chart below shows how each patient feature influenced "
        "the model's prediction."
    )
    st.caption(
        "🟥 Red = increases the model output  |  "
        "🟦 Blue = decreases the model output"
        )
  
    display_shap = sample_shap[0]
    # display_shap = sample_shap[0]

    # Create user-friendly feature labels
    display_labels = []
    display_indices = []

    for i, feature in enumerate(patient_data.columns):
        value = patient_data[feature].iloc[0]

    # for feature in patient_data.columns:
    #     value = patient_data[feature].iloc[0]

        if feature == "gender":
            label = f"Gender: {'Female' if value == 0 else 'Male' if value == 1 else 'Other'}"

        elif feature == "age":
            label = "Age"

        elif feature == "hypertension":
            label = f"Hypertension: {'Yes' if value == 1 else 'No'}"

        elif feature == "heart_disease":
            label = f"Heart Disease: {'Yes' if value == 1 else 'No'}"

        elif feature == "ever_married":
            label = f"Ever Married: {'Yes' if value == 1 else 'No'}"

        elif feature == "Residence_type":
            label = f"Residence Type: {'Urban' if value == 1 else 'Rural'}"

        elif feature == "avg_glucose_level":
            label = "Average Glucose Level"

        elif feature == "bmi":
            label = "BMI"

    # -------- WORK TYPE --------
        elif feature.startswith("work_type_"):
            category = feature.replace("work_type_", "")
            category = category.replace("_", " ").title()
            if category == "Self Employed":
                category = "Self-Employed"

             # Show ONLY the selected work type
            if (
                (feature == "work_type_Private" and work_type == "Private") or
                (feature == "work_type_Self-employed" and work_type == "Self-employed") or
                (feature == "work_type_Govt_job" and work_type == "Govt_job") or
                (feature == "work_type_children" and work_type == "children") or
                (feature == "work_type_Never_worked" and work_type == "Never_worked")
            ):
                label = f"Work Type: {category}"
            else:
                continue

        # -------- SMOKING STATUS --------
        elif feature.startswith("smoking_status_"):

            category = feature.replace("smoking_status_", "")
            category = category.replace("_", " ").title()

            # Show ONLY the selected smoking status
            if (
                (feature == "smoking_status_Unknown" and smoking_status == "Unknown") or
                (feature == "smoking_status_formerly smoked" and smoking_status == "formerly smoked") or
                (feature == "smoking_status_never smoked" and smoking_status == "never smoked") or
                (feature == "smoking_status_smokes" and smoking_status == "smokes")
            ):
                label = f"Smoking Status: {category}"
            else:
                 continue
        else:
            label = display_names.get(feature, feature)

        display_labels.append(label)
        display_indices.append(i)
    # Keep only selected/relevant SHAP features
    display_shap = sample_shap[0][display_indices]

    # Apply user-friendly labels
    display_shap.feature_names = display_labels

    # Apply labels to SHAP explanation
    display_shap.feature_names = display_labels

    # Display SHAP waterfall chart
    shap.plots.waterfall(
        display_shap,
        show=False
    )

    ax = plt.gca()
    # Remove automatic encoded value text from labels
    labels = []

    for label in ax.get_yticklabels():
        text = label.get_text()

        if "=" in text:
            text = text.split("=", 1)[1].strip()

        labels.append(text)
    ax.set_yticklabels(labels)
    st.pyplot(plt.gcf())
    plt.close()

#Key factors influencing prediction
    st.subheader("Key Factors Influencing Prediction")
    
    # shap_values = sample_shap[0].values
    # feature_names = patient_data.columns

    shap_values = display_shap.values
    feature_names = display_shap.feature_names

    # Get the indices of the 5 most influential features
    top_indices = sorted(
        range(len(shap_values)),
        key=lambda i: abs(shap_values[i]),
        reverse=True
    )[:5]
    st.write("**Top 5 Factors Influencing This Prediction**")
    st.caption(
    "These factors increased or decreased the model's predicted output."
    )
    for i in top_indices:
        # feature = feature_names[i]
        feature = display_labels[i]
        value = shap_values[i]
        direction = "↑ Increases" if value > 0 else "↓ Decreases"

        st.write(
            f"**{feature}** — {direction} ({value:+.2f})"
        )

    st.subheader("Feature Contribution Details")
    contribution_data = []
    for i in top_indices:
        feature = display_labels[i]
        value = shap_values[i]
        # Convert encoded model value into user-friendly value
        if feature == "Gender: Female":
            patient_value = "Female"
        elif feature == "Gender: Male":
            patient_value = "Male"
        elif feature == "Gender: Other":
            patient_value = "Other"
        elif feature.startswith("Ever Married:"):
            patient_value = "Yes" if patient_data.iloc[0, i] == 1 else "No"
        elif feature.startswith("Hypertension:"):
            patient_value = "Yes" if patient_data.iloc[0, i] == 1 else "No"
        elif feature.startswith("Heart Disease:"):
            patient_value = "Yes" if patient_data.iloc[0, i] == 1 else "No"
        elif feature.startswith("Residence Type:"):
            patient_value = "Urban" if patient_data.iloc[0, i] == 1 else "Rural"
        # elif feature.startswith("Smoking Status:"):
        #     patient_value = feature.split(": ", 1)[1]
        # elif feature.startswith("Work Type:"):
        #     patient_value = feature.split(": ", 1)[1]
        elif feature.startswith("Smoking Status:"):
            patient_value = smoking_status.replace("_", " ").title()
        elif feature.startswith("Work Type:"):
            patient_value = work_type.replace("_", " ").title()
        else:
            patient_value = patient_data.iloc[0, i]

        contribution_data.append({
            "Factor": display_labels[i],
            # "Patient Value": patient_data.iloc[0, i],
            "Patient Value": patient_value,
            "SHAP Contribution": round(shap_values[i], 3)
        })
    contribution_df = pd.DataFrame(contribution_data)
    st.dataframe(
        contribution_df,
        use_container_width=True,
        hide_index=True
    )

#===========================================
# PERSONALIZED STROKE RISK GUIDANCE
#===========================================

    st.markdown("---")
    st.subheader("🩺 What Should You Focus On?")

    guidance = []

    #--------------------------------------------------
    # 1. RISK-BASED OVERALL MESSAGE
    #--------------------------------------------------

    if probability >= 0.50:
        st.error(
            "⚠️ **Higher estimated stroke risk:** "
            "The model indicates a higher estimated risk based on the information provided. "
            "Focus on controlling modifiable risk factors and arrange appropriate medical follow-up."
        )

    elif probability >= 0.20:
        st.warning(
            "⚠️ **Moderate estimated stroke risk:** "
            "The model indicates an intermediate level of estimated risk. "
            "Pay particular attention to your modifiable risk factors and maintain regular health checkups."
        )

    else:
        st.success(
            "💚 **Lower estimated stroke risk:** "
            "The model indicates a lower estimated risk based on the information provided. "
            "Continue maintaining healthy habits and regular preventive checkups."
        )


    # ------------------------------------------------------
    # 2. BLOOD PRESSURE / HYPERTENSION
    # ------------------------------------------------------

    if hypertension == 1:

        guidance.append(
            "❤️ **Blood Pressure:** "
            "Monitor your blood pressure regularly. Follow the treatment plan prescribed "
            "by your healthcare professional and discuss consistently high readings during follow-up visits."
        )

    else:

        guidance.append(
            "❤️ **Blood Pressure:** "
            "Continue maintaining healthy blood-pressure habits and include blood-pressure "
            "checks during routine health visits."
        )


    # ------------------------------------------------------
    # 3. HEART DISEASE
    # ------------------------------------------------------

    if heart_disease == 1:

        guidance.append(
            "🫀 **Heart Health:** "
            "Because heart disease is present, keep regular cardiovascular follow-ups "
            "and take prescribed medicines consistently. Report new or worsening symptoms "
            "to a healthcare professional."
        )


    # ------------------------------------------------------
    # 4. GLUCOSE
    # ------------------------------------------------------

    if avg_glucose_level >= 200:

        guidance.append(
            "🩸 **Blood Glucose:** "
            "Your recorded average glucose level is high. Discuss glucose evaluation and "
            "appropriate management with a healthcare professional and follow recommended monitoring."
        )

    elif avg_glucose_level >= 126:

        guidance.append(
            "🩸 **Blood Glucose:** "
            "Your recorded average glucose level is elevated. Consider appropriate glucose "
            "testing and discuss the result with a healthcare professional."
        )

    else:

        guidance.append(
            "🩸 **Blood Glucose:** "
            "Continue maintaining healthy eating habits and include glucose checks "
            "as recommended during routine health assessments."
        )


    # ------------------------------------------------------
    # 5. BMI / WEIGHT
    # ------------------------------------------------------
    # Dynamic BMI-based guidance
    if bmi >= 30:
        weight_guidance = (
            "Your BMI is in the obesity range. "
            "Focus on gradual, sustainable weight management through "
            "balanced nutrition and regular physical activity appropriate for your health."
        )
    elif bmi >= 25:
        weight_guidance = (
            "Your BMI is in the overweight range. "
            "Focus on balanced nutrition, regular physical activity, "
            "and gradual, sustainable weight management."
        )
    else:
        weight_guidance = (
            "Your BMI is within the healthy range. "
            "Continue maintaining balanced nutrition and regular physical activity."
        )

    guidance.append(f"⚖️ **Weight Management:** {weight_guidance}") 

    # if bmi >= 30:

    #     guidance.append(
    #         f"⚖️ **Weight Management:** {weight_guidance}."
    #     )

    # elif bmi >= 25:

    #     guidance.append(
    #          f"⚖️ **Weight Management:** {weight_guidance}"
    #     )

    # else:

    #     guidance.append(
    #          f"⚖️ **Weight Management:** {weight_guidance}"
    #     )

    # ---------------------------------------------------
    # 6. SMOKING
    # ------------------------------------------------------

    if smoking_status == "smokes":

        guidance.append(
            "🚭 **Smoking:** "
            "If you currently smoke, work toward quitting. Consider professional "
            "smoking-cessation support and avoid returning to regular smoking."
        )

    elif smoking_status == "formerly smoked":

        guidance.append(
            "🚭 **Smoking:** "
            "You have previously smoked. Maintaining a smoke-free lifestyle is an important "
            "step toward reducing cardiovascular and stroke-related risk."
        )

    else:

        guidance.append(
            "🚭 **Smoking:** "
            "Continue avoiding tobacco and second-hand smoke exposure."
        )


    # ------------------------------------------------------
    # 7. PHYSICAL ACTIVITY / LIFESTYLE
    # ------------------------------------------------------

    guidance.append(
        "🏃 **Physical Activity:** "
        "Aim for regular physical activity that is appropriate for your age and health condition. "
        "If you have existing medical conditions, discuss suitable activity levels with a professional."
    )

    #---------------------------------------------------
    # 8. REGULAR CHECKUPS
    #---------------------------------------------------

    if probability >= 0.20:

        guidance.append(
            "🩺 **Regular Follow-up:** "
            "Because the estimated risk is not low, arrange consistent health checkups "
            "and review important risk factors such as blood pressure, glucose, heart health, "
            "weight and smoking status with your healthcare professional."
        )

    else:

        guidance.append(
            "🩺 **Regular Follow-up:** "
            "Continue routine preventive health checkups and monitor important factors "
            "such as blood pressure, glucose, weight and heart health."
        )


    # ------------------------------------------------------
    # DISPLAY GUIDANCE
    # ------------------------------------------------------

    st.markdown("### 🎯 Recommended Focus Areas")

    for item in guidance:
        st.markdown(f"- {item}")


    # ------------------------------------------------------
    # IMPORTANT NOTE
    # ------------------------------------------------------

    st.caption(
        "These recommendations are general risk-factor guidance based on the information "
        "entered into the screening system. They do not constitute a medical diagnosis "
        "or replace advice from a qualified healthcare professional."
    )

       

    pdf_report = create_pdf_report(
    age=age,
    gender="Female" if gender == 0 else "Male" if gender == 1 else "Other",
    ever_married="Yes" if ever_married == 1 else "No",
    Residence_type="Urban" if Residence_type == 1 else "Rural",
    hypertension="Yes" if hypertension == 1 else "No",
    heart_disease="Yes" if heart_disease == 1 else "No",
    avg_glucose_level=avg_glucose_level,
    bmi=bmi,
    work_type=work_type,
    smoking_status=smoking_status,
    probability=probability,
    prediction=prediction,
    guidance = guidance
)

#Downloadable Assessment Report

    # Create downloadable assessment report

    report = f"""
    MEDINSIGHT
    STROKE RISK SCREENING REPORT
    ========================================

    PATIENT INFORMATION
    Age: {age}

    HEALTH FACTORS
    Hypertension: {"Yes" if hypertension == 1 else "No"}
    Heart Disease: {"Yes" if heart_disease == 1 else "No"}
    Average Glucose Level: {avg_glucose_level}
    BMI: {bmi}

    LIFESTYLE & SOCIAL FACTORS
    Work Type: {work_type}
    Smoking Status: {smoking_status}

    ASSESSMENT
    Estimated Stroke Probability: {probability * 100:.2f}%
    Model Prediction: {"Higher Predicted Risk" if prediction == 1 else "Lower Predicted Risk"}

    TOP CONTRIBUTING FACTORS
    """

    for i in top_indices:
        report += (
            f"{display_labels[i]}: "
            f"{shap_values[i]:+.3f}\n"
        )
#Add this new text to the existing report.
    report += """

    ========================================
    DISCLAIMER

    This tool is intended for screening and decision-support purposes only
    and is not a substitute for professional medical advice or diagnosis.
    """
    st.download_button(
    label="📄 Download Assessment Report",
    data= pdf_report,
    file_name="MedInsight_Assessment_Report.pdf",
    mime="application/pdf",
    use_container_width=True
)

#Header
    st.divider()
    st.subheader("About MedInsight")
    st.write(
        "MedInsight is an explainable machine-learning system designed "
        "to estimate stroke risk from selected patient health and lifestyle factors."
    )

    st.caption(
        "The system combines Logistic Regression with SHAP-based explainability "
        "to provide prediction and feature-level reasoning for screening and decision support."
    )
        
    st.divider()
    st.caption(
        "MedInsight • Explainable ML for Stroke Risk Screening & Decision Support"
    )


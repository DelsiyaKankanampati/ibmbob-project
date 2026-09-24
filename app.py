"""
Delsiya's Heart Disease Analytics Dashboard — Streamlit App
Run: streamlit run app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    roc_auc_score, roc_curve, ConfusionMatrixDisplay
)
import warnings
warnings.filterwarnings('ignore')

# ─────────────────────────────────────────────
# Page config
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Heart Disease Analytics",
    page_icon="❤️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────
# Custom CSS
# ─────────────────────────────────────────────
st.markdown("""
<style>
    .metric-card {
        background: #f7f8fa;
        border-radius: 10px;
        padding: 16px 20px;
        text-align: center;
        border-left: 4px solid #3b82f6;
    }
    .metric-card h3 { font-size: 1.8rem; color: #1f2328; margin: 0; }
    .metric-card p  { color: #57606a; font-size: 0.85rem; margin: 4px 0 0 0; }
    section.main > div { padding-top: 1rem; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# Data loading & processing (cached)
# ─────────────────────────────────────────────
@st.cache_data
def load_and_process():
    df = pd.read_csv("Dataset.csv", encoding="utf-8-sig")
    df = df.dropna(axis=1, how="all")
    df.columns = [
        "patient_id", "age", "gender", "cholesterol", "bmi",
        "heart_rate", "glucose", "systolic_bp", "diastolic_bp", "ecg_result",
    ]
    df.dropna(how="all", inplace=True)
    df.drop_duplicates(inplace=True)

    num_cols = ["age", "cholesterol", "bmi", "heart_rate", "glucose", "systolic_bp", "diastolic_bp"]
    for col in num_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")
        df[col].fillna(df[col].median(), inplace=True)
    for col in ["gender", "ecg_result"]:
        df[col].fillna(df[col].mode()[0], inplace=True)

    df["chol_age_ratio"] = (df["cholesterol"] / df["age"]).round(3)
    df["pulse_pressure"] = df["systolic_bp"] - df["diastolic_bp"]
    df["ecg_abnormal"] = df["ecg_result"].apply(lambda x: 0 if x == "Normal sinus rhythm" else 1)
    df["high_risk"] = ((df["age"] > 60) & (df["cholesterol"] > 240) & (df["ecg_abnormal"] == 1)).astype(int)
    df["gender_enc"] = LabelEncoder().fit_transform(df["gender"])

    def bmi_cat(b):
        if b < 18.5: return "Underweight"
        elif b < 25:  return "Normal"
        elif b < 30:  return "Overweight"
        else:         return "Obese"

    df["bmi_category"] = df["bmi"].apply(bmi_cat)
    df["age_group"] = pd.cut(df["age"], bins=[0, 30, 45, 60, 75, 100],
                              labels=["<30", "30-45", "45-60", "60-75", "75+"])
    df.reset_index(drop=True, inplace=True)
    return df


@st.cache_resource
def train_models(df):
    FEATURES = ["age", "gender_enc", "cholesterol", "bmi", "heart_rate",
                "glucose", "systolic_bp", "diastolic_bp", "pulse_pressure", "chol_age_ratio"]
    X = df[FEATURES].values
    y = df["ecg_abnormal"].values
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc  = scaler.transform(X_test)

    models = {
        "Logistic Regression": (LogisticRegression(max_iter=1000, random_state=42), X_train_sc, X_test_sc),
        "Decision Tree":       (DecisionTreeClassifier(max_depth=6, random_state=42), X_train, X_test),
        "Random Forest":       (RandomForestClassifier(n_estimators=200, max_depth=8, random_state=42), X_train, X_test),
        "Gradient Boosting":   (GradientBoostingClassifier(n_estimators=200, max_depth=4, random_state=42), X_train, X_test),
    }

    results = {}
    for name, (model, X_tr, X_te) in models.items():
        model.fit(X_tr, y_train)
        y_pred  = model.predict(X_te)
        y_proba = model.predict_proba(X_te)[:, 1]
        results[name] = {
            "model":    model,
            "X_te":     X_te,
            "y_pred":   y_pred,
            "y_proba":  y_proba,
            "accuracy": accuracy_score(y_test, y_pred),
            "roc_auc":  roc_auc_score(y_test, y_proba),
            "report":   classification_report(y_test, y_pred,
                            target_names=["Normal", "Abnormal"], output_dict=True),
            "cm":       confusion_matrix(y_test, y_pred),
        }

    rf_imp = pd.Series(
        results["Random Forest"]["model"].feature_importances_, index=FEATURES
    ).sort_values(ascending=False)

    return results, y_test, X_test, X_test_sc, FEATURES, rf_imp, scaler, X_train, X_train_sc, y_train


# ─────────────────────────────────────────────
# Load data
# ─────────────────────────────────────────────
df = load_and_process()
model_results, y_test, X_test, X_test_sc, FEATURES, rf_imp, scaler, X_train, X_train_sc, y_train = train_models(df)

# ─────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────
st.sidebar.image("https://img.icons8.com/color/96/heart-with-pulse.png", width=80)
st.sidebar.title("❤️ Heart Disease\nAnalytics")
st.sidebar.markdown("---")
page = st.sidebar.radio(
    "Navigate",
    ["🏠 Overview", "📊 EDA", "🔬 Hypothesis Tests", "🤖 Model Results", "🔍 Patient Predictor"],
)

st.sidebar.markdown("---")
st.sidebar.markdown("**Filters**")
gender_filter = st.sidebar.multiselect("Gender", df["gender"].unique().tolist(), default=df["gender"].unique().tolist())
age_min, age_max = int(df["age"].min()), int(df["age"].max())
age_range = st.sidebar.slider("Age Range", age_min, age_max, (age_min, age_max))

dff = df[df["gender"].isin(gender_filter) & df["age"].between(*age_range)]

# ─────────────────────────────────────────────
# Page: Overview
# ─────────────────────────────────────────────
if page == "🏠 Overview":
    st.title("❤️ Heart Disease Data Analytics Dashboard")
    st.markdown("**Dataset:** 1,000 patients · 9 clinical features · Binary target: Abnormal ECG")
    st.markdown("---")

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.markdown(f"<div class='metric-card'><h3>{len(dff)}</h3><p>Patients</p></div>", unsafe_allow_html=True)
    c2.markdown(f"<div class='metric-card'><h3>{dff['ecg_abnormal'].sum()}</h3><p>Abnormal ECG</p></div>", unsafe_allow_html=True)
    c3.markdown(f"<div class='metric-card'><h3>{dff['high_risk'].sum()}</h3><p>High-Risk</p></div>", unsafe_allow_html=True)
    c4.markdown(f"<div class='metric-card'><h3>{dff['cholesterol'].mean():.0f}</h3><p>Avg Cholesterol</p></div>", unsafe_allow_html=True)
    c5.markdown(f"<div class='metric-card'><h3>{dff['bmi'].mean():.1f}</h3><p>Avg BMI</p></div>", unsafe_allow_html=True)

    st.markdown("---")
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("ECG Result Distribution")
        ecg_counts = dff["ecg_result"].value_counts()
        fig, ax = plt.subplots(figsize=(5, 5))
        ax.pie(ecg_counts, labels=ecg_counts.index, autopct="%1.1f%%", startangle=140, pctdistance=0.82)
        ax.set_title("ECG Categories")
        st.pyplot(fig)
        plt.close()

    with col2:
        st.subheader("Age Group vs Abnormal ECG (%)")
        age_ecg = dff.groupby("age_group", observed=True)["ecg_abnormal"].mean() * 100
        fig, ax = plt.subplots(figsize=(5, 4))
        age_ecg.plot(kind="bar", ax=ax, color="coral", edgecolor="white")
        ax.set_ylabel("Abnormal ECG (%)")
        ax.set_xticklabels(ax.get_xticklabels(), rotation=0)
        st.pyplot(fig)
        plt.close()

    st.subheader("Descriptive Statistics")
    st.dataframe(dff[["age", "cholesterol", "bmi", "heart_rate", "glucose", "systolic_bp", "diastolic_bp"]]
                 .describe().round(2), use_container_width=True)


# ─────────────────────────────────────────────
# Page: EDA
# ─────────────────────────────────────────────
elif page == "📊 EDA":
    st.title("📊 Exploratory Data Analysis")

    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "Cholesterol", "BMI vs BP", "Correlation Heatmap", "Boxplots", "Risk Groups"
    ])

    with tab1:
        st.subheader("Cholesterol Distribution")
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.hist(dff["cholesterol"], bins=30, color="steelblue", edgecolor="white")
        ax.axvline(dff["cholesterol"].mean(), color="red", linestyle="--",
                   label=f"Mean: {dff['cholesterol'].mean():.1f}")
        ax.set_xlabel("Cholesterol (mg/dl)")
        ax.set_ylabel("Count")
        ax.legend()
        st.pyplot(fig); plt.close()

        col1, col2 = st.columns(2)
        col1.metric("Mean", f"{dff['cholesterol'].mean():.1f} mg/dl")
        col2.metric("Std Dev", f"{dff['cholesterol'].std():.1f} mg/dl")

    with tab2:
        st.subheader("BMI vs Systolic Blood Pressure")
        fig, ax = plt.subplots(figsize=(7, 5))
        sc = ax.scatter(dff["bmi"], dff["systolic_bp"], c=dff["ecg_abnormal"],
                        cmap="RdYlGn_r", alpha=0.6)
        plt.colorbar(sc, ax=ax, label="Abnormal ECG")
        m, b = np.polyfit(dff["bmi"], dff["systolic_bp"], 1)
        ax.plot(sorted(dff["bmi"]), [m*x+b for x in sorted(dff["bmi"])], "r--", linewidth=1.5)
        ax.set_xlabel("BMI")
        ax.set_ylabel("Systolic BP")
        st.pyplot(fig); plt.close()

    with tab3:
        st.subheader("Correlation Matrix")
        corr_cols = ["age", "cholesterol", "bmi", "heart_rate", "glucose",
                     "systolic_bp", "diastolic_bp", "pulse_pressure", "ecg_abnormal"]
        corr = dff[corr_cols].corr().round(2)
        fig, ax = plt.subplots(figsize=(9, 7))
        sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm",
                    linewidths=0.5, ax=ax, vmin=-1, vmax=1)
        st.pyplot(fig); plt.close()

    with tab4:
        st.subheader("Cholesterol by ECG Result")
        fig, ax = plt.subplots(figsize=(10, 5))
        order = dff["ecg_result"].value_counts().index
        sns.boxplot(data=dff, x="ecg_result", y="cholesterol", order=order, ax=ax, palette="Set2")
        ax.set_xticklabels(ax.get_xticklabels(), rotation=30, ha="right")
        ax.set_xlabel("")
        st.pyplot(fig); plt.close()

    with tab5:
        st.subheader("Risk Stratification")
        col1, col2 = st.columns(2)
        with col1:
            bmi_ecg = dff.groupby("bmi_category")["ecg_abnormal"].mean() * 100
            fig, ax = plt.subplots(figsize=(5, 4))
            bmi_ecg.sort_values().plot(kind="barh", ax=ax, color="mediumpurple", edgecolor="white")
            ax.set_xlabel("Abnormal ECG (%)")
            ax.set_title("Abnormal ECG Rate by BMI Category")
            st.pyplot(fig); plt.close()
        with col2:
            st.metric("High-Risk Patients", int(dff["high_risk"].sum()),
                      f"{dff['high_risk'].mean()*100:.1f}% of filtered set")
            hr = dff[dff["high_risk"] == 1]["ecg_result"].value_counts()
            st.dataframe(hr.rename("Count").reset_index().rename(columns={"index": "ECG Result"}),
                         use_container_width=True)


# ─────────────────────────────────────────────
# Page: Hypothesis Tests
# ─────────────────────────────────────────────
elif page == "🔬 Hypothesis Tests":
    st.title("🔬 Correlation & Hypothesis Testing")

    st.subheader("Pearson Correlations")
    cols_to_test = ["cholesterol", "bmi", "age", "glucose", "heart_rate", "systolic_bp"]
    rows = []
    for col in cols_to_test:
        r, p = stats.pearsonr(dff[col], dff["ecg_abnormal"])
        rows.append({"Feature": col, "Pearson r": round(r, 4), "p-value": round(p, 4),
                     "Significant (p<0.05)": "✅" if p < 0.05 else "❌"})
    st.dataframe(pd.DataFrame(rows), use_container_width=True)

    st.markdown("---")
    st.subheader("Chi-Square: Gender vs Abnormal ECG")
    ct = pd.crosstab(dff["gender"], dff["ecg_abnormal"])
    chi2, p_chi, dof, _ = stats.chi2_contingency(ct)
    col1, col2, col3 = st.columns(3)
    col1.metric("Chi² Statistic", f"{chi2:.4f}")
    col2.metric("p-value", f"{p_chi:.4f}")
    col3.metric("Result", "Significant" if p_chi < 0.05 else "Not Significant")
    st.dataframe(ct, use_container_width=True)

    st.markdown("---")
    st.subheader("ANOVA: Cholesterol across ECG Groups")
    ecg_groups = [grp["cholesterol"].values for _, grp in dff.groupby("ecg_result")]
    f_stat, p_anova = stats.f_oneway(*ecg_groups)
    col1, col2 = st.columns(2)
    col1.metric("F-Statistic", f"{f_stat:.4f}")
    col2.metric("p-value", f"{p_anova:.4f}")
    if p_anova < 0.05:
        st.success("✅ Significant difference in cholesterol across ECG groups (ANOVA p < 0.05)")
    else:
        st.info("ℹ️ No significant difference detected.")


# ─────────────────────────────────────────────
# Page: Model Results
# ─────────────────────────────────────────────
elif page == "🤖 Model Results":
    st.title("🤖 Machine Learning Model Results")

    model_names = list(model_results.keys())
    accs  = [model_results[m]["accuracy"] for m in model_names]
    aucs  = [model_results[m]["roc_auc"]  for m in model_names]

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Accuracy & ROC-AUC Comparison")
        x = np.arange(len(model_names))
        width = 0.35
        fig, ax = plt.subplots(figsize=(7, 4))
        b1 = ax.bar(x - width/2, accs, width, label="Accuracy",  color="steelblue")
        b2 = ax.bar(x + width/2, aucs, width, label="ROC-AUC",   color="coral")
        ax.set_ylim(0, 1.15)
        ax.set_xticks(x)
        ax.set_xticklabels([n.replace(" ", "\n") for n in model_names])
        ax.legend()
        for bar in list(b1) + list(b2):
            ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.01,
                    f"{bar.get_height():.3f}", ha="center", fontsize=8)
        st.pyplot(fig); plt.close()

    with col2:
        st.subheader("Feature Importance (Random Forest)")
        fig, ax = plt.subplots(figsize=(6, 4))
        rf_imp.sort_values().plot(kind="barh", ax=ax, color="teal", edgecolor="white")
        ax.set_xlabel("Importance")
        st.pyplot(fig); plt.close()

    st.markdown("---")
    selected_model = st.selectbox("Inspect Model", model_names)
    res = model_results[selected_model]

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Confusion Matrix")
        fig, ax = plt.subplots(figsize=(4, 3))
        ConfusionMatrixDisplay(confusion_matrix=res["cm"],
                               display_labels=["Normal", "Abnormal"]).plot(ax=ax, cmap="Blues", colorbar=False)
        st.pyplot(fig); plt.close()

    with col2:
        st.subheader("ROC Curve")
        fig, ax = plt.subplots(figsize=(4, 3))
        fpr, tpr, _ = roc_curve(y_test, res["y_proba"])
        ax.plot(fpr, tpr, label=f"AUC={res['roc_auc']:.3f}", color="steelblue")
        ax.plot([0, 1], [0, 1], "k--")
        ax.set_xlabel("FPR"); ax.set_ylabel("TPR")
        ax.legend()
        st.pyplot(fig); plt.close()

    st.subheader("Classification Report")
    report_df = pd.DataFrame(res["report"]).T.round(3)
    st.dataframe(report_df, use_container_width=True)


# ─────────────────────────────────────────────
# Page: Patient Predictor
# ─────────────────────────────────────────────
elif page == "🔍 Patient Predictor":
    st.title("🔍 Individual Patient ECG Risk Predictor")
    st.markdown("Enter patient details below to predict whether their ECG result will be **Normal** or **Abnormal**.")

    col1, col2, col3 = st.columns(3)
    with col1:
        age       = st.slider("Age", 18, 100, 50)
        gender    = st.selectbox("Gender", ["Male", "Female"])
        chol      = st.slider("Cholesterol (mg/dl)", 100, 400, 210)
    with col2:
        bmi       = st.slider("BMI", 10.0, 55.0, 25.0, step=0.1)
        hr        = st.slider("Heart Rate (bpm)", 40, 130, 75)
        glucose   = st.slider("Glucose (mg/dl)", 60, 200, 95)
    with col3:
        sys_bp    = st.slider("Systolic BP (mm Hg)", 80, 200, 120)
        dia_bp    = st.slider("Diastolic BP (mm Hg)", 50, 130, 80)

    gender_enc     = 1 if gender == "Male" else 0
    pulse_pressure = sys_bp - dia_bp
    chol_age_ratio = round(chol / age, 3)

    input_data = np.array([[age, gender_enc, chol, bmi, hr, glucose,
                            sys_bp, dia_bp, pulse_pressure, chol_age_ratio]])

    if st.button("🔮 Predict ECG Risk", use_container_width=True):
        rf_model = model_results["Random Forest"]["model"]
        pred      = rf_model.predict(input_data)[0]
        proba     = rf_model.predict_proba(input_data)[0][1]

        if pred == 0:
            st.success(f"✅ **Normal ECG** predicted — Abnormal probability: **{proba*100:.1f}%**")
        else:
            st.error(f"⚠️ **Abnormal ECG** predicted — Abnormal probability: **{proba*100:.1f}%**")

        # Mini feature importance for this patient
        st.markdown("### Risk Factor Summary")
        row = pd.DataFrame({
            "Feature":      FEATURES,
            "Patient Value": input_data[0],
            "Dataset Mean":  [df[f].mean() if f in df.columns else np.nan for f in FEATURES],
        })
        row["vs Mean"] = row["Patient Value"] - row["Dataset Mean"]
        st.dataframe(row.round(3), use_container_width=True)

# ─────────────────────────────────────────────
# Footer
# ─────────────────────────────────────────────
st.markdown("---")
st.markdown(
    "<p style='text-align:center;color:#57606a;font-size:0.8rem;'>"
    "❤️ Heart Disease Analytics · Delsiya · Built with Streamlit & scikit-learn"
    "</p>",
    unsafe_allow_html=True,
)

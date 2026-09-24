# ❤️ Heart Disease Data Analytics Project

> **Author:** Delsiya  
> **File:** `Delsiya_HeartDisease.ipynb` + `app.py` (Streamlit Dashboard)  
> **Dataset:** Heart Disease Patient Dataset — 1,000 records, 9 clinical features

---

## 📋 Project Description

This project is a complete **end-to-end data analytics and machine learning pipeline** applied to a heart disease patient dataset. The goal is to identify clinical risk factors associated with **abnormal ECG outcomes** and build predictive models that classify patients as normal or abnormal.

The project covers:

| Step | Task |
|------|------|
| 1 | **Data Cleaning** — missing values, duplicates, type coercion |
| 2 | **Exploratory Data Analysis (EDA)** — descriptive stats, distributions, demographics |
| 3 | **Feature Engineering** — cholesterol-to-age ratio, BMI category, pulse pressure, age groups |
| 4 | **Correlation & Hypothesis Testing** — Pearson r, Chi-Square, ANOVA |
| 5 | **Predictive Modeling** — Logistic Regression, Decision Tree, Random Forest, Gradient Boosting |
| 6 | **Validation & Evaluation** — Accuracy, ROC-AUC, Confusion Matrix, Classification Report |
| 7 | **Visualization** — Histograms, scatter plots, heatmaps, pie charts, feature importance |
| 8 | **Streamlit Dashboard** — Interactive 5-page web app with a real-time patient predictor |

---

## 📂 Dataset

| Attribute | Detail |
|-----------|--------|
| **Source** | Provided dataset (`Dataset.csv`) |
| **Records** | 1,000 patients |
| **Features** | 9 clinical variables |
| **Target** | `ecg_result` → binary `ecg_abnormal` (0 = Normal sinus rhythm, 1 = Abnormal) |

### Features

| Column | Description | Unit |
|--------|-------------|------|
| `patient_id` | Unique patient identifier | Integer |
| `age` | Patient age | Years |
| `gender` | Patient sex | Male / Female |
| `cholesterol` | Total blood cholesterol | mg/dl |
| `bmi` | Body Mass Index | kg/m² |
| `heart_rate` | Resting heart rate | bpm |
| `glucose` | Fasting blood glucose | mg/dl |
| `systolic_bp` | Systolic blood pressure | mm Hg |
| `diastolic_bp` | Diastolic blood pressure | mm Hg |
| `ecg_result` | Resting ECG result (target) | 6-category string |

**ECG Categories:** Normal sinus rhythm · Sinus bradycardia · Atrial fibrillation · ST depression (ischemia) · Left bundle branch block · ST elevation (injury)

---

## 🗂️ Project Files

```
HeartDisease/
├── Dataset.csv                   ← Raw dataset
├── Delsiya_HeartDisease.ipynb    ← Jupyter Notebook (full analytics pipeline)
├── app.py                        ← Streamlit dashboard application
├── requirements.txt              ← Python dependencies
├── Delsiya_ProjectReport.docx    ← Full project report (Word)
└── README.md                     ← This file
```

**Generated outputs** (created when the notebook is run):

```
cholesterol_hist.png · age_gender_hist.png · ecg_pie.png · bmi_sysbp_scatter.png
correlation_heatmap.png · ecg_abnormal_by_age.png · cholesterol_by_ecg.png
ecg_by_bmi.png · model_comparison.png · confusion_matrix_rf.png
roc_curves.png · feature_importance.png · decision_tree.png
```

---

## 🛠️ Technologies Used

| Category | Library / Tool |
|----------|---------------|
| Data manipulation | `pandas`, `numpy` |
| Visualization | `matplotlib`, `seaborn` |
| Statistical testing | `scipy.stats` |
| Machine learning | `scikit-learn` |
| Web dashboard | `streamlit` |
| Notebook | `jupyter` / `ipykernel` |
| Report | Microsoft Word (`.docx`) |

---

## ⚙️ Setup & Run Instructions

### 1. Prerequisites

- Python 3.10 or higher
- `pip` package manager

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Run the Jupyter Notebook

```bash
jupyter notebook Delsiya_HeartDisease.ipynb
```

Run all cells in order (`Kernel → Restart & Run All`). All charts will be saved as PNG files in the same directory.

### 4. Launch the Streamlit Dashboard

```bash
streamlit run app.py
```

The dashboard opens at `http://localhost:8501` in your default browser.

> **Note:** `Dataset.csv` must be in the same directory as both `Delsiya_HeartDisease.ipynb` and `app.py`.

---

## 📊 Key Findings

1. **Age is the #1 risk factor** — patients over 60 show significantly higher abnormal ECG rates across all six ECG categories.
2. **BMI and systolic BP are strongly correlated** (r ≈ 0.72) — obesity compounds cardiovascular risk through elevated blood pressure.
3. **Atrial fibrillation** is disproportionately prevalent in elderly patients with high cholesterol.
4. **Random Forest** is the best-performing model (accuracy ~0.75, ROC-AUC ~0.82).
5. **Cholesterol-to-age ratio** outperforms raw cholesterol as a predictive feature, validating domain-driven feature engineering.

---

## 🌐 Streamlit Dashboard Pages

| Page | Content |
|------|---------|
| 🏠 Overview | KPI cards, ECG pie chart, age-group bar chart, descriptive stats |
| 📊 EDA | Cholesterol histogram, BMI vs BP scatter, correlation heatmap, boxplots, risk groups |
| 🔬 Hypothesis Tests | Pearson correlations table, chi-square test, ANOVA results |
| 🤖 Model Results | Model comparison, confusion matrix, ROC curves, classification report |
| 🔍 Patient Predictor | Real-time individual risk prediction using trained Random Forest model |

---

## 📜 License

This project is submitted as an academic data analytics assignment. The dataset is provided for educational purposes only.

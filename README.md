# 🚗 Vehicle Insurance Risk Assessment System

### Machine Learning Based Vehicle Insurance Risk Assessment

A web-based application that uses Machine Learning to estimate the probability of vehicle insurance claims based on driver-related information.

## 📌 Project Overview

The Vehicle Insurance Risk Assessment System helps estimate the likelihood of an insurance claim using driver details such as age, past accidents, speeding tickets, and annual mileage.

The application provides role-based access for Admin, Analyst, and Underwriter users.

## 🎯 Objectives

- Estimate vehicle insurance claim probability.
- Identify low, medium, and high claim-risk levels.
- Provide understandable risk indicators.
- Maintain assessment history.
- Support manual underwriting decisions.
- Evaluate Machine Learning model performance.

## 🛠️ Technologies Used

| Category | Technologies |
|---|---|
| Programming Language | Python |
| Machine Learning | Scikit-learn |
| Algorithm | Logistic Regression |
| Backend | Flask |
| Frontend | HTML, CSS |
| Database | SQLite |
| Data Processing | Pandas, NumPy |

## 📊 Dataset Details

The model uses the following input features:

- Driver Age
- Past Accidents
- Speeding Tickets
- Annual Mileage

**Target Variable:** `claim_filed`

The target indicates whether an insurance claim was filed.

## 🤖 Machine Learning Algorithm

**Logistic Regression**

Logistic Regression is a supervised Machine Learning classification algorithm used to predict the probability of a binary outcome.

It is suitable for this project because the target variable has two classes:

- 0 – No Claim
- 1 – Claim Filed

The model estimates the probability of an insurance claim based on the input features.

## ⚙️ Application Modules

### 1. Admin
- Manage user accounts.
- Access vehicle risk assessments.
- View assessment history.
- View model performance.

### 2. Analyst
- Perform risk assessments.
- View claim-risk predictions.
- Review assessment history.
- Access model performance information.

### 3. Underwriter
- Review driver information.
- Examine estimated claim probability.
- View risk indicators.
- Select underwriting decisions:
  - Accept
  - Reject
  - Refer for Review

Underwriting decisions are manually selected and are not automatically determined by the ML model.

## 📈 Model Performance

| Evaluation Metric | Result |
|---|---:|
| Accuracy | 71.67% |
| Precision | 0.7143 |
| Recall | 0.4737 |
| F1-Score | 0.5696 |
| ROC-AUC | 0.7823 |

These results are based on the model evaluation performed on the test dataset.

## 🔄 Project Workflow

1. Dataset Collection
2. Data Preprocessing
3. Exploratory Data Analysis
4. Train-Test Splitting
5. Model Training
6. Model Evaluation
7. Flask Application Development
8. User Authentication and Role Management
9. Risk Prediction
10. Underwriting Decision and History Management

## ▶️ How to Run the Project

1. Clone the repository:

```bash
git clone https://github.com/mahathipilli4/Vehicle-Insurance-Risk-Assessment.git
```

2. Navigate to the project folder:

```bash
cd Vehicle-Insurance-Risk-Assessment
```

3. Install the required libraries:

```bash
pip install flask pandas numpy scikit-learn joblib
```

4. Run the Flask application:

```bash
python app.py
```

5. Open the application in your browser:

```text
http://127.0.0.1:5000
```

## 🔐 User Roles

The application uses role-based authentication to provide different features and access permissions for Admin, Analyst, and Underwriter users.

## ⚠️ Disclaimer

This system provides estimated claim probabilities for decision support. It does not guarantee future insurance claims or automatically make final insurance approval or rejection decisions.

## 👩‍💻 Project Repository

[Vehicle Insurance Risk Assessment – GitHub](https://github.com/mahathipilli4/Vehicle-Insurance-Risk-Assessment)

---

**Developed as an academic Machine Learning project.**

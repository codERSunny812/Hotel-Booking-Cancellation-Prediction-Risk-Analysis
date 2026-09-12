# 🏨 Hotel Booking Cancellation Prediction

An end-to-end **Machine Learning project** that predicts whether a hotel booking is likely to be **canceled** based on booking details, customer information, and reservation characteristics.

The project focuses on building a complete ML pipeline — from **Exploratory Data Analysis (EDA)** and feature engineering to model training, **XGBoost**, explainability using **SHAP**, and deployment with **Streamlit**.

---

## 📌 Problem Statement

Hotel booking cancellations can cause significant revenue loss and make it difficult for hotels to manage room availability and resources.

The objective of this project is to build a machine learning model that can predict:

> **Will a hotel booking be canceled or not?**

This can help hotels identify high-risk bookings and take proactive actions such as confirmation reminders, targeted communication, or optimized room allocation.

---

## 🎯 Objectives

- Understand the patterns behind hotel booking cancellations.
- Perform detailed **Exploratory Data Analysis (EDA)**.
- Handle missing values and inconsistent data.
- Identify and treat outliers where appropriate.
- Perform feature engineering.
- Encode categorical variables.
- Train and compare multiple machine learning models.
- Build an optimized **XGBoost classification model**.
- Evaluate model performance using appropriate classification metrics.
- Use **SHAP** to explain model predictions.
- Build an interactive **Streamlit application** for prediction.

---

## 📊 Dataset

The project uses the **Hotel Booking Demand** dataset.

The dataset contains booking information from two types of hotels:

- Resort Hotel
- City Hotel

### Important Features

Some of the important variables include:

| Feature | Description |
|---|---|
| `hotel` | Type of hotel |
| `is_canceled` | Target variable indicating whether the booking was canceled |
| `lead_time` | Number of days between booking and arrival |
| `arrival_date_year` | Year of arrival |
| `arrival_date_month` | Month of arrival |
| `arrival_date_week_number` | Week number of arrival |
| `arrival_date_day_of_month` | Day of arrival |
| `stays_in_weekend_nights` | Weekend nights booked |
| `stays_in_week_nights` | Weekday nights booked |
| `adults` | Number of adults |
| `children` | Number of children |
| `babies` | Number of babies |
| `meal` | Meal type |
| `country` | Country of origin |
| `market_segment` | Market segment |
| `distribution_channel` | Booking distribution channel |
| `is_repeated_guest` | Whether the guest has previously stayed at the hotel |
| `previous_cancellations` | Previous canceled bookings |
| `previous_bookings_not_canceled` | Previous bookings that were not canceled |
| `reserved_room_type` | Reserved room type |
| `assigned_room_type` | Assigned room type |
| `booking_changes` | Number of booking changes |
| `deposit_type` | Deposit type |
| `days_in_waiting_list` | Number of days on waiting list |
| `customer_type` | Type of customer |
| `adr` | Average Daily Rate |
| `required_car_parking_spaces` | Parking spaces requested |
| `total_of_special_requests` | Number of special requests |

### Target Variable

```text
is_canceled

0 → Booking was not canceled
1 → Booking was canceled
```

---

# 🔍 Project Workflow

```text
Raw Dataset
     │
     ▼
Data Understanding
     │
     ▼
Exploratory Data Analysis
     │
     ├── Missing Values
     ├── Distributions
     ├── Correlations
     ├── Categorical Analysis
     └── Cancellation Patterns
     │
     ▼
Data Cleaning
     │
     ├── Missing Value Handling
     ├── Duplicate Handling
     └── Outlier Analysis
     │
     ▼
Feature Engineering
     │
     ├── Date Features
     ├── Stay Duration
     ├── Total Guests
     └── Other Derived Features
     │
     ▼
Feature Encoding
     │
     ▼
Train / Validation / Test Split
     │
     ▼
Baseline ML Models
     │
     ├── Logistic Regression
     ├── Decision Tree
     ├── Random Forest
     └── XGBoost
     │
     ▼
Model Evaluation
     │
     ├── Accuracy
     ├── Precision
     ├── Recall
     ├── F1 Score
     ├── ROC-AUC
     └── Confusion Matrix
     │
     ▼
XGBoost Optimization
     │
     ▼
SHAP Explainability
     │
     ▼
Streamlit Application
```

---

# 🔎 Exploratory Data Analysis

The EDA phase investigates:

### Dataset Structure

- Number of observations
- Number of features
- Data types
- Unique values
- Duplicate records

### Missing Values

Missing-value analysis is performed to identify columns requiring:

- Imputation
- Removal
- Special treatment

### Univariate Analysis

Distribution analysis of numerical variables such as:

- `lead_time`
- `adr`
- `adults`
- `children`
- `stays_in_week_nights`
- `stays_in_weekend_nights`

### Categorical Analysis

Analysis of variables such as:

- Hotel type
- Market segment
- Deposit type
- Customer type
- Meal type
- Distribution channel

### Cancellation Analysis

The project investigates questions such as:

- Which hotel has a higher cancellation rate?
- Does longer lead time increase cancellation probability?
- Which market segments cancel more frequently?
- Does deposit type influence cancellations?
- How does ADR relate to cancellation?
- Do repeated guests cancel less frequently?

---

# 🛠️ Feature Engineering

Several meaningful features can be created from the raw dataset.

### Total Stay Duration

```python
total_stay_nights = (
    stays_in_weekend_nights +
    stays_in_week_nights
)
```

### Total Guests

```python
total_guests = adults + children + babies
```

### Total Special Requests

The existing `total_of_special_requests` feature can be used to capture customer requirements and engagement.

### Date-Based Features

Arrival-related information can be transformed into useful temporal features such as:

- Arrival month
- Arrival quarter
- Arrival season
- Arrival day

Feature engineering will be evaluated carefully to avoid introducing unnecessary or redundant variables.

---

# 🤖 Machine Learning Models

The project compares multiple classification algorithms.

### 1. Logistic Regression

Used as a baseline classification model.

### 2. Decision Tree

Used to capture non-linear relationships between features.

### 3. Random Forest

An ensemble of decision trees that can capture complex feature interactions while generally providing better robustness than a single tree.

### 4. XGBoost

The primary model used for this project.

XGBoost is particularly suitable for structured/tabular data and can model complex non-linear relationships between booking characteristics and cancellation behavior.

---

# 📈 Model Evaluation

Since this is a binary classification problem, multiple evaluation metrics will be considered.

### Accuracy

Measures the overall percentage of correct predictions.

### Precision

Measures how many bookings predicted as canceled were actually canceled.

### Recall

Measures how many of the actual canceled bookings were successfully identified.

### F1 Score

Provides a balance between precision and recall.

### ROC-AUC

Measures the model's ability to distinguish between canceled and non-canceled bookings across different classification thresholds.

### Confusion Matrix

The confusion matrix provides:

```text
                 Predicted
               No       Yes
Actual No      TN       FP
Actual Yes     FN       TP
```

---

# ⚖️ Class Imbalance

The target variable may not be perfectly balanced.

Therefore, class distribution will be analyzed before model training.

Depending on the observed imbalance, techniques such as:

- Class weights
- Threshold tuning
- Stratified splitting

may be considered.

The goal is to avoid relying solely on accuracy when evaluating the model.

---

# 🚀 XGBoost Optimization

After establishing baseline models, XGBoost will be optimized using techniques such as:

- Hyperparameter tuning
- Cross-validation
- Regularization
- Learning-rate optimization
- Tree-depth optimization
- Number of estimators

Potential hyperparameters include:

```text
n_estimators
max_depth
learning_rate
subsample
colsample_bytree
min_child_weight
gamma
reg_alpha
reg_lambda
```

The final model will be selected based on validation performance rather than simply choosing the model with the highest training accuracy.

---

# 🧠 Model Explainability with SHAP

**SHAP (SHapley Additive exPlanations)** will be used to understand how individual features influence model predictions.

SHAP helps answer questions such as:

- Which features are most important?
- Why did the model predict a booking as likely to be canceled?
- Which features increase cancellation probability?
- Which features decrease cancellation probability?

Example explainability outputs may include:

- SHAP feature importance
- SHAP summary plot
- SHAP bar plot
- Individual prediction explanations

This makes the model more interpretable and useful from a business perspective.

---

# 🌐 Streamlit Application

A Streamlit web application will be developed to demonstrate the trained model.

The application will allow a user to enter booking information such as:

```text
Hotel Type
Lead Time
Arrival Information
Number of Guests
Stay Duration
Market Segment
Deposit Type
ADR
Customer Type
Special Requests
...
```

The application will then return:

```text
Prediction: Booking Likely to be Canceled
```

along with the predicted probability.

Example:

```text
Cancellation Probability: 78.4%

Prediction:
⚠️ High Risk of Cancellation
```

The application may also provide model explanations using SHAP.

---

# 📁 Project Structure

```text
hotel-booking-cancellation-prediction/
│
├── data/
│   └── hotel_bookings.csv
│
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_feature_engineering.ipynb
│   ├── 03_model_training.ipynb
│   └── 04_model_explainability.ipynb
│
├── src/
│   ├── data_preprocessing.py
│   ├── feature_engineering.py
│   ├── train.py
│   └── predict.py
│
├── models/
│   └── xgboost_model.pkl
│
├── app/
│   └── app.py
│
├── requirements.txt
├── README.md
└── .gitignore
```

---

# ⚙️ Installation

Clone the repository:

```bash
git clone https://github.com/your-username/hotel-booking-cancellation-prediction.git
```

Navigate to the project directory:

```bash
cd hotel-booking-cancellation-prediction
```

Create a virtual environment:

```bash
python -m venv venv
```

Activate the environment.

### macOS / Linux

```bash
source venv/bin/activate
```

### Windows

```bash
venv\Scripts\activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

# ▶️ Running the Project

Run the Streamlit application:

```bash
streamlit run app/app.py
```

The application will open in your browser.

---

# 🧰 Tech Stack

### Programming Language

- Python

### Data Analysis

- Pandas
- NumPy

### Visualization

- Matplotlib
- Seaborn

### Machine Learning

- Scikit-learn
- XGBoost

### Explainable AI

- SHAP

### Deployment

- Streamlit

### Development Environment

- Jupyter Notebook
- VS Code

---

# 📊 Expected Outcomes

The project aims to produce:

- A cleaned and well-understood hotel booking dataset.
- Meaningful insights from EDA.
- A reliable cancellation prediction model.
- Comparison between multiple ML algorithms.
- An optimized XGBoost classifier.
- Interpretable model predictions using SHAP.
- An interactive Streamlit application.

Final model performance metrics will be added after model training and evaluation.

---

# 💡 Business Impact

A reliable cancellation prediction system can help hotels:

- Identify high-risk bookings.
- Reduce revenue loss from unexpected cancellations.
- Improve room inventory management.
- Optimize booking strategies.
- Prioritize confirmation and retention efforts.
- Better understand customer cancellation behavior.

---

# 🔮 Future Improvements

Potential extensions include:

- Advanced hyperparameter optimization using Optuna.
- Probability calibration.
- Cost-sensitive learning.
- Real-time prediction API.
- Cloud deployment.
- Monitoring model performance after deployment.
- Integration with hotel reservation systems.
- Automated retraining with new booking data.

---

# 👨‍💻 Author

**Sushil Pandey**

M.Tech Data Science  
Army Institute of Technology, Pune

GitHub: [codERSunny812](https://github.com/codERSunny812)

LinkedIn: [sunnydotjsx](https://www.linkedin.com/in/sunnydotjsx/)

---

## ⭐ If you found this project useful

Consider giving the repository a ⭐ and feel free to explore the implementation.
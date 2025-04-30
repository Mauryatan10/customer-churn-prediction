#  Telco Customer Churn Prediction

This project uses machine learning to predict customer churn for a telecom company using customer demographics, service usage, and billing data. It features exploratory data analysis, multiple model evaluations, and an interactive web app built with Streamlit.

---

##  Project Structure

Customer_Churn\
 ├── app.py # Streamlit app for model inference\
 ├── model.pkl # Trained machine learning model \
 ├── preprocessor.pkl # Preprocessing pipeline (encoders, scalers) \
 ├── WA_Fn-UseC_-Telco-Customer-Churn.csv # Dataset \
 └── telco-customer-churn-prediction.ipynb # Jupyter notebook for analysis & training\

---

##  Dataset

- **Source**: [Kaggle - Telco Customer Churn](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)
- **Description**: Contains information on customer demographics, services subscribed, and whether they churned or not.

---

##  Technologies Used

- **Python Libraries**: `pandas`, `numpy`, `matplotlib`, `seaborn`, `scikit-learn`, `xgboost`, `tensorflow`
- **Modeling Techniques**: Logistic Regression, Random Forest, KNN, XGBoost, Neural Networks
- **Web App**: `Streamlit`

---

##  How to Run the App

### 1. Clone the Repository

```bash
git clone https://github.com/Mauryatan10/customer-churn-prediction.git
cd customer-chrun-prediction
```

### 2. Install Required Libraries

```bash
pip install -r requirements.txt
```

### 3. Launch the Streamlit App

```bash
streamlit run app.py
```

---

##  Model Summary
Multiple machine learning models were trained and evaluated using metrics such as Accuracy, F1-Score, Precision, and Recall. After comparing performance, the best-performing model was saved as `model.pkl`, and the associated preprocessing steps (such as encoding and scaling) were saved in `preprocessor.pkl`. These components are integrated into the deployed Streamlit app to ensure consistent and reliable predictions on new input data.

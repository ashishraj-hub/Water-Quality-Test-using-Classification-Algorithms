# **💧 Water Quality Prediction using Machine Learning**

This project predicts whether water is safe (potable) or unsafe for drinking using Machine Learning classification algorithms. It analyzes water quality based on physicochemical properties and builds an automated classification system.

📌 Project Overview

Access to safe drinking water is essential for public health. Traditional laboratory testing methods are accurate but:

⏳ Time-consuming

💰 Expensive

👨‍🔬 Require skilled professionals

To solve this, we developed a Machine Learning-based system that predicts water potability efficiently and accurately.

🎯 Objectives

Understand water quality indicators

Perform Exploratory Data Analysis (EDA)

Preprocess dataset (missing values, scaling, imbalance)

Train multiple ML models

Compare performance

Select best model

Build Final Prediction Pipeline

Save and deploy the model

📂 Dataset

Source: Kaggle – Water Potability Dataset
Dataset contains 3276 samples with 10 features:

Feature	Description
pH	Acidity level of water
Hardness	Water hardness minerals
Solids	Total dissolved solids
Chloramines	Disinfectant level
Sulfate	Sulfate presence
Conductivity	Electrical conductivity
Organic Carbon	Organic contamination
Trihalomethanes	Chemical compounds
Turbidity	Water clarity
Potability	0 = Unsafe, 1 = Safe
🛠️ Tech Stack

Python

Pandas

NumPy

Matplotlib / Seaborn

Scikit-Learn

XGBoost

Joblib

🧪 Exploratory Data Analysis

✔ Checked missing values
✔ Analyzed feature distributions
✔ Studied class imbalance
✔ Understood correlation patterns

Median imputation was used for missing values due to outliers.

🔧 Data Preprocessing

Handled missing values

Feature scaling using StandardScaler

Train-Test Split (80-20)

Stratified sampling applied

🤖 Machine Learning Models Trained

Logistic Regression

Decision Tree

Random Forest

Support Vector Machine (SVM)

XGBoost

📊 Model Evaluation Metrics

Accuracy

Precision

Recall

F1 Score

Confusion Matrix

ROC Curve

ROC-AUC Score

🏆 Best Model
✔ Final Selected Model: Support Vector Machine (SVM)
🔥 Reason:

Highest Accuracy

Highest Precision

Highest ROC-AUC Score

Best separation between safe & unsafe water

Stable performance

🚀 Final Prediction Pipeline

Pipeline includes:

StandardScaler

SVM Model

Model Export using Joblib

Prediction function for new data input

🧑‍💻 How to Run the Project
1️⃣ Clone the Repository
git clone https://github.com/your-username/water-quality-ml.git
cd water-quality-ml

2️⃣ Install Dependencies
pip install -r requirements.txt

3️⃣ Run the Notebook / Script
jupyter notebook


or

python main.py

📁 Project Deliverables

✔ Clean Processed Dataset
✔ Trained ML Model
✔ Jupyter Notebook
✔ Project Report
✔ PPT Presentation

🚧 Future Improvements

Use SMOTE for class balancing

Deploy using Flask/Streamlit

Integrate IoT sensors for real-time monitoring

Improve recall for potable water cases

✅ Results

Machine Learning successfully predicts water quality and supports safer decision-making to protect public health.

🙌 Acknowledgements

Dataset Source: Kaggle – Water Potability Dataset

🧑‍🎓 Developer

Ashish Raj
Machine Learning & Data Science Enthusiast

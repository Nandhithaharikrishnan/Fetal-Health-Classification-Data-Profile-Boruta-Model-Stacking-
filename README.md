# Tech Stack: Python, Data Profile, Boruta Algorithm, Model Stacking, Ensemble Learning 
• Challenge: CTG datasets (2,126 records, 21 features) suffer from class imbalance and redundant features, causing baseline models to 
misclassify pathological cases at 28% false-positive rate. Applied Boruta feature selection to reduce from 21 to 11 features and stacked 
ensemble (Random Forest + XGBoost + Logistic meta-learner). 

• Lifting F1 Score on the minority class from 0.71 to 0.89 and overall accuracy to ~95%. Reduced false positives from 28% to 8% via 
hyperparameter tuning with 5-fold cross-validation, directly improving clinical triage precision by 15%. 

• Published a patent under AI & Healthcare domain, demonstrating novel contribution; research validated against industry standards. 


# 🤰 Fetal Health Prediction using Explainable AI and Ensemble Machine Learning 
An Automated Machine Learning System for Accurate and Interpretable Fetal Health Prediction using Cardiotocography (CTG) Data

📌 Overview
Fetal health monitoring is critical for detecting early signs of fetal distress and preventing adverse pregnancy outcomes. Traditional interpretation of Cardiotocography (CTG) recordings often depends on expert judgment, leading to inconsistencies and delayed diagnosis.The system combines advanced feature selection, class imbalance handling, ensemble learning, and SHAP-based explainability to provide clinically interpretable predictions.This project presents an Explainable Artificial Intelligence (XAI) powered fetal health prediction system that automatically analyzes CTG measurements and classifies fetal conditions into:
✅ Normal
⚠️ Suspect
🚨 Pathological

🎯 Key Features
# Automated End-to-End Pipeline
Data preprocessing
Feature engineering
Feature selection
Model training
Prediction generation
Explainability visualization
# Explainable AI (XAI)
SHAP Summary Plots
SHAP Force Plots
Global Feature Importance
Local Prediction Explanations
# Advanced Feature Selection
BorutaShap Algorithm
Shadow Feature Comparison
Clinically Relevant Predictor Identification
# Ensemble Machine Learning
Random Forest
Gradient Boosting
XGBoost
Stacking Ensemble Architecture
# Class Imbalance Handling
Class Weight Computation
Sample Weight Optimization
Improved Minority Class Detection
# Interactive Deployment
Streamlit Web Application
Single Patient Prediction
Batch CSV Prediction

# Performance Highlights

✔ High classification accuracy

✔ Strong minority class detection

✔ Improved F1-Score using weighted learning

✔ Reduced overfitting through BorutaShap feature selection

✔ Clinically interpretable predictions using SHAP

🔬 Explainable AI Integration Unlike traditional black-box models, this system provides transparency through SHAP explanations.Global Interpretability Identifies the most influential CTG features.Explains overall model behavior.Local Interpretability Explains individual patient predictions.
Highlights factors influencing fetal risk classification.

# Technologies Used
Python
Pandas
NumPy
Scikit-Learn
XGBoost
SHAP
BorutaShap
Matplotlib
Seaborn
Streamlit

# Research Contributions
This project introduces:
Explainability-driven feature selection using BorutaShap SHAP-based medical interpretability Ensemble stacking architecture for fetal health prediction Automated CTG analysis pipeline Real-time prediction interface for healthcare applications

# Patent Publication

*Title: An Automated Machine-Learning System for Fetal Health Prediction Using Cardiotocography (CTG) Data
*Publication Number: IN202641024383 A1
*Publication Date: 13 March 2026
*Applicant: Vellore Institute of Technology (VIT)

# Future Enhancements
*Deep Learning-based CTG Classification
*Real-Time IoT Sensor Integration
*Cloud Deployment
*Electronic Health Record (EHR) Integration
*Mobile Healthcare Application

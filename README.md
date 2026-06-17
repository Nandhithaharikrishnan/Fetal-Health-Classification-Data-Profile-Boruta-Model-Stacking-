# Tech Stack: Python, Data Profile, Boruta Algorithm, Model Stacking, Ensemble Learning 
• Challenge: CTG datasets (2,126 records, 21 features) suffer from class imbalance and redundant features, causing baseline models to 
misclassify pathological cases at 28% false-positive rate. Applied Boruta feature selection to reduce from 21 to 11 features and stacked 
ensemble (Random Forest + XGBoost + Logistic meta-learner). 

• Lifting F1 Score on the minority class from 0.71 to 0.89 and overall accuracy to ~95%. Reduced false positives from 28% to 8% via 
hyperparameter tuning with 5-fold cross-validation, directly improving clinical triage precision by 15%. 

• Published a patent under AI & Healthcare domain, demonstrating novel contribution; research validated against industry standards. 

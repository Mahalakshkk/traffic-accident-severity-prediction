🚦 Traffic Accident Severity Prediction

This repository contains a Machine Learning–based web application that predicts the severity of traffic accidents based on environmental and road-related factors.
The system classifies accidents into Low, Moderate, High, and Critical severity levels using a trained ML model and a Flask web interface.

📦 Project Overview

Objective
To build a machine learning model that predicts traffic accident severity using historical accident data and deploy it as a web application.

Dataset
US Accidents Dataset (March 2023) containing millions of accident records with weather and road conditions.

Severity Levels

Low

Moderate

High

Critical

Technologies Used

Python

Pandas, NumPy

Scikit-learn

XGBoost

Flask

HTML, CSS

🗂 Repository Structure
Traffic-Accident-Severity-Prediction/
│
├── app.py                     # Flask application
├── train_model.py             # Model training script
├── model.pkl                  # Trained ML model
├── scaler.pkl                 # Feature scaler
├── encoder.pkl                # Weather condition encoder
├── US_Accidents_March23.csv   # Dataset (optional if large)
├── feature_importance.png     # Feature importance visualization
│
├── templates/
│   └── index.html             # Frontend UI
│
├── static/
│   └── (CSS / images if any)
│
└── README.md                  # Project documentation

⚙️ Project Workflow
1️⃣ Data Collection & Preprocessing

Loaded US accident dataset

Selected relevant features

Handled missing values

Encoded categorical variables

Scaled numerical features

2️⃣ Model Development

Trained an XGBoost Classifier

Split data into training and testing sets

Evaluated model using classification metrics

Visualized feature importance

3️⃣ Model Deployment

Saved trained model and preprocessors

Built a Flask web application

Created a user-friendly HTML form

Displayed severity prediction with color-coded output

📊 Model Performance

Achieved good accuracy on large-scale data

Handles class imbalance effectively

Feature importance analysis included

(Detailed classification report is printed during training)

▶️ How to Run the Project
Step 1: Train the Model
python train_model.py


This will generate:

model.pkl

scaler.pkl

encoder.pkl

Step 2: Run the Flask App
python app.py

Step 3: Open Browser

Go to:

http://127.0.0.1:5000


Enter values and predict accident severity.

🎯 Key Features

Real-time severity prediction

Clean and simple web interface

Color-coded severity output

End-to-end ML pipeline

Resume-ready project structure

📌 Future Enhancements

Deploy on cloud (Render / Railway / AWS)

Add more input features

Improve model performance with deep learning

Integrate maps and real-time data



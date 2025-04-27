#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Ransomware Classification Model Training Script
This script trains and evaluates multiple machine learning models for ransomware classification.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, 
    f1_score, roc_auc_score, confusion_matrix
)
import joblib

def main():
    print("Starting Ransomware Classification Model Training")
    
    # Check if the dataset file exists
    file_path = 'Ransomware.csv'
    if not os.path.exists(file_path):
        print(f"Error: Dataset file {file_path} not found.")
        return
    
    print("Loading dataset...")
    # Load the dataset
    df = pd.read_csv(file_path, delimiter='|')
    df.columns = df.columns.str.strip()
    print(f"Dataset loaded successfully with {df.shape[0]} rows and {df.shape[1]} columns.")
    
    # Data Preprocessing
    print("Preprocessing data...")
    df.fillna(0, inplace=True)
    
    # Encode categorical variables
    df['Name'] = df['Name'].astype('category').cat.codes
    df['md5'] = df['md5'].astype('category').cat.codes
    
    # Split the data into features and target variable
    X = df.drop('legitimate', axis=1)
    y = df['legitimate']
    
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Standardize the features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Initialize models
    print("Initializing models...")
    models = {
        'Decision Tree': DecisionTreeClassifier(),
        'Random Forest': RandomForestClassifier(),
        'SVM': SVC(probability=True),
        'KNN': KNeighborsClassifier(),
        'Neural Network': MLPClassifier(),
        'Naive Bayes': GaussianNB()
    }
    
    # Train and evaluate models
    results = {}
    print("\nTraining and evaluating models:")
    for name, model in models.items():
        print(f"  - Training {name}...")
        model.fit(X_train_scaled, y_train)
        
        y_pred = model.predict(X_test_scaled)
        y_prob = model.predict_proba(X_test_scaled)[:, 1]
        
        results[name] = {
            'Accuracy': accuracy_score(y_test, y_pred),
            'Precision': precision_score(y_test, y_pred),
            'Recall': recall_score(y_test, y_pred),
            'F1 Score': f1_score(y_test, y_pred),
            'ROC AUC': roc_auc_score(y_test, y_prob)
        }
        
        # Save the model
        joblib.dump(model, f'{name.replace(" ", "_")}.joblib')
        print(f"  - {name} trained and saved successfully.")
    
    # Create a results DataFrame
    results_df = pd.DataFrame(results).T
    print("\nModel Performance Comparison:")
    print(results_df)
    
    # Save the results
    results_df.to_csv('results_df.csv')
    df.to_csv('df.csv', index=False)
    
    # Plot confusion matrices
    output_dir = 'output_plots'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    for name, model in models.items():
        y_pred = model.predict(X_test_scaled)
        cm = confusion_matrix(y_test, y_pred)
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=['Predicted Negative', 'Predicted Positive'], 
                   yticklabels=['Actual Negative', 'Actual Positive'])
        plt.title(f'Confusion Matrix for {name}')
        plt.tight_layout()
        plt.savefig(f'{output_dir}/confusion_matrix_{name.replace(" ", "_")}.png')
        plt.close()
    
    # Plot model comparison
    plt.figure(figsize=(12, 8))
    results_df.plot(kind='bar')
    plt.title('Model Performance Comparison')
    plt.ylabel('Score')
    plt.tight_layout()
    plt.savefig(f'{output_dir}/model_comparison.png')
    plt.close()
    
    print("\nRansomware Classification Model Training Completed Successfully!")
    print(f"Results saved to 'results_df.csv'")
    print(f"Plots saved to '{output_dir}' directory")

if __name__ == "__main__":
    main()
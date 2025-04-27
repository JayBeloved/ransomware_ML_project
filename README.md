# Ransomware Classification Project

This project aims to classify ransomware based on various features, using machine learning models.

## Project Overview

This project analyzes ransomware samples to distinguish between legitimate software and ransomware. It implements several machine learning algorithms including:
- Decision Tree
- Random Forest
- Support Vector Machines (SVM)
- K-Nearest Neighbors (KNN)
- Neural Networks
- Naive Bayes

## Dataset

The project uses a dataset (`Ransomware.csv`) containing various features extracted from executable files, with a binary classification target (`legitimate`):
- 0: Ransomware
- 1: Legitimate software

## Installation

1. Clone this repository
2. Install the required dependencies:
```
pip install -r requirements.txt
```

## Usage

### Running the Jupyter Notebook

To explore the data analysis and model development process:
```
jupyter notebook ransomeware_classification.ipynb
```

### Running the Model Training Script

For a quick model training and evaluation:
```
python run.py
```

## Project Structure

- `ransomeware_classification.ipynb`: Jupyter notebook containing the full data analysis, feature exploration, and model development
- `Ransomware.csv`: Dataset file
- `requirements.txt`: List of required Python packages
- `run.py`: Python script to run the model training and evaluation
- `README.md`: This file

## Results

The project evaluates models using several metrics:
- Accuracy
- Precision
- Recall
- F1 Score
- ROC AUC

Model performance comparisons and feature importance analyses are included in the notebook and output files.

## Export Files

The notebook exports several files:
- `df.csv`: The processed dataframe
- `feature_importances_df.csv`: Feature importance values
- `results_df.csv`: Model performance metrics
- Model files (`.joblib`): Trained models saved for future use

## Contact

For any questions or issues, please contact [Mr. Adedayo Omoniyi].

## License

This project is provided as is, without any warranty.
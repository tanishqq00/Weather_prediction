# Australia Weather Prediction System

## Overview
This project implements a machine learning pipeline to predict whether it will rain tomorrow in Australia based on historical weather data. It handles missing values, categorical encoding, and feature scaling using Scikit-Learn Pipelines.

## Features
- **Modular Architecture**: Separate scripts for data loading, preprocessing, and training.
- **Leakage Prevention**: Transformations are fitted exclusively on training data.
- **Robust Evaluation**: Uses Precision-Recall and F1-score to handle class imbalances.

## How to Run
1. Install dependencies: `pip install -r requirements.txt`
2. Run the pipeline: `python model/main.py`

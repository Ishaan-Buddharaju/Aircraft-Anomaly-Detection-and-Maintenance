#Anomaly Detection & Predictive Maintenance on Airplane Engines

This project applies machine learning and deep learning methods to detect anomalies and predict maintenance needs in industrial systems, specifically airplane engines. It uses LSTM for Remaining Useful Life (RUL) prediction and Gaussian Anomaly Detection and K-Nearest Neighbors (KNN) for classification-based maintenance prediction.

##Datasets
LSTM-based Sequential Dataset
Source: https://www.kaggle.com/datasets/maternusherold/pred-maintanance-data
Used for RUL prediction using time-series sensor data.

####Classification Dataset
Source: https://www.kaggle.com/datasets/shivamb/machine-predictive-maintenance-classification
Used for anomaly detection and failure classification with Gaussian and KNN methods.

Model Overview
LSTM (PyTorch): Predicts Remaining Useful Life from time-series sensor readings.

###Gaussian Anomaly Detection: Unsupervised detection using probabilistic modeling.

###KNN: Supervised method to classify machine state based on proximity to labeled examples.

##AWS MLOps Pipeline
Raw and processed data are stored in Amazon S3.

The LSTM model is trained and evaluated using Amazon SageMaker with a custom PyTorch training script.

Preprocessing and retraining can be automated via SageMaker Processing and Step Functions.

Models are deployed via SageMaker Endpoints or Batch Transform, and monitored using CloudWatch and SageMaker Model Monitor.
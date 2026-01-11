#  Smart Water Analytics - Deep Learning Pipeline

Advanced Deep Learning system for Water Consumption Anomaly Detection and Demand Forecasting using LSTM Autoencoders and Attention-based Neural Networks.

## Project Overview

This project implements a comprehensive water management system that combines:
- **Anomaly Detection**: Identifies unusual water consumption patterns indicating leaks or equipment failures
- **Demand Forecasting**: Predicts future water consumption for optimal resource planning

## Key Features

- **LSTM Autoencoder** for anomaly detection with 0.279% detection rate
- **Attention-LSTM** model for 24-hour demand forecasting
- **Interactive Streamlit Dashboard** for real-time monitoring
- **Comprehensive EDA** with statistical analysis and visualizations
- **Hyperparameter optimization** using Optuna
- **Real-time alerts** and performance monitoring

##  Performance Metrics

| Metric | Value |
|--------|-------|
| Anomalies Detected | 737 out of 263,763 sequences |
| Detection Threshold | 0.0162 |
| Average Forecast Accuracy | 41.17 units |
| Forecast Horizon | 24 hours |

## Installation

### Prerequisites
- Python 3.8+
- pip package manager

### Quick Setup

# Clone the repository
git clone <repository-url>
cd Nitharshna_AML_PROJECT_CODE

# Run setup script
python setup.py

# Or install manually
pip install -r requirements.txt


### Dependencies

pandas>=1.5.0
numpy>=1.21.0
scikit-learn>=1.2.0
tensorflow>=2.10.0
matplotlib>=3.5.0
seaborn>=0.11.0
plotly>=5.0.0
streamlit>=1.20.0
joblib>=1.2.0
scipy>=1.9.0
statsmodels>=0.13.0
optuna>=3.0.0


## Project Structure

Nitharshna_AML_PROJECT_CODE/
├── Dataset/
│   ├── water-consumption-data-main/
│   │   ├── minutely-water-consumption-data.csv
│   │   └── README.md
│   └── water_leak_detection_1000_rows.csv
├── Dashboard.py                 # Interactive Streamlit dashboard
├── water_analytics.py          # Main analytics engine
├── run_analysis.py            # Complete pipeline runner
├── setup.py                   # Setup and installation script
├── requirements.txt           # Project dependencies
└── README.md                  # Project documentation


## Usage

### 1. Run Complete Analysis Pipeline

python run_analysis.py


### 2. Launch Interactive Dashboard

streamlit run Dashboard.py


### 3. Use Analytics Engine Directly

from water_analytics import WaterAnalytics

# Initialize system
analytics = WaterAnalytics()

# Run complete analysis
results = analytics.run_complete_analysis()

# Access results
anomalies = results['anomalies']
forecasts = results['forecasts']


##  Model Architecture

### Anomaly Detection Model
- **Type**: LSTM Autoencoder
- **Architecture**: Encoder-Decoder with 3 LSTM layers each
- **Input**: 48-hour consumption sequences
- **Training**: 20 epochs on normal patterns only
- **Detection**: Reconstruction error threshold (95th percentile)

### Demand Forecasting Model
- **Type**: Attention-LSTM
- **Architecture**: Multi-head attention + LSTM layers
- **Input**: 48-hour historical data
- **Output**: 24-hour predictions
- **Optimization**: Hyperparameter tuning with Optuna

## Dashboard Features

### Navigation Sections
1. **Project Overview**: Key metrics and system description
2. **Anomaly Testing**: Real-time anomaly detection interface
3. **Forecasting Testing**: Interactive demand prediction
4. **Model Results**: Training performance and visualizations

### Interactive Testing
- Generate sample patterns or input custom data
- Real-time anomaly classification
- 24-hour demand forecasting
- Visual result analysis

## Data Processing Pipeline

1. **Data Loading**: CSV files or synthetic data generation
2. **Exploratory Analysis**: Statistical analysis and visualizations
3. **Feature Engineering**: Cyclical encoding, rolling statistics
4. **Preprocessing**: Scaling, sequence creation, anomaly labeling
5. **Model Training**: LSTM autoencoder and attention-LSTM
6. **Prediction**: Anomaly detection and demand forecasting
7. **Visualization**: Interactive plots and dashboards

## Key Algorithms

### Anomaly Detection Process

# 1. Train autoencoder on normal patterns
autoencoder.fit(normal_sequences)

# 2. Calculate reconstruction errors
predictions = autoencoder.predict(test_sequences)
mse_scores = mean_squared_error(test_sequences, predictions)

# 3. Classify anomalies
anomalies = mse_scores > threshold


### Forecasting Process

# 1. Prepare sequence data
sequences = create_sequences(data, length=48)

# 2. Train attention-LSTM model
model.fit(sequences, targets)

# 3. Generate predictions
forecasts = model.predict(last_sequence)


## Configuration Options

### Model Parameters
- **Sequence Length**: 48 hours (configurable)
- **Forecast Horizon**: 24 hours (configurable)
- **Anomaly Threshold**: Auto-calculated (95th percentile)
- **Training Epochs**: 20 (anomaly), 30 (forecast)

### Feature Engineering
- Cyclical time encoding (hour, day, month)
- Rolling statistics (24-hour windows)
- Lag features (1-hour, 24-hour)
- Z-score anomaly indicators

## Results Interpretation

### Anomaly Detection
- **Normal Pattern**: MSE < 0.0162
- **Anomalous Pattern**: MSE > 0.0162
- **Confidence**: Based on reconstruction error magnitude

### Demand Forecasting
- **Accuracy**: Mean absolute error on validation set
- **Trend Analysis**: Increasing/decreasing consumption patterns
- **Peak Detection**: Identification of high-usage periods

## Alert System

The system provides automated alerts for:
- Detected anomalies requiring immediate attention
- Unusual consumption patterns
- Equipment malfunction indicators
- Resource planning recommendations

## Technical Details

### Data Requirements
- Minimum 100 data points for analysis
- Hourly consumption measurements
- Optional: temperature, pressure sensors
- Timestamp information required

### Performance Optimization
- Early stopping to prevent overfitting
- Learning rate scheduling
- Dropout regularization
- L1/L2 weight regularization

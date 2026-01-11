import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.ensemble import IsolationForest
from sklearn.cluster import DBSCAN

import tensorflow as tf
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import LSTM, GRU, Dense, Dropout, Input, RepeatVector, TimeDistributed
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Attention, MultiHeadAttention
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.regularizers import l1_l2

import optuna
from scipy import stats
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller

class WaterAnalytics:
    def __init__(self):
        self.scaler = MinMaxScaler()
        self.anomaly_model = None
        self.forecast_model = None
        self.threshold = None
        
    def load_and_explore_data(self):
        """Load and perform comprehensive EDA"""
        print("🔍 Loading and Exploring Data...")
        
        # Load datasets
        try:
            self.leak_data = pd.read_csv('water_leak_detection_1000_rows.csv')
            self.consumption_data = pd.read_csv('water-consumption-data-main/minutely-water-consumption-data.csv')
            print(f"✅ Leak data shape: {self.leak_data.shape}")
            print(f"✅ Consumption data shape: {self.consumption_data.shape}")
            
            # Convert to standard format
            if 'created_at' in self.consumption_data.columns:
                self.consumption_data['timestamp'] = pd.to_datetime(self.consumption_data['created_at'], unit='s')
                self.consumption_data['consumption'] = self.consumption_data['sensor_value']
            
        except:
            # Generate synthetic data if files not found
            print("📊 Generating synthetic water consumption data...")
            self.consumption_data = self._generate_synthetic_data()
            
        return self.consumption_data
    
    def _generate_synthetic_data(self):
        """Generate realistic water consumption data"""
        dates = pd.date_range('2023-01-01', periods=10000, freq='H')
        
        # Base consumption with daily and weekly patterns
        base = 50 + 20 * np.sin(2 * np.pi * np.arange(len(dates)) / 24)  # Daily pattern
        weekly = 10 * np.sin(2 * np.pi * np.arange(len(dates)) / (24*7))  # Weekly pattern
        
        # Add noise and anomalies
        noise = np.random.normal(0, 5, len(dates))
        anomalies = np.random.choice([0, 1], len(dates), p=[0.95, 0.05])
        anomaly_values = anomalies * np.random.normal(100, 30, len(dates))
        
        consumption = base + weekly + noise + anomaly_values
        consumption = np.maximum(consumption, 0)  # Ensure non-negative
        
        return pd.DataFrame({
            'timestamp': dates,
            'consumption': consumption,
            'temperature': 20 + 10 * np.sin(2 * np.pi * np.arange(len(dates)) / (24*365)) + np.random.normal(0, 2, len(dates)),
            'pressure': 2.5 + 0.5 * np.sin(2 * np.pi * np.arange(len(dates)) / 24) + np.random.normal(0, 0.1, len(dates))
        })
    
    def comprehensive_eda(self, data):
        """Perform comprehensive exploratory data analysis"""
        print("\n📈 Performing Comprehensive EDA...")
        
        # Basic statistics
        print("\n📊 Dataset Overview:")
        print(data.describe())
        print(f"\n🔍 Missing values:\n{data.isnull().sum()}")
        
        # Time series plots
        fig = make_subplots(
            rows=3, cols=2,
            subplot_titles=['Water Consumption Over Time', 'Distribution', 
                          'Hourly Pattern', 'Daily Pattern',
                          'Seasonal Decomposition', 'Anomaly Detection'],
            specs=[[{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}],
                   [{"colspan": 2}, None]]
        )
        
        # Time series plot
        fig.add_trace(go.Scatter(x=data['timestamp'], y=data['consumption'], 
                                name='Consumption', line=dict(width=1)), row=1, col=1)
        
        # Distribution
        fig.add_trace(go.Histogram(x=data['consumption'], name='Distribution', 
                                  nbinsx=50), row=1, col=2)
        
        # Hourly pattern
        hourly_avg = data.groupby(data['timestamp'].dt.hour)['consumption'].mean()
        fig.add_trace(go.Scatter(x=hourly_avg.index, y=hourly_avg.values, 
                                name='Hourly Avg', mode='lines+markers'), row=2, col=1)
        
        # Daily pattern
        daily_avg = data.groupby(data['timestamp'].dt.dayofweek)['consumption'].mean()
        fig.add_trace(go.Scatter(x=['Mon','Tue','Wed','Thu','Fri','Sat','Sun'], 
                                y=daily_avg.values, name='Daily Avg', mode='lines+markers'), row=2, col=2)
        
        # Seasonal decomposition
        if len(data) > 100:
            decomposition = seasonal_decompose(data['consumption'].dropna(), 
                                             model='additive', period=24)
            fig.add_trace(go.Scatter(x=data['timestamp'][:len(decomposition.trend)], 
                                    y=decomposition.trend, name='Trend'), row=3, col=1)
        
        fig.update_layout(height=800, title_text="Water Consumption Analysis Dashboard")
        fig.show()
        
        # Statistical tests
        print(f"\n🔬 Statistical Analysis:")
        adf_result = adfuller(data['consumption'].dropna())
        print(f"ADF Statistic: {adf_result[0]:.4f}")
        print(f"p-value: {adf_result[1]:.4f}")
        print(f"Stationarity: {'Stationary' if adf_result[1] < 0.05 else 'Non-stationary'}")
        
        return data
    
    def advanced_preprocessing(self, data):
        """Advanced preprocessing with feature engineering"""
        print("\n🔧 Advanced Preprocessing...")
        
        data = data.copy()
        
        # Ensure we have the right columns
        if 'created_at' in data.columns and 'timestamp' not in data.columns:
            data['timestamp'] = pd.to_datetime(data['created_at'], unit='s')
        if 'sensor_value' in data.columns and 'consumption' not in data.columns:
            data['consumption'] = data['sensor_value']
            
        data['timestamp'] = pd.to_datetime(data['timestamp'])
        data = data.sort_values('timestamp').reset_index(drop=True)
        
        # Feature engineering
        data['hour'] = data['timestamp'].dt.hour
        data['day_of_week'] = data['timestamp'].dt.dayofweek
        data['month'] = data['timestamp'].dt.month
        data['is_weekend'] = (data['day_of_week'] >= 5).astype(int)
        
        # Cyclical encoding
        data['hour_sin'] = np.sin(2 * np.pi * data['hour'] / 24)
        data['hour_cos'] = np.cos(2 * np.pi * data['hour'] / 24)
        data['day_sin'] = np.sin(2 * np.pi * data['day_of_week'] / 7)
        data['day_cos'] = np.cos(2 * np.pi * data['day_of_week'] / 7)
        
        # Rolling statistics
        data['consumption_ma_24'] = data['consumption'].rolling(24).mean()
        data['consumption_std_24'] = data['consumption'].rolling(24).std()
        data['consumption_lag_1'] = data['consumption'].shift(1)
        data['consumption_lag_24'] = data['consumption'].shift(24)
        
        # Anomaly indicators
        data['z_score'] = np.abs(stats.zscore(data['consumption'].fillna(data['consumption'].mean())))
        data['is_anomaly_zscore'] = (data['z_score'] > 3).astype(int)
        
        # Fill missing values
        data = data.fillna(method='ffill').fillna(method='bfill')
        
        print(f"✅ Preprocessed data shape: {data.shape}")
        return data
    
    def create_sequences(self, data, sequence_length=48, target_col='consumption'):
        """Create sequences for time series modeling"""
        sequences = []
        targets = []
        
        for i in range(len(data) - sequence_length):
            seq = data.iloc[i:i+sequence_length][target_col].values
            target = data.iloc[i+sequence_length][target_col]
            sequences.append(seq)
            targets.append(target)
            
        return np.array(sequences), np.array(targets)
    
    def build_lstm_autoencoder(self, sequence_length, n_features=1):
        """Build advanced LSTM Autoencoder for anomaly detection"""
        print("\n🏗️ Building LSTM Autoencoder...")
        
        # Encoder
        encoder_inputs = Input(shape=(sequence_length, n_features))
        encoder_lstm1 = LSTM(128, return_sequences=True, dropout=0.2, 
                           recurrent_dropout=0.2, kernel_regularizer=l1_l2(0.01, 0.01))(encoder_inputs)
        encoder_lstm2 = LSTM(64, return_sequences=True, dropout=0.2)(encoder_lstm1)
        encoder_lstm3 = LSTM(32, return_sequences=False, dropout=0.2)(encoder_lstm2)
        
        # Decoder
        decoder_dense = Dense(32, activation='relu')(encoder_lstm3)
        decoder_repeat = RepeatVector(sequence_length)(decoder_dense)
        decoder_lstm1 = LSTM(32, return_sequences=True, dropout=0.2)(decoder_repeat)
        decoder_lstm2 = LSTM(64, return_sequences=True, dropout=0.2)(decoder_lstm1)
        decoder_lstm3 = LSTM(128, return_sequences=True, dropout=0.2)(decoder_lstm2)
        decoder_outputs = TimeDistributed(Dense(n_features))(decoder_lstm3)
        
        autoencoder = Model(encoder_inputs, decoder_outputs)
        autoencoder.compile(optimizer=Adam(0.001), loss='mse', metrics=['mae'])
        
        return autoencoder
    
    def build_attention_forecast_model(self, sequence_length, n_features=1):
        """Build advanced forecasting model with attention mechanism"""
        print("\n🏗️ Building Attention-based Forecasting Model...")
        
        inputs = Input(shape=(sequence_length, n_features))
        
        # Multi-head attention
        attention_layer = MultiHeadAttention(num_heads=4, key_dim=32)(inputs, inputs)
        attention_output = Dropout(0.2)(attention_layer)
        
        # LSTM layers
        lstm1 = LSTM(128, return_sequences=True, dropout=0.2)(attention_output)
        lstm2 = LSTM(64, return_sequences=True, dropout=0.2)(lstm1)
        lstm3 = LSTM(32, return_sequences=False, dropout=0.2)(lstm2)
        
        # Dense layers
        dense1 = Dense(64, activation='relu', kernel_regularizer=l1_l2(0.01, 0.01))(lstm3)
        dense2 = Dense(32, activation='relu')(dense1)
        outputs = Dense(1, activation='linear')(dense2)
        
        model = Model(inputs, outputs)
        model.compile(optimizer=Adam(0.001), loss='mse', metrics=['mae'])
        
        return model
    
    def train_anomaly_detection(self, data, sequence_length=48):
        """Train anomaly detection model"""
        print("\n🎯 Training Anomaly Detection Model...")
        
        # Prepare data (use only normal data for training)
        normal_data = data[data['is_anomaly_zscore'] == 0]
        sequences, _ = self.create_sequences(normal_data, sequence_length)
        sequences = sequences.reshape(sequences.shape[0], sequences.shape[1], 1)
        
        # Scale data
        sequences_scaled = self.scaler.fit_transform(sequences.reshape(-1, 1)).reshape(sequences.shape)
        
        # Build and train model
        self.anomaly_model = self.build_lstm_autoencoder(sequence_length)
        
        callbacks = [
            EarlyStopping(patience=10, restore_best_weights=True),
            ReduceLROnPlateau(patience=5, factor=0.5)
        ]
        
        history = self.anomaly_model.fit(
            sequences_scaled, sequences_scaled,
            epochs=20, batch_size=32, validation_split=0.2,
            callbacks=callbacks, verbose=1
        )
        
        # Calculate threshold
        predictions = self.anomaly_model.predict(sequences_scaled)
        mse = np.mean(np.power(sequences_scaled - predictions, 2), axis=(1, 2))
        self.threshold = np.percentile(mse, 95)
        
        print(f"✅ Anomaly detection threshold: {self.threshold:.4f}")
        return history
    
    def train_demand_forecasting(self, data, sequence_length=48):
        """Train demand forecasting model with hyperparameter optimization"""
        print("\n📊 Training Demand Forecasting Model...")
        
        def objective(trial):
            # Hyperparameter suggestions
            lstm_units = trial.suggest_int('lstm_units', 32, 128)
            dropout_rate = trial.suggest_float('dropout_rate', 0.1, 0.5)
            learning_rate = trial.suggest_float('learning_rate', 1e-4, 1e-2, log=True)
            
            # Prepare data
            sequences, targets = self.create_sequences(data, sequence_length)
            sequences = sequences.reshape(sequences.shape[0], sequences.shape[1], 1)
            
            # Scale data
            sequences_scaled = self.scaler.fit_transform(sequences.reshape(-1, 1)).reshape(sequences.shape)
            targets_scaled = self.scaler.fit_transform(targets.reshape(-1, 1)).flatten()
            
            # Split data
            split_idx = int(0.8 * len(sequences_scaled))
            X_train, X_val = sequences_scaled[:split_idx], sequences_scaled[split_idx:]
            y_train, y_val = targets_scaled[:split_idx], targets_scaled[split_idx:]
            
            # Build model
            model = Sequential([
                LSTM(lstm_units, return_sequences=True, dropout=dropout_rate, input_shape=(sequence_length, 1)),
                LSTM(lstm_units//2, return_sequences=False, dropout=dropout_rate),
                Dense(32, activation='relu'),
                Dense(1)
            ])
            
            model.compile(optimizer=Adam(learning_rate), loss='mse')
            
            # Train model
            history = model.fit(X_train, y_train, epochs=10, batch_size=32, 
                              validation_data=(X_val, y_val), verbose=0)
            
            return min(history.history['val_loss'])
        
        # Optimize hyperparameters
        study = optuna.create_study(direction='minimize')
        study.optimize(objective, n_trials=5)
        
        # Train final model with best parameters
        best_params = study.best_params
        print(f"🎯 Best parameters: {best_params}")
        
        sequences, targets = self.create_sequences(data, sequence_length)
        sequences = sequences.reshape(sequences.shape[0], sequences.shape[1], 1)
        sequences_scaled = self.scaler.fit_transform(sequences.reshape(-1, 1)).reshape(sequences.shape)
        targets_scaled = self.scaler.fit_transform(targets.reshape(-1, 1)).flatten()
        
        self.forecast_model = self.build_attention_forecast_model(sequence_length)
        
        callbacks = [
            EarlyStopping(patience=15, restore_best_weights=True),
            ReduceLROnPlateau(patience=7, factor=0.5)
        ]
        
        history = self.forecast_model.fit(
            sequences_scaled, targets_scaled,
            epochs=30, batch_size=32, validation_split=0.2,
            callbacks=callbacks, verbose=1
        )
        
        return history
    
    def detect_anomalies(self, data, sequence_length=48):
        """Detect anomalies using trained model"""
        sequences, _ = self.create_sequences(data, sequence_length)
        sequences = sequences.reshape(sequences.shape[0], sequences.shape[1], 1)
        sequences_scaled = self.scaler.transform(sequences.reshape(-1, 1)).reshape(sequences.shape)
        
        predictions = self.anomaly_model.predict(sequences_scaled)
        mse = np.mean(np.power(sequences_scaled - predictions, 2), axis=(1, 2))
        
        anomalies = mse > self.threshold
        return anomalies, mse
    
    def forecast_demand(self, data, sequence_length=48, steps_ahead=24):
        """Forecast future demand"""
        sequences, _ = self.create_sequences(data, sequence_length)
        last_sequence = sequences[-1].reshape(1, sequence_length, 1)
        last_sequence_scaled = self.scaler.transform(last_sequence.reshape(-1, 1)).reshape(last_sequence.shape)
        
        forecasts = []
        current_sequence = last_sequence_scaled.copy()
        
        for _ in range(steps_ahead):
            pred = self.forecast_model.predict(current_sequence, verbose=0)
            forecasts.append(pred[0, 0])
            
            # Update sequence
            current_sequence = np.roll(current_sequence, -1, axis=1)
            current_sequence[0, -1, 0] = pred[0, 0]
        
        # Inverse transform
        forecasts = self.scaler.inverse_transform(np.array(forecasts).reshape(-1, 1)).flatten()
        return forecasts
    
    def visualize_results(self, data, anomalies, forecasts):
        """Visualize anomaly detection and forecasting results"""
        fig = make_subplots(
            rows=2, cols=1,
            subplot_titles=['Anomaly Detection Results', 'Demand Forecasting'],
            vertical_spacing=0.1
        )
        
        # Anomaly detection plot
        fig.add_trace(go.Scatter(x=data['timestamp'][48:], y=data['consumption'][48:], 
                                name='Normal', line=dict(color='blue')), row=1, col=1)
        
        anomaly_indices = np.where(anomalies)[0] + 48
        if len(anomaly_indices) > 0:
            fig.add_trace(go.Scatter(x=data['timestamp'].iloc[anomaly_indices], 
                                    y=data['consumption'].iloc[anomaly_indices],
                                    mode='markers', name='Anomalies', 
                                    marker=dict(color='red', size=8)), row=1, col=1)
        
        # Forecasting plot
        last_timestamp = data['timestamp'].iloc[-1]
        future_timestamps = pd.date_range(last_timestamp, periods=len(forecasts)+1, freq='H')[1:]
        
        fig.add_trace(go.Scatter(x=data['timestamp'][-100:], y=data['consumption'][-100:], 
                                name='Historical', line=dict(color='blue')), row=2, col=1)
        fig.add_trace(go.Scatter(x=future_timestamps, y=forecasts, 
                                name='Forecast', line=dict(color='orange', dash='dash')), row=2, col=1)
        
        fig.update_layout(height=800, title_text="Smart Water Analytics Results")
        fig.show()
    
    def run_complete_analysis(self):
        """Run the complete water analytics pipeline"""
        print("🚀 Starting Smart Water Analytics Pipeline...")
        
        # Load and explore data
        data = self.load_and_explore_data()
        data = self.comprehensive_eda(data)
        
        # Preprocess data
        processed_data = self.advanced_preprocessing(data)
        
        # Train models
        anomaly_history = self.train_anomaly_detection(processed_data)
        forecast_history = self.train_demand_forecasting(processed_data)
        
        # Make predictions
        anomalies, mse_scores = self.detect_anomalies(processed_data)
        forecasts = self.forecast_demand(processed_data)
        
        # Visualize results
        self.visualize_results(processed_data, anomalies, forecasts)
        
        # Performance summary
        print(f"\n📊 Analysis Summary:")
        print(f"🔍 Anomalies detected: {np.sum(anomalies)} out of {len(anomalies)} sequences")
        print(f"📈 Forecast generated for next {len(forecasts)} hours")
        print(f"🎯 Average forecasted consumption: {np.mean(forecasts):.2f}")
        
        return {
            'anomalies': anomalies,
            'forecasts': forecasts,
            'mse_scores': mse_scores,
            'processed_data': processed_data
        }

# Initialize and run analysis
if __name__ == "__main__":
    analytics = WaterAnalytics()
    results = analytics.run_complete_analysis()
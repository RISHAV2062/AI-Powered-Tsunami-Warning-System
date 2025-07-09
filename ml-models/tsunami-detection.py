#!/usr/bin/env python3
"""
Tsunami Detection Model - Advanced ML pipeline for tsunami risk assessment
This module implements a comprehensive machine learning system for detecting
tsunami threats from seismic and oceanographic data.
"""

import os
import sys
import json
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models, callbacks, optimizers
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.feature_selection import SelectKBest, chi2
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import datetime
import logging
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional, Any
import warnings

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('tsunami_detection.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

warnings.filterwarnings('ignore')

@dataclass
class ModelConfig:
    """Configuration class for tsunami detection model"""
    model_name: str = "tsunami_detection_v2.1"
    input_features: int = 15
    hidden_layers: List[int] = None
    dropout_rate: float = 0.3
    learning_rate: float = 0.001
    batch_size: int = 32
    epochs: int = 100
    validation_split: float = 0.2
    early_stopping_patience: int = 10
    reduce_lr_patience: int = 5
    random_state: int = 42
    
    def __post_init__(self):
        if self.hidden_layers is None:
            self.hidden_layers = [128, 64, 32, 16]

class DataProcessor:
    """Data preprocessing and feature engineering for tsunami detection"""
    
    def __init__(self):
        self.scaler = StandardScaler()
        self.min_max_scaler = MinMaxScaler()
        self.label_encoder = LabelEncoder()
        self.feature_names = []
        self.is_fitted = False
        
    def load_seismic_data(self, file_path: str) -> pd.DataFrame:
        """Load seismic data from various sources"""
        try:
            if file_path.endswith('.csv'):
                data = pd.read_csv(file_path)
            elif file_path.endswith('.json'):
                data = pd.read_json(file_path)
            else:
                raise ValueError(f"Unsupported file format: {file_path}")
            
            logger.info(f"Loaded {len(data)} seismic records from {file_path}")
            return data
        except Exception as e:
            logger.error(f"Error loading data: {str(e)}")
            raise
    
    def generate_synthetic_data(self, n_samples: int = 10000) -> pd.DataFrame:
        """Generate synthetic seismic data for training"""
        np.random.seed(42)
        
        # Generate base features
        magnitude = np.random.uniform(3.0, 9.0, n_samples)
        depth = np.random.uniform(0, 700, n_samples)
        latitude = np.random.uniform(-90, 90, n_samples)
        longitude = np.random.uniform(-180, 180, n_samples)
        
        # Generate derived features
        is_shallow = (depth < 70).astype(int)
        is_strong = (magnitude > 6.0).astype(int)
        is_coastal = self._is_coastal(latitude, longitude)
        
        # Generate oceanographic features
        sea_level = np.random.normal(0, 0.5, n_samples)
        wave_height = np.random.exponential(2.0, n_samples)
        tide_level = np.random.uniform(-2, 2, n_samples)
        
        # Generate seismic features
        p_wave_velocity = np.random.uniform(6.0, 8.5, n_samples)
        s_wave_velocity = np.random.uniform(3.0, 5.0, n_samples)
        focal_mechanism = np.random.uniform(0, 360, n_samples)
        
        # Generate environmental features
        bathymetry = np.random.uniform(-11000, 0, n_samples)
        sediment_thickness = np.random.uniform(0, 10000, n_samples)
        crustal_age = np.random.uniform(0, 200, n_samples)
        
        # Calculate tsunami risk based on complex rules
        tsunami_risk = self._calculate_tsunami_risk(
            magnitude, depth, is_coastal, wave_height, bathymetry
        )
        
        data = pd.DataFrame({
            'magnitude': magnitude,
            'depth': depth,
            'latitude': latitude,
            'longitude': longitude,
            'is_shallow': is_shallow,
            'is_strong': is_strong,
            'is_coastal': is_coastal,
            'sea_level': sea_level,
            'wave_height': wave_height,
            'tide_level': tide_level,
            'p_wave_velocity': p_wave_velocity,
            's_wave_velocity': s_wave_velocity,
            'focal_mechanism': focal_mechanism,
            'bathymetry': bathymetry,
            'sediment_thickness': sediment_thickness,
            'crustal_age': crustal_age,
            'tsunami_risk': tsunami_risk
        })
        
        logger.info(f"Generated {n_samples} synthetic seismic records")
        return data
    
    def _is_coastal(self, lat: np.ndarray, lng: np.ndarray) -> np.ndarray:
        """Determine if coordinates are in coastal regions"""
        # Simplified coastal detection using known high-risk areas
        coastal_regions = [
            (35.0, 139.0, 500),  # Japan
            (40.0, -125.0, 300), # US West Coast
            (-10.0, 110.0, 400), # Indonesia
            (36.0, 28.0, 200),   # Greece/Turkey
            (-40.0, 175.0, 300), # New Zealand
        ]
        
        is_coastal = np.zeros(len(lat), dtype=int)
        
        for region_lat, region_lng, radius_km in coastal_regions:
            # Simple distance calculation (not accurate for large distances)
            distance = np.sqrt((lat - region_lat)**2 + (lng - region_lng)**2) * 111  # km
            is_coastal |= (distance < radius_km).astype(int)
        
        return is_coastal
    
    def _calculate_tsunami_risk(self, magnitude: np.ndarray, depth: np.ndarray, 
                              is_coastal: np.ndarray, wave_height: np.ndarray,
                              bathymetry: np.ndarray) -> np.ndarray:
        """Calculate tsunami risk based on seismic and oceanographic parameters"""
        risk_score = np.zeros(len(magnitude))
        
        # Magnitude contribution (0-40 points)
        risk_score += np.clip((magnitude - 6.0) * 10, 0, 40)
        
        # Depth contribution (0-20 points, shallow earthquakes more dangerous)
        risk_score += np.clip((100 - depth) / 5, 0, 20)
        
        # Coastal proximity (0-25 points)
        risk_score += is_coastal * 25
        
        # Wave height contribution (0-10 points)
        risk_score += np.clip(wave_height * 2, 0, 10)
        
        # Bathymetry contribution (0-5 points, deeper water can amplify)
        risk_score += np.clip(abs(bathymetry) / 2000, 0, 5)
        
        # Normalize to 0-1 range
        risk_score = risk_score / 100
        
        # Convert to categorical labels
        risk_labels = np.zeros(len(risk_score), dtype=int)
        risk_labels[risk_score >= 0.75] = 3  # Critical
        risk_labels[(risk_score >= 0.5) & (risk_score < 0.75)] = 2  # High
        risk_labels[(risk_score >= 0.25) & (risk_score < 0.5)] = 1  # Medium
        # risk_labels < 0.25 remains 0 (Low)
        
        return risk_labels
    
    def engineer_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Engineer additional features for better model performance"""
        try:
            # Make a copy to avoid modifying original data
            df = data.copy()
            
            # Distance-based features
            df['distance_to_coast'] = np.sqrt(df['latitude']**2 + df['longitude']**2)
            df['magnitude_depth_ratio'] = df['magnitude'] / (df['depth'] + 1)
            
            # Seismic moment calculation
            df['seismic_moment'] = 10**(1.5 * df['magnitude'] + 9.1)
            
            # Energy calculations
            df['seismic_energy'] = 10**(1.5 * df['magnitude'] + 4.8)
            
            # Interaction features
            df['magnitude_x_shallow'] = df['magnitude'] * df['is_shallow']
            df['magnitude_x_coastal'] = df['magnitude'] * df['is_coastal']
            df['depth_x_coastal'] = df['depth'] * df['is_coastal']
            
            # Temporal features (if timestamp available)
            if 'timestamp' in df.columns:
                df['timestamp'] = pd.to_datetime(df['timestamp'])
                df['hour'] = df['timestamp'].dt.hour
                df['day_of_week'] = df['timestamp'].dt.dayofweek
                df['month'] = df['timestamp'].dt.month
                df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
            
            # Statistical features for waveform data (if available)
            if 'waveform_data' in df.columns:
                df['waveform_max'] = df['waveform_data'].apply(
                    lambda x: max(x) if isinstance(x, list) and len(x) > 0 else 0
                )
                df['waveform_min'] = df['waveform_data'].apply(
                    lambda x: min(x) if isinstance(x, list) and len(x) > 0 else 0
                )
                df['waveform_mean'] = df['waveform_data'].apply(
                    lambda x: np.mean(x) if isinstance(x, list) and len(x) > 0 else 0
                )
                df['waveform_std'] = df['waveform_data'].apply(
                    lambda x: np.std(x) if isinstance(x, list) and len(x) > 0 else 0
                )
            
            # Remove non-numeric columns for model training
            numeric_columns = df.select_dtypes(include=[np.number]).columns
            df = df[numeric_columns]
            
            logger.info(f"Engineered features. Shape: {df.shape}")
            return df
            
        except Exception as e:
            logger.error(f"Error in feature engineering: {str(e)}")
            raise
    
    def preprocess_data(self, data: pd.DataFrame, target_column: str = 'tsunami_risk') -> Tuple[np.ndarray, np.ndarray]:
        """Preprocess data for model training"""
        try:
            # Separate features and target
            if target_column not in data.columns:
                raise ValueError(f"Target column '{target_column}' not found in data")
            
            X = data.drop(target_column, axis=1)
            y = data[target_column]
            
            # Store feature names
            self.feature_names = X.columns.tolist()
            
            # Handle missing values
            X = X.fillna(X.mean())
            
            # Scale features
            if not self.is_fitted:
                X_scaled = self.scaler.fit_transform(X)
                self.is_fitted = True
            else:
                X_scaled = self.scaler.transform(X)
            
            # Encode labels
            y_encoded = self.label_encoder.fit_transform(y)
            
            logger.info(f"Preprocessed data. Features: {X_scaled.shape}, Labels: {y_encoded.shape}")
            return X_scaled, y_encoded
            
        except Exception as e:
            logger.error(f"Error in data preprocessing: {str(e)}")
            raise
    
    def save_preprocessor(self, file_path: str):
        """Save preprocessor objects"""
        try:
            preprocessor_data = {
                'scaler': self.scaler,
                'min_max_scaler': self.min_max_scaler,
                'label_encoder': self.label_encoder,
                'feature_names': self.feature_names,
                'is_fitted': self.is_fitted
            }
            joblib.dump(preprocessor_data, file_path)
            logger.info(f"Preprocessor saved to {file_path}")
        except Exception as e:
            logger.error(f"Error saving preprocessor: {str(e)}")
            raise
    
    def load_preprocessor(self, file_path: str):
        """Load preprocessor objects"""
        try:
            preprocessor_data = joblib.load(file_path)
            self.scaler = preprocessor_data['scaler']
            self.min_max_scaler = preprocessor_data['min_max_scaler']
            self.label_encoder = preprocessor_data['label_encoder']
            self.feature_names = preprocessor_data['feature_names']
            self.is_fitted = preprocessor_data['is_fitted']
            logger.info(f"Preprocessor loaded from {file_path}")
        except Exception as e:
            logger.error(f"Error loading preprocessor: {str(e)}")
            raise

class TsunamiDetectionModel:
    """Deep learning model for tsunami detection"""
    
    def __init__(self, config: ModelConfig):
        self.config = config
        self.model = None
        self.training_history = None
        self.ensemble_models = []
        
    def build_model(self, input_shape: int, num_classes: int = 4) -> keras.Model:
        """Build deep neural network model"""
        try:
            model = models.Sequential([
                layers.Input(shape=(input_shape,)),
                
                # First hidden layer with batch normalization
                layers.Dense(self.config.hidden_layers[0], activation='relu'),
                layers.BatchNormalization(),
                layers.Dropout(self.config.dropout_rate),
                
                # Second hidden layer
                layers.Dense(self.config.hidden_layers[1], activation='relu'),
                layers.BatchNormalization(),
                layers.Dropout(self.config.dropout_rate),
                
                # Third hidden layer
                layers.Dense(self.config.hidden_layers[2], activation='relu'),
                layers.BatchNormalization(),
                layers.Dropout(self.config.dropout_rate),
                
                # Fourth hidden layer
                layers.Dense(self.config.hidden_layers[3], activation='relu'),
                layers.BatchNormalization(),
                layers.Dropout(self.config.dropout_rate),
                
                # Output layer
                layers.Dense(num_classes, activation='softmax')
            ])
            
            # Compile model
            model.compile(
                optimizer=optimizers.Adam(learning_rate=self.config.learning_rate),
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy', 'precision', 'recall']
            )
            
            logger.info(f"Built model with {model.count_params()} parameters")
            return model
            
        except Exception as e:
            logger.error(f"Error building model: {str(e)}")
            raise
    
    def train_model(self, X_train: np.ndarray, y_train: np.ndarray,
                   X_val: np.ndarray = None, y_val: np.ndarray = None) -> keras.Model:
        """Train the tsunami detection model"""
        try:
            # Build model
            self.model = self.build_model(X_train.shape[1], len(np.unique(y_train)))
            
            # Prepare validation data
            if X_val is None or y_val is None:
                X_train, X_val, y_train, y_val = train_test_split(
                    X_train, y_train, test_size=self.config.validation_split,
                    random_state=self.config.random_state, stratify=y_train
                )
            
            # Callbacks
            callbacks_list = [
                callbacks.EarlyStopping(
                    monitor='val_loss',
                    patience=self.config.early_stopping_patience,
                    restore_best_weights=True
                ),
                callbacks.ReduceLROnPlateau(
                    monitor='val_loss',
                    factor=0.5,
                    patience=self.config.reduce_lr_patience,
                    min_lr=1e-7
                ),
                callbacks.ModelCheckpoint(
                    'best_model.h5',
                    monitor='val_loss',
                    save_best_only=True,
                    save_weights_only=False
                )
            ]
            
            # Train model
            logger.info("Starting model training...")
            self.training_history = self.model.fit(
                X_train, y_train,
                validation_data=(X_val, y_val),
                epochs=self.config.epochs,
                batch_size=self.config.batch_size,
                callbacks=callbacks_list,
                verbose=1
            )
            
            logger.info("Model training completed")
            return self.model
            
        except Exception as e:
            logger.error(f"Error training model: {str(e)}")
            raise
    
    def build_ensemble(self, X_train: np.ndarray, y_train: np.ndarray):
        """Build ensemble of models for improved performance"""
        try:
            logger.info("Building ensemble models...")
            
            # Random Forest
            rf_model = RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                random_state=self.config.random_state
            )
            rf_model.fit(X_train, y_train)
            self.ensemble_models.append(('RandomForest', rf_model))
            
            # Gradient Boosting
            gb_model = GradientBoostingClassifier(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=6,
                random_state=self.config.random_state
            )
            gb_model.fit(X_train, y_train)
            self.ensemble_models.append(('GradientBoosting', gb_model))
            
            # Neural Network (different architecture)
            nn_model = models.Sequential([
                layers.Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
                layers.Dropout(0.3),
                layers.Dense(32, activation='relu'),
                layers.Dropout(0.3),
                layers.Dense(len(np.unique(y_train)), activation='softmax')
            ])
            
            nn_model.compile(
                optimizer='adam',
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy']
            )
            
            nn_model.fit(X_train, y_train, epochs=50, batch_size=32, verbose=0)
            self.ensemble_models.append(('NeuralNetwork', nn_model))
            
            logger.info(f"Built ensemble with {len(self.ensemble_models)} models")
            
        except Exception as e:
            logger.error(f"Error building ensemble: {str(e)}")
            raise
    
    def predict_ensemble(self, X: np.ndarray) -> np.ndarray:
        """Make predictions using ensemble of models"""
        try:
            predictions = []
            
            for name, model in self.ensemble_models:
                if name == 'NeuralNetwork':
                    pred = model.predict(X)
                    pred = np.argmax(pred, axis=1)
                else:
                    pred = model.predict(X)
                predictions.append(pred)
            
            # Voting classifier (majority vote)
            ensemble_predictions = np.array(predictions).T
            final_predictions = np.apply_along_axis(
                lambda x: np.bincount(x).argmax(), 
                axis=1, 
                arr=ensemble_predictions
            )
            
            return final_predictions
            
        except Exception as e:
            logger.error(f"Error in ensemble prediction: {str(e)}")
            raise
    
    def evaluate_model(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, Any]:
        """Evaluate model performance"""
        try:
            results = {}
            
            # Main model evaluation
            if self.model is not None:
                loss, accuracy, precision, recall = self.model.evaluate(X_test, y_test, verbose=0)
                y_pred = np.argmax(self.model.predict(X_test), axis=1)
                
                results['main_model'] = {
                    'loss': loss,
                    'accuracy': accuracy,
                    'precision': precision,
                    'recall': recall,
                    'classification_report': classification_report(y_test, y_pred),
                    'confusion_matrix': confusion_matrix(y_test, y_pred).tolist()
                }
            
            # Ensemble evaluation
            if self.ensemble_models:
                y_pred_ensemble = self.predict_ensemble(X_test)
                ensemble_accuracy = np.mean(y_pred_ensemble == y_test)
                
                results['ensemble'] = {
                    'accuracy': ensemble_accuracy,
                    'classification_report': classification_report(y_test, y_pred_ensemble),
                    'confusion_matrix': confusion_matrix(y_test, y_pred_ensemble).tolist()
                }
            
            logger.info("Model evaluation completed")
            return results
            
        except Exception as e:
            logger.error(f"Error evaluating model: {str(e)}")
            raise
    
    def save_model(self, model_path: str, weights_path: str = None):
        """Save trained model"""
        try:
            if self.model is not None:
                self.model.save(model_path)
                logger.info(f"Model saved to {model_path}")
                
                if weights_path:
                    self.model.save_weights(weights_path)
                    logger.info(f"Model weights saved to {weights_path}")
            
            # Save ensemble models
            if self.ensemble_models:
                ensemble_path = model_path.replace('.h5', '_ensemble.pkl')
                joblib.dump(self.ensemble_models, ensemble_path)
                logger.info(f"Ensemble models saved to {ensemble_path}")
                
        except Exception as e:
            logger.error(f"Error saving model: {str(e)}")
            raise
    
    def load_model(self, model_path: str):
        """Load trained model"""
        try:
            self.model = keras.models.load_model(model_path)
            logger.info(f"Model loaded from {model_path}")
            
            # Load ensemble models
            ensemble_path = model_path.replace('.h5', '_ensemble.pkl')
            if os.path.exists(ensemble_path):
                self.ensemble_models = joblib.load(ensemble_path)
                logger.info(f"Ensemble models loaded from {ensemble_path}")
                
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            raise
    
    def predict_tsunami_risk(self, features: np.ndarray) -> Dict[str, Any]:
        """Predict tsunami risk for new data"""
        try:
            if self.model is None:
                raise ValueError("Model not trained or loaded")
            
            # Main model prediction
            prediction_probs = self.model.predict(features)
            prediction_class = np.argmax(prediction_probs, axis=1)
            
            # Ensemble prediction if available
            ensemble_prediction = None
            if self.ensemble_models:
                ensemble_prediction = self.predict_ensemble(features)
            
            risk_levels = ['Low', 'Medium', 'High', 'Critical']
            
            results = {
                'risk_level': risk_levels[prediction_class[0]],
                'confidence': float(np.max(prediction_probs)),
                'probabilities': {
                    risk_levels[i]: float(prediction_probs[0][i]) 
                    for i in range(len(risk_levels))
                },
                'ensemble_prediction': (
                    risk_levels[ensemble_prediction[0]] 
                    if ensemble_prediction is not None else None
                ),
                'timestamp': datetime.datetime.now().isoformat()
            }
            
            return results
            
        except Exception as e:
            logger.error(f"Error predicting tsunami risk: {str(e)}")
            raise
    
    def plot_training_history(self, save_path: str = None):
        """Plot training history"""
        try:
            if self.training_history is None:
                logger.warning("No training history available")
                return
            
            history = self.training_history.history
            
            plt.figure(figsize=(15, 5))
            
            # Plot training & validation accuracy
            plt.subplot(1, 3, 1)
            plt.plot(history['accuracy'], label='Training Accuracy')
            plt.plot(history['val_accuracy'], label='Validation Accuracy')
            plt.title('Model Accuracy')
            plt.xlabel('Epoch')
            plt.ylabel('Accuracy')
            plt.legend()
            
            # Plot training & validation loss
            plt.subplot(1, 3, 2)
            plt.plot(history['loss'], label='Training Loss')
            plt.plot(history['val_loss'], label='Validation Loss')
            plt.title('Model Loss')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.legend()
            
            # Plot precision and recall
            plt.subplot(1, 3, 3)
            plt.plot(history['precision'], label='Training Precision')
            plt.plot(history['val_precision'], label='Validation Precision')
            plt.plot(history['recall'], label='Training Recall')
            plt.plot(history['val_recall'], label='Validation Recall')
            plt.title('Model Precision & Recall')
            plt.xlabel('Epoch')
            plt.ylabel('Score')
            plt.legend()
            
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                logger.info(f"Training history plot saved to {save_path}")
            
            plt.show()
            
        except Exception as e:
            logger.error(f"Error plotting training history: {str(e)}")
            raise

class ModelTrainingPipeline:
    """Complete training pipeline for tsunami detection model"""
    
    def __init__(self, config: ModelConfig):
        self.config = config
        self.processor = DataProcessor()
        self.model = TsunamiDetectionModel(config)
        
    def run_training_pipeline(self, data_path: str = None, use_synthetic: bool = True):
        """Run complete training pipeline"""
        try:
            logger.info("Starting tsunami detection model training pipeline...")
            
            # Load or generate data
            if use_synthetic or data_path is None:
                logger.info("Generating synthetic training data...")
                raw_data = self.processor.generate_synthetic_data(n_samples=10000)
            else:
                logger.info(f"Loading data from {data_path}...")
                raw_data = self.processor.load_seismic_data(data_path)
            
            # Feature engineering
            logger.info("Engineering features...")
            featured_data = self.processor.engineer_features(raw_data)
            
            # Preprocess data
            logger.info("Preprocessing data...")
            X, y = self.processor.preprocess_data(featured_data)
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=self.config.random_state,
                stratify=y
            )
            
            # Train main model
            logger.info("Training main model...")
            self.model.train_model(X_train, y_train)
            
            # Build ensemble
            logger.info("Building ensemble models...")
            self.model.build_ensemble(X_train, y_train)
            
            # Evaluate models
            logger.info("Evaluating models...")
            results = self.model.evaluate_model(X_test, y_test)
            
            # Save results
            self._save_results(results)
            
            # Save model and preprocessor
            self.model.save_model(f'{self.config.model_name}.h5')
            self.processor.save_preprocessor(f'{self.config.model_name}_preprocessor.pkl')
            
            # Plot training history
            self.model.plot_training_history(f'{self.config.model_name}_training_history.png')
            
            logger.info("Training pipeline completed successfully!")
            return results
            
        except Exception as e:
            logger.error(f"Error in training pipeline: {str(e)}")
            raise
    
    def _save_results(self, results: Dict[str, Any]):
        """Save evaluation results"""
        try:
            results_file = f'{self.config.model_name}_results.json'
            with open(results_file, 'w') as f:
                json.dump(results, f, indent=2, default=str)
            logger.info(f"Results saved to {results_file}")
        except Exception as e:
            logger.error(f"Error saving results: {str(e)}")

def main():
    """Main function to run the tsunami detection model training"""
    try:
        # Configuration
        config = ModelConfig(
            model_name="tsunami_detection_v2.1",
            input_features=15,
            hidden_layers=[128, 64, 32, 16],
            dropout_rate=0.3,
            learning_rate=0.001,
            batch_size=32,
            epochs=100,
            validation_split=0.2,
            early_stopping_patience=10,
            reduce_lr_patience=5
        )
        
        # Run training pipeline
        pipeline = ModelTrainingPipeline(config)
        results = pipeline.run_training_pipeline(use_synthetic=True)
        
        print("\n" + "="*50)
        print("TRAINING RESULTS SUMMARY")
        print("="*50)
        
        if 'main_model' in results:
            main_results = results['main_model']
            print(f"Main Model Accuracy: {main_results['accuracy']:.4f}")
            print(f"Main Model Precision: {main_results['precision']:.4f}")
            print(f"Main Model Recall: {main_results['recall']:.4f}")
        
        if 'ensemble' in results:
            ensemble_results = results['ensemble']
            print(f"Ensemble Accuracy: {ensemble_results['accuracy']:.4f}")
        
        print("\nModel training completed successfully!")
        print("Model files saved:")
        print(f"- {config.model_name}.h5")
        print(f"- {config.model_name}_preprocessor.pkl")
        print(f"- {config.model_name}_ensemble.pkl")
        
    except Exception as e:
        logger.error(f"Error in main function: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()
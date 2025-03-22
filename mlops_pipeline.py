# import os
# import numpy as np
# import pandas as pd
# import shap
# import mlflow
# import mlflow.sklearn
# import matplotlib.pyplot as plt
# import seaborn as sns
# from tensorflow.keras.datasets import fashion_mnist
# from ydata_profiling import ProfileReport
# from sklearn.decomposition import PCA
# from sklearn.feature_selection import VarianceThreshold
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import StandardScaler
# from tpot import TPOTClassifier
# from scipy.stats import ks_2samp

# # Set MLflow tracking URI
# mlflow.set_tracking_uri("http://localhost:5000")  # Change if using remote MLflow server

# # Load Fashion MNIST dataset
# (x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

# # Flatten images for EDA
# x_train_flat = x_train.reshape(x_train.shape[0], -1)
# x_test_flat = x_test.reshape(x_test.shape[0], -1)

# # Convert to DataFrame for EDA
# df_train = pd.DataFrame(x_train_flat)
# df_train['label'] = y_train

# # Generate EDA report
# def perform_eda():
#     profile = ProfileReport(df_train, title="Fashion MNIST EDA", explorative=True, minimal=True)
#     profile.to_file("fashion_mnist_eda.html")

#     plt.figure(figsize=(8, 6))
#     sns.countplot(x=y_train)
#     plt.title("Class Distribution")
#     plt.xlabel("Class Label")
#     plt.ylabel("Count")
#     plt.savefig("class_distribution.png")
#     plt.show()

#     print("EDA report saved as 'fashion_mnist_eda.html'")

# # Feature Engineering: Normalization, Variance Thresholding, PCA
# def preprocess_data(x_train, x_test):
#     # Normalize pixel values
#     x_train = x_train / 255.0
#     x_test = x_test / 255.0

#     # Variance Threshold (removing low-variance features)
#     selector = VarianceThreshold(threshold=0.01)
#     x_train = selector.fit_transform(x_train)
#     x_test = selector.transform(x_test)

#     # Apply PCA (reduce to 50 components)
#     pca = PCA(n_components=50)
#     x_train_pca = pca.fit_transform(x_train)
#     x_test_pca = pca.transform(x_test)

#     return x_train_pca, x_test_pca

# # Train Model using TPOT AutoML with Optimization
# def train_tpot(x_train_flat, y_train):
#     x_train_pca, x_test_pca = preprocess_data(x_train_flat, x_test_flat)

#     # Split training data further for validation
#     X_train, X_val, y_train_split, y_val = train_test_split(x_train_pca, y_train, test_size=0.2, random_state=42)

#     print(X_train.shape)
#     print(X_val.shape)
#     print(y_train_split.shape)
#     print(y_val.shape)

#     with mlflow.start_run():
#         print("Running TPOT AutoML...")
#         tpot = TPOTClassifier(generations=3, population_size=10, random_state=42)
#         tpot.fit(X_train, y_train_split)

#         # Evaluate performance
#         accuracy = tpot.score(X_val, y_val)
#         print(f"TPOT Best Model Accuracy: {accuracy:.4f}")

#         # Log model and metrics to MLflow
#         mlflow.log_metric("tpot_validation_accuracy", accuracy)
#         tpot.export("best_tpot_pipeline.py")
#         mlflow.log_artifact("best_tpot_pipeline.py")

#         return tpot.fitted_pipeline_, x_test_pca  # Return the best model

# # Explainability using SHAP
# def explain_model_shap(model, x_test):
#     explainer = shap.Explainer(model.predict, x_test)
#     shap_values = explainer(x_test[:10])

#     # Global feature importance
#     plt.figure(figsize=(10, 6))
#     shap.summary_plot(shap_values, x_test[:10], feature_names=[f"pixel_{i}" for i in range(100)])
#     plt.savefig("shap_summary_plot.png")
#     mlflow.log_artifact("shap_summary_plot.png")
#     print("SHAP Summary Plot saved as 'shap_summary_plot.png'")

#     # Local explanation for a single instance
#     shap.initjs()
#     instance_idx = np.random.randint(0, len(x_test))
#     shap_plot = shap.force_plot(explainer.expected_value, shap_values[instance_idx], x_test[instance_idx], 
#                                 feature_names=[f"pixel_{i}" for i in range(100)], show=False)
#     shap.save_html("shap_force_plot.html", shap_plot)
#     mlflow.log_artifact("shap_force_plot.html")
#     print("SHAP Force Plot saved as 'shap_force_plot.html'")

# # Data Drift Detection using KS-Test
# def detect_drift(x_train, x_test):
#     drift_scores = [ks_2samp(x_train[:, i], x_test[:, i]).pvalue for i in range(x_train.shape[1])]
#     avg_drift_score = np.mean(drift_scores)
#     return avg_drift_score

# # Model Monitoring & Performance Tracking
# def log_performance_metrics(x_train, x_test):
#     drift_score = detect_drift(x_train, x_test)
#     with mlflow.start_run():
#         mlflow.log_metric("model_drift_score", drift_score)
#         print(f"Performance metrics logged to MLflow. Drift Score: {drift_score:.4f}")

# # Run pipeline
# if __name__ == "__main__":
#     perform_eda()
    
#     # Train TPOT AutoML model and get best pipeline
#     best_model, x_test_pca = train_tpot(x_train_flat, y_train)
    
#     # Explain the final model using SHAP
#     explain_model_shap(best_model, x_test_pca)
    
#     # Log performance metrics and detect drift
#     log_performance_metrics(x_train_flat, x_test_flat)

import numpy as np
import pandas as pd
from tensorflow.keras.datasets import fashion_mnist
from tpot import TPOTClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
from sklearn.decomposition import PCA
import joblib
from pathlib import Path
import json
import logging
import time
from datetime import datetime
import matplotlib
matplotlib.use('Agg')  # Set non-interactive backend before importing pyplot
import matplotlib.pyplot as plt
import seaborn as sns
import shap
from ydata_profiling import ProfileReport
import mlflow
import mlflow.sklearn
from alibi_detect.cd import TabularDrift
from sklearn.inspection import permutation_importance

class ModelSelectionPipeline:
    def __init__(self):
        # Set matplotlib backend again to ensure it's non-interactive
        plt.switch_backend('Agg')
        
        # Create separate directories for outputs and MLflow
        self.output_dir = Path('model_selection_outputs')
        # self.mlflow_dir = Path('mlflow_outputs')
        self.output_dir.mkdir(exist_ok=True)
        # self.mlflow_dir.mkdir(exist_ok=True)
        
        # Setup logging
        logging.basicConfig(
            filename=self.output_dir / 'model_selection.log',
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
        
        self.class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
                           'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
        
        # Initialize MLflow with separate tracking URI
        # mlflow.set_tracking_uri(f'file://{self.mlflow_dir.absolute()}')
        mlflow.set_tracking_uri("http://localhost:5000") 
        mlflow.set_experiment('fashion_mnist_experiment')
        
        # Dictionary to store all results
        self.results = {
            'eda_results': {},
            'feature_engineering_results': {},
            'model_selection_results': {},
            'drift_detection_results': {}
        }

    def perform_eda(self):
        """Perform Exploratory Data Analysis"""
        self.logger.info("Starting EDA...")
        
        # Plot class distribution
        plt.figure(figsize=(10, 6))
        sns.countplot(data=pd.DataFrame({'label': self.data['y_train']}), x='label')
        plt.title('Class Distribution (Training Set)')
        plt.xlabel('Class')
        plt.xticks(range(10), self.class_names, rotation=45)
        plt.savefig(self.output_dir / 'class_distribution.png')
        plt.close('all')  # Explicitly close all figures
        
        # Plot sample images
        plt.figure(figsize=(15, 8))
        for i in range(10):
            plt.subplot(2, 5, i + 1)
            idx = np.where(self.data['y_train'] == i)[0][0]
            plt.imshow(self.X_train_original[idx], cmap='gray')
            plt.title(self.class_names[i])
            plt.axis('off')
        plt.savefig(self.output_dir / 'sample_images.png')
        plt.close('all')  # Explicitly close all figures
        
        # Save EDA results
        self.results['eda_results'] = {
            'class_distribution': dict(zip(self.class_names, 
                np.bincount(self.data['y_train']).tolist())),
            'image_shape': self.X_train_original.shape[1:],
            'pixel_value_range': [float(self.X_train_original.min()), 
                                float(self.X_train_original.max())]
        }
        
        self.logger.info("EDA completed")

    def extract_statistical_features(self, X):
        """Extract statistical features from images"""
        means = X.mean(axis=(1, 2)).reshape(-1, 1)
        stds = X.std(axis=(1, 2)).reshape(-1, 1)
        medians = np.median(X, axis=(1, 2)).reshape(-1, 1)
        maxs = X.max(axis=(1, 2)).reshape(-1, 1)
        mins = X.min(axis=(1, 2)).reshape(-1, 1)
        return np.hstack([means, stds, medians, maxs, mins])

    def engineer_features(self):
        """Apply feature engineering"""
        self.logger.info("Starting feature engineering...")
        
        # Get indices used in load_and_preprocess_data
        n_samples = 5000  # Same as in load_and_preprocess_data
        indices = np.random.RandomState(42).choice(len(self.X_train_original), n_samples, replace=False)
        
        # Split indices for train and validation
        train_idx, val_idx = train_test_split(indices, test_size=0.2, random_state=42)
        
        # Statistical features
        train_stats = self.extract_statistical_features(
            self.X_train_original[train_idx].reshape(-1, 28, 28))
        val_stats = self.extract_statistical_features(
            self.X_train_original[val_idx].reshape(-1, 28, 28))
        test_stats = self.extract_statistical_features(
            self.X_test_original[:1000].reshape(-1, 28, 28))
        
        # PCA features
        pca = PCA(n_components=50)
        X_train_pca = pca.fit_transform(self.data['X_train'])
        X_val_pca = pca.transform(self.data['X_val'])
        X_test_pca = pca.transform(self.data['X_test'])
        
        # Combine features
        self.data['X_train'] = np.hstack([self.data['X_train'], train_stats, X_train_pca])
        self.data['X_val'] = np.hstack([self.data['X_val'], val_stats, X_val_pca])
        self.data['X_test'] = np.hstack([self.data['X_test'], test_stats, X_test_pca])
        
        # Save feature engineering results
        self.results['feature_engineering_results'] = {
            'n_original_features': self.X_train_original.shape[1] * self.X_train_original.shape[2],
            'n_engineered_features': self.data['X_train'].shape[1],
            'pca_explained_variance_ratio': pca.explained_variance_ratio_.tolist()[:10]
        }
        
        self.logger.info(f"Feature engineering completed. New feature shape: {self.data['X_train'].shape}")

    def load_and_preprocess_data(self):
        """Load and preprocess Fashion MNIST dataset"""
        self.logger.info("Loading and preprocessing data...")
        
        # Load Fashion MNIST
        (self.X_train_original, y_train), (self.X_test_original, y_test) = fashion_mnist.load_data()
        
        # Flatten the images
        X_train_flat = self.X_train_original.reshape(self.X_train_original.shape[0], -1)
        X_test_flat = self.X_test_original.reshape(self.X_test_original.shape[0], -1)
        
        # Take a smaller subset for faster processing
        n_samples = 5000
        
        # Scale the data
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train_flat)
        X_test_scaled = scaler.transform(X_test_flat)
        
        # Take subset of samples
        indices = np.random.choice(len(X_train_scaled), n_samples, replace=False)
        X_train_subset = X_train_scaled[indices]
        y_train_subset = y_train[indices]
        
        # Split training data
        X_train_final, X_val, y_train_final, y_val = train_test_split(
            X_train_subset, y_train_subset, test_size=0.2, random_state=42
        )
        
        self.data = {
            'X_train': X_train_final,
            'X_val': X_val,
            'X_test': X_test_scaled[:1000],
            'y_train': y_train_final,
            'y_val': y_val,
            'y_test': y_test[:1000]
        }
        
        self.logger.info(f"Data preprocessed. Training set shape: {X_train_final.shape}")

    def run_automl(self):
        """Run TPOT AutoML to find the best model"""
        self.logger.info("Starting AutoML with TPOT...")

        # TPOT Configuration
        tpot = TPOTClassifier(
            generations=3,           # Number of generations to evolve
            population_size=10,      # Number of models in each generation
            # cv=3,                    # Cross-validation folds
            random_state=42,
            # verbosity=2,
            n_jobs=1,               # Use all available CPU cores
            # max_time_mins=10,        # Max execution time in minutes
        )
        
        # Remove constant features
        feature_var = np.var(self.data['X_train'], axis=0)
        non_constant_features = feature_var > 0
        X_train_filtered = self.data['X_train'][:, non_constant_features]
        X_val_filtered = self.data['X_val'][:, non_constant_features]
        
        # Fit TPOT
        start_time = time.time()
        tpot.fit(X_train_filtered, self.data['y_train'])
        end_time = time.time()
        
        # Get validation score using fitted pipeline
        val_score = tpot.fitted_pipeline_.score(X_val_filtered, self.data['y_val'])
        
        # Save TPOT results
        self.results['model_selection_results'] = {
            'best_score': val_score,
            'best_pipeline': str(tpot.fitted_pipeline_),
            'runtime_seconds': end_time - start_time
        }
        
        # Export the best pipeline
        pipeline_file = self.output_dir / 'tpot_best_pipeline.py'
        with open(pipeline_file, 'w') as f:
            f.write(f"# TPOT Best Pipeline\n")
            f.write(f"# Score: {val_score:.4f}\n")
            f.write(f"# Runtime: {end_time - start_time:.2f} seconds\n\n")
            f.write(f"best_pipeline = {str(tpot.fitted_pipeline_)}\n")
        
        self.logger.info(f"AutoML completed. Best score: {val_score:.4f}")
        
        return tpot.fitted_pipeline_

    def generate_eda_report(self):
        """Generate automated EDA report using ydata-profiling"""
        self.logger.info("Generating EDA report...")
        
        # Create DataFrame for analysis
        df_train = pd.DataFrame(self.data['X_train'])
        df_train['label'] = self.data['y_train']
        df_train['class_name'] = [self.class_names[y] for y in self.data['y_train']]
        
        # Generate profile report
        profile = ProfileReport(
            df_train,
            title="Fashion MNIST Analysis",
            minimal=True
        )
        profile.to_file(self.output_dir / "eda_report.html")
        
        # Basic visualizations
        self.perform_eda()
        
        self.logger.info("EDA report generated successfully")

    def explain_features(self, model, X_sample):
        """Generate feature importance explanations using permutation importance"""
        self.logger.info("Generating feature explanations...")
        
        try:
            # Get the same feature set used for training
            feature_var = np.var(self.data['X_train'], axis=0)
            non_constant_features = feature_var > 0
            X_sample_filtered = X_sample[:, non_constant_features]
            
            # Calculate permutation importance
            result = permutation_importance(
                model, 
                X_sample_filtered, 
                self.data['y_train'][:len(X_sample_filtered)],
                n_repeats=5,
                random_state=42,
                n_jobs=1
            )
            
            # Get feature importances
            importances = result.importances_mean
            
            # Plot feature importance
            plt.figure(figsize=(10, 6))
            sorted_idx = np.argsort(importances)
            pos = np.arange(sorted_idx.shape[0]) + .5
            
            # Plot top 20 features (or all if less than 20)
            n_features = min(20, len(importances))
            plt.barh(pos[-n_features:], importances[sorted_idx][-n_features:])
            feature_names = [f'Feature {i}' for i in range(X_sample_filtered.shape[1])]
            plt.yticks(pos[-n_features:], np.array(feature_names)[sorted_idx][-n_features:])
            plt.xlabel("Permutation Importance")
            plt.title('Feature Importance')
            plt.tight_layout()
            plt.savefig(self.output_dir / 'feature_importance.png')
            plt.close()
            
            # Save importance values
            importance_dict = {
                feature_names[i]: float(importances[i])
                for i in sorted_idx[-n_features:]
            }
            
            with open(self.output_dir / 'feature_importance.json', 'w') as f:
                json.dump(importance_dict, f, indent=4)
            
            self.logger.info("Feature importance analysis completed successfully")
            return importance_dict
            
        except Exception as e:
            self.logger.error(f"Error in feature importance analysis: {str(e)}")
            return None

    def detect_drift(self, reference_data, current_data):
        """Detect data drift between reference and current data"""
        self.logger.info("Performing drift detection...")
        
        try:
            # Remove constant features
            feature_var = np.var(reference_data, axis=0)
            non_constant_features = feature_var > 0
            reference_filtered = reference_data[:, non_constant_features]
            current_filtered = current_data[:, non_constant_features]
            
            # Initialize drift detector with proper configuration
            drift_detector = TabularDrift(
                reference_filtered,
                p_val=.05,
                preprocess_fn=None,
                backend='pytorch',  # Use PyTorch backend
                categories_per_feature={},  # Specify all features as numerical
                n_features=reference_filtered.shape[1]
            )
            
            # Predict drift
            drift_preds = drift_detector.predict(current_filtered)
            
            # Extract drift results safely
            is_drift = drift_preds.get('data', {}).get('is_drift', False)
            p_value = drift_preds.get('data', {}).get('p_val', [0.0])[0]  # Get first p-value if array
            
            # Save drift results
            self.results['drift_detection_results'] = {
                'drift_detected': bool(is_drift),
                'p_value': float(p_value),
                'threshold': 0.05,
                'n_features_analyzed': reference_filtered.shape[1]
            }
            
            self.logger.info(f"Drift detection completed. Drift detected: {is_drift}")
            return self.results['drift_detection_results']
            
        except Exception as e:
            self.logger.error(f"Error in drift detection: {str(e)}")
            # Return default results if error occurs
            return {
                'drift_detected': False,
                'p_value': 1.0,
                'threshold': 0.05,
                'error': str(e)
            }

    def run_pipeline(self):
        """Run the complete pipeline"""
        self.logger.info("Starting pipeline...")
        
        with mlflow.start_run():
            try:
                # Load and preprocess data
                self.load_and_preprocess_data()
                
                # Generate EDA report
                self.generate_eda_report()
                
                # Engineer features
                self.engineer_features()
                
                # Run model selection
                best_model = self.run_automl()
                
                # Generate feature explanations with smaller sample
                sample_size = min(100, len(self.data['X_train']))
                importance_dict = self.explain_features(
                    best_model,
                    self.data['X_train'][:sample_size]
                )
                
                # Detect drift with smaller samples
                n_drift_samples = min(1000, len(self.data['X_train']))
                drift_results = self.detect_drift(
                    self.data['X_train'][:n_drift_samples],
                    self.data['X_test'][:n_drift_samples]
                )
                
                # Log metrics with MLflow
                mlflow.log_metrics({
                    'accuracy': self.results['model_selection_results']['best_score'],
                    'drift_p_value': drift_results['p_value']
                })
                
                # Log individual artifacts instead of entire directory
                mlflow.log_artifact(self.output_dir / 'model_selection.log')
                mlflow.log_artifact(self.output_dir / 'feature_importance.png')
                mlflow.log_artifact(self.output_dir / 'feature_importance.json')
                mlflow.log_artifact(self.output_dir / 'class_distribution.png')
                
                # Save final model
                mlflow.sklearn.log_model(best_model, "model")
                
                # Save results to JSON
                with open(self.output_dir / 'results.json', 'w') as f:
                    json.dump(self.results, f, indent=4)
                mlflow.log_artifact(self.output_dir / 'results.json')
                
            except Exception as e:
                self.logger.error(f"Pipeline error: {str(e)}")
                raise
            
        self.logger.info("Pipeline completed successfully!")
        return best_model

if __name__ == "__main__":
    pipeline = ModelSelectionPipeline()
    final_model = pipeline.run_pipeline()

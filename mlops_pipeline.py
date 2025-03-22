import os
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'

import numpy as np
import pandas as pd
from tensorflow.keras.datasets import fashion_mnist
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import autokeras as ak
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
import joblib
from ydata_profiling import ProfileReport
from pathlib import Path
import logging
import mlflow
import mlflow.keras
import shap
import optuna
from alibi_detect.cd import TabularDrift
import json
from datetime import datetime
from sklearn.decomposition import PCA
from tensorflow.keras.utils import to_categorical
import platform
from tensorflow.keras.models import model_from_json
from tensorflow.keras.models import load_model
import tensorflow as tf
from scipy import stats


class MLOpsPipeline:
    def __init__(self):
        self.class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
                           'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
        self.model = None
        self.scaler = StandardScaler()
        
        # Setup directories
        self.output_dir = Path('mlops_outputs')
        self.output_dir.mkdir(exist_ok=True)
        
        # Setup logging
        logging.basicConfig(
            filename=self.output_dir / 'pipeline.log',
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
        
        # Setup MLflow
        mlflow.set_tracking_uri("http://localhost:5000")
        mlflow.set_experiment('fashion_mnist_experiment')

    def load_and_preprocess_data(self):
        """Load and preprocess Fashion MNIST data"""
        self.logger.info("Loading and preprocessing data...")
        
        # Load Fashion MNIST dataset
        (X_train, y_train), (X_test, y_test) = fashion_mnist.load_data()
        
        # Store original images
        self.X_train_original = X_train
        
        # Flatten and scale
        X_train_flat = X_train.reshape(X_train.shape[0], -1)
        X_test_flat = X_test.reshape(X_test.shape[0], -1)
        
        # Take subset for faster processing
        # if n_samples and n_samples < len(X_train_flat):
        #     idx = np.random.RandomState(42).choice(len(X_train_flat), n_samples, replace=False)
        #     X_train_flat = X_train_flat[idx]
        #     y_train = y_train[idx]
        #     self.X_train_original = self.X_train_original[idx]
        
        # Scale the data
        X_train_scaled = self.scaler.fit_transform(X_train_flat)
        X_test_scaled = self.scaler.transform(X_test_flat)
        
        # Split into train and validation sets
        X_train_final, X_val, y_train_final, y_val = train_test_split(
            X_train_scaled, y_train, test_size=0.2, random_state=42
        )
        
        self.data = {
            'X_train': X_train_final,
            'y_train': y_train_final,
            'X_val': X_val,
            'y_val': y_val,
            'X_test': X_test_scaled,
            'y_test': y_test
        }
        
        return self.data

    def perform_eda(self):
        """M1: Exploratory Data Analysis"""
        self.logger.info("Starting EDA...")
        
        # Create DataFrame for analysis
        df_train = pd.DataFrame(self.data['X_train'])
        df_train['label'] = self.data['y_train']
        df_train['class_name'] = [self.class_names[y] for y in self.data['y_train']]
        
        # Generate YData Profile Report
        profile = ProfileReport(
            df_train,
            title="Fashion MNIST Analysis",
            minimal=True
        )
        profile.to_file(self.output_dir / "eda_report.html")
        
        # Create basic visualizations
        self._create_visualizations()
        
        self.logger.info("EDA completed successfully")

    def _create_visualizations(self):
        """Create and save basic visualizations"""
        # Class distribution
        plt.figure(figsize=(10, 6))
        sns.countplot(data=pd.DataFrame({'label': self.data['y_train']}), x='label')
        plt.title('Class Distribution')
        plt.savefig(self.output_dir / 'class_distribution.png')
        plt.close()
        
        # Sample images
        fig, axes = plt.subplots(2, 5, figsize=(15, 6))
        for i, ax in enumerate(axes.flat):
            idx = np.where(self.data['y_train'] == i)[0][0]
            ax.imshow(self.X_train_original[idx], cmap='gray')
            ax.set_title(self.class_names[i])
            ax.axis('off')
        plt.savefig(self.output_dir / 'sample_images.png')
        plt.close()

    def engineer_features(self):
        """M2: Feature Engineering Pipeline"""
        self.logger.info("Starting feature engineering...")
        
        # Initialize PCA
        self.pca = PCA(n_components=0.95)  # Preserve 95% of variance
        
        # Apply PCA transformation
        X_train_pca = self.pca.fit_transform(self.data['X_train'])
        X_val_pca = self.pca.transform(self.data['X_val'])
        X_test_pca = self.pca.transform(self.data['X_test'])
        
        # Store transformed data
        self.data['X_train_pca'] = X_train_pca
        self.data['X_val_pca'] = X_val_pca
        self.data['X_test_pca'] = X_test_pca
        
        # Log transformation details
        self.logger.info(f"Number of components selected by PCA: {X_train_pca.shape[1]}")
        self.logger.info(f"Explained variance ratio: {sum(self.pca.explained_variance_ratio_):.4f}")
        
        # Analyze PCA components
        self._analyze_pca_components()
        
        return self.data

    def _analyze_pca_components(self):
        """Analyze and visualize PCA components"""
        # Plot explained variance ratio
        plt.figure(figsize=(10, 6))
        plt.plot(np.cumsum(self.pca.explained_variance_ratio_))
        plt.xlabel('Number of Components')
        plt.ylabel('Cumulative Explained Variance Ratio')
        plt.title('Explained Variance Ratio vs Number of Components')
        plt.savefig(self.output_dir / 'pca_explained_variance.png')
        plt.close()
        
        # Save component analysis
        component_importance = pd.DataFrame({
            'Component': range(1, len(self.pca.explained_variance_ratio_) + 1),
            'Explained_Variance_Ratio': self.pca.explained_variance_ratio_,
            'Cumulative_Variance_Ratio': np.cumsum(self.pca.explained_variance_ratio_)
        })
        component_importance.to_csv(self.output_dir / 'pca_component_importance.csv', index=False)

    def explain_features(self):
        """M2: Feature Explainability Analysis"""
        self.logger.info("Generating feature explanations...")
        
        try:
            # Create a function that returns the model's prediction probabilities
            def model_predict(X):
                return self.model.predict(X.astype(np.float32))
            
            # Create background data for SHAP
            background_data = shap.sample(self.data['X_train_pca'], 50)
            
            # Create KernelExplainer
            explainer = shap.KernelExplainer(
                model_predict,
                background_data,
                link="identity"
            )
            
            # Calculate SHAP values for a subset of validation data
            n_explain = min(50, len(self.data['X_val_pca']))
            shap_values = explainer.shap_values(
                self.data['X_val_pca'][:n_explain],
                n_samples=50
            )
            
            # Generate and save SHAP plots
            self._save_shap_plots(shap_values, self.data['X_val_pca'][:n_explain])
            
            # Generate feature importance report
            self._generate_feature_report(shap_values)
            
            return shap_values
            
        except Exception as e:
            self.logger.error(f"Error in feature explanation: {str(e)}")
            raise

    def _save_shap_plots(self, shap_values, X_val):
        """Generate and save SHAP visualization plots"""
        try:
            # 1. Summary plot for all classes
            plt.figure(figsize=(12, 8))
            # If shap_values is a list (one per class), use the first class
            values = shap_values[0] if isinstance(shap_values, list) else shap_values
            shap.summary_plot(
                values,
                X_val,
                plot_type="bar",
                show=False
            )
            plt.tight_layout()
            plt.savefig(self.output_dir / 'shap_summary_all_classes.png')
            plt.close()
            
            # 2. Feature importance plot
            plt.figure(figsize=(12, 8))
            shap.summary_plot(
                values,
                X_val,
                plot_type="violin",
                show=False
            )
            plt.tight_layout()
            plt.savefig(self.output_dir / 'shap_feature_importance.png')
            plt.close()
            
            # 3. Class-specific plots (if we have multiple classes)
            if isinstance(shap_values, list):
                for i, class_values in enumerate(shap_values):
                    plt.figure(figsize=(12, 8))
                    shap.summary_plot(
                        class_values,
                        X_val,
                        plot_type="bar",
                        show=False,
                        title=f"Feature Importance for Class {i}"
                    )
                    plt.tight_layout()
                    plt.savefig(self.output_dir / f'shap_summary_class_{i}.png')
                    plt.close()
                
        except Exception as e:
            self.logger.error(f"Error in saving SHAP plots: {str(e)}")
            raise

    def _generate_feature_report(self, shap_values):
        """Generate comprehensive feature importance report"""
        try:
            # Handle both list and single array of shap_values
            if isinstance(shap_values, list):
                global_importance = np.mean([np.abs(sv).mean(0) for sv in shap_values], axis=0)
                class_specific = {
                    f"class_{i}": np.abs(sv).mean(0).tolist()
                    for i, sv in enumerate(shap_values)
                }
            else:
                global_importance = np.abs(shap_values).mean(0)
                class_specific = {"class_0": np.abs(shap_values).mean(0).tolist()}
            
            report = {
                'pca_analysis': {
                    'n_components': self.data['X_val_pca'].shape[1],
                    'total_variance_explained': float(sum(self.pca.explained_variance_ratio_)),
                    'component_importance': self.pca.explained_variance_ratio_.tolist()
                },
                'feature_importance': {
                    'method': 'SHAP',
                    'global_importance': global_importance.tolist(),
                    'class_specific_importance': class_specific
                }
            }
            
            # Save report
            with open(self.output_dir / 'feature_analysis_report.json', 'w') as f:
                json.dump(report, f, indent=4)
            
        except Exception as e:
            self.logger.error(f"Error in generating feature report: {str(e)}")
            raise

    def optimize_hyperparameters(self):
        """M3: Model Selection & Hyperparameter Optimization"""
        def objective(trial):
            try:
                # AutoKeras hyperparameters
                max_trials = trial.suggest_int('max_trials', 1, 2)  # Reduced max_trials
                epochs = trial.suggest_int('epochs', 3, 5)  # Reduced epochs
                batch_size = trial.suggest_int('batch_size', 32, 128, log=True)
                learning_rate = trial.suggest_float('learning_rate', 1e-4, 1e-2, log=True)
                
                # Initialize AutoKeras model with stricter constraints
                input_node = ak.Input(shape=(self.data['X_train_pca'].shape[1],))
                output_node = ak.DenseBlock(
                    num_layers=trial.suggest_int('num_layers', 1, 3),
                    dropout=trial.suggest_float('dropout', 0.1, 0.5)
                )(input_node)
                output_node = ak.ClassificationHead(
                    num_classes=10,
                    dropout=trial.suggest_float('head_dropout', 0.1, 0.5)
                )(output_node)
                
                clf = ak.AutoModel(
                    inputs=input_node,
                    outputs=output_node,
                    max_trials=max_trials,
                    overwrite=True,
                    project_name=f'fashion_mnist_trial_{trial.number}',
                    tuner='greedy'  # Use greedy strategy for faster convergence
                )
                
                # Early stopping callback
                early_stopping = tf.keras.callbacks.EarlyStopping(
                    monitor='val_loss',
                    patience=2,
                    restore_best_weights=True
                )
                
                # Train the model
                clf.fit(
                    self.data['X_train_pca'],
                    self.data['y_train'],
                    epochs=epochs,
                    batch_size=batch_size,
                    validation_data=(self.data['X_val_pca'], self.data['y_val']),
                    callbacks=[early_stopping],
                    verbose=0  # Reduce output verbosity
                )
                
                # Evaluate the model
                score = clf.evaluate(self.data['X_val_pca'], self.data['y_val'])[1]
                
                # Cleanup immediately after evaluation
                import shutil
                project_dir = f'fashion_mnist_trial_{trial.number}'
                if os.path.exists(project_dir):
                    shutil.rmtree(project_dir)
                
                return score
                
            except Exception as e:
                self.logger.error(f"Trial failed with error: {str(e)}")
                return float('-inf')
        
        try:
            # Create study with more focused search space
            study = optuna.create_study(
                direction='maximize',
                sampler=optuna.samplers.TPESampler(seed=42)
            )
            
            # Add time limit and reduced number of trials
            study.optimize(
                objective,
                n_trials=3,  # Reduced number of trials
                timeout=600,  # 10-minute timeout
                catch=(Exception,)
            )
            
            # Log best parameters
            self.logger.info(f"Best trial accuracy: {study.best_value:.4f}")
            self.logger.info(f"Best parameters: {study.best_params}")
            
            return study.best_params, study.best_value
            
        except Exception as e:
            self.logger.error(f"Hyperparameter optimization failed: {str(e)}")
            return {
                'max_trials': 1,
                'epochs': 3,
                'batch_size': 64,
                'learning_rate': 0.001,
                'num_layers': 2,
                'dropout': 0.3,
                'head_dropout': 0.3
            }, None

    def train_model(self, params):
        """Train the model using best parameters"""
        # Initialize AutoKeras model
        input_node = ak.Input(shape=(self.data['X_train_pca'].shape[1],))
        output_node = ak.DenseBlock()(input_node)
        output_node = ak.ClassificationHead(num_classes=10)(output_node)

        clf = ak.AutoModel(
            inputs=input_node,
            outputs=output_node,
            max_trials=params['max_trials'],
            overwrite=True,
            project_name='fashion_mnist_final'
        )
        
        # Train the model on PCA-transformed data
        clf.fit(
            self.data['X_train_pca'],
            self.data['y_train'],
            epochs=params['epochs'],
            validation_data=(self.data['X_val_pca'], self.data['y_val']),
            verbose=1
        )
        
        # Get the best model and save it properly
        self.model = clf.export_model()
        
        return self.model

    def monitor_performance(self):
        """M4: Model Monitoring & Performance Tracking"""
        # Convert labels to one-hot encoded format
        y_train_cat = to_categorical(self.data['y_train'], num_classes=10)
        y_val_cat = to_categorical(self.data['y_val'], num_classes=10)
        
        # Calculate performance metrics
        train_score = self.model.evaluate(self.data['X_train_pca'], y_train_cat)[1]
        val_score = self.model.evaluate(self.data['X_val_pca'], y_val_cat)[1]
        
        # Data Drift Detection using both TabularDrift and KS-test
        from alibi_detect.cd import TabularDrift
        from scipy import stats
        
        # 1. TabularDrift detection
        drift_detector = TabularDrift(
            x_ref=self.data['X_train_pca'],  # Changed from X_ref to x_ref
            p_val=.05,
            # preprocess_kwargs=None,
            # x_ref_preprocessed=True
        )
        
        # Detect drift using TabularDrift
        drift_preds = drift_detector.predict(self.data['X_val_pca'])
        
        # 2. KS-test for each feature
        ks_results = []
        ks_p_values = []
        drifted_features_ks = []
        
        for feature_idx in range(self.data['X_train_pca'].shape[1]):
            ks_statistic, p_value = stats.ks_2samp(
                self.data['X_train_pca'][:, feature_idx],
                self.data['X_val_pca'][:, feature_idx]
            )
            ks_results.append({
                'feature': feature_idx,
                'statistic': float(ks_statistic),
                'p_value': float(p_value),
                'is_drift': p_value < 0.05
            })
            ks_p_values.append(float(p_value))
            if p_value < 0.05:
                drifted_features_ks.append(feature_idx)
        
        # Combine drift detection results
        feature_p_vals = drift_preds['data']['p_val']
        drifted_features_tabular = np.where(feature_p_vals < 0.05)[0]
        
        # Create comprehensive drift report
        drift_report = {
            'tabular_drift': {
                'overall_drift': bool(drift_preds['data']['is_drift']),
                'feature_p_values': [float(p) for p in feature_p_vals],
                'drifted_features': [int(i) for i in drifted_features_tabular],
                'n_drifted_features': int(len(drifted_features_tabular))
            },
            'ks_test': {
                'feature_results': [{
                    'feature': int(result['feature']),
                    'statistic': float(result['statistic']),
                    'p_value': float(result['p_value']),
                    'is_drift': bool(result['is_drift'])
                } for result in ks_results],
                'drifted_features': [int(i) for i in drifted_features_ks],
                'n_drifted_features': int(len(drifted_features_ks)),
                'mean_p_value': float(np.mean(ks_p_values))
            },
            'consensus': {
                'features_agreed_drift': [int(i) for i in list(set(drifted_features_tabular) & set(drifted_features_ks))],
                'n_features_agreed_drift': int(len(set(drifted_features_tabular) & set(drifted_features_ks)))
            }
        }

        # # Save drift report
        with open(self.output_dir / 'drift_report.json', 'w') as f:
            json.dump(drift_report, f, indent=4)
        

        # # Generate timestamp and setup directories
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_dir = self.output_dir / 'models' / timestamp
        run_dir.mkdir(parents=True, exist_ok=True)
        
        # Save model and artifacts
        model_path = run_dir / 'model.keras'  # Add .keras extension
        self.model.save(model_path)
        joblib.dump(self.scaler, run_dir / 'scaler.joblib')
        joblib.dump(self.pca, run_dir / 'pca.joblib')
        
        # Save model info and metadata
        
        # Log to MLflow
        with mlflow.start_run(run_name=f"fashion_mnist_{timestamp}"):
            mlflow.log_metrics({
                'train_accuracy': train_score,
                'val_accuracy': val_score,
                'drift_detected': int(drift_preds['data']['is_drift']),
                'n_drifted_features': len(drifted_features_tabular),
                'pca_components': self.data['X_train_pca'].shape[1],
                'explained_variance': float(sum(self.pca.explained_variance_ratio_))
            })
            
            # Log drift p-values as metrics
            # for i, p_val in enumerate(feature_p_vals):
            #     mlflow.log_metric(f"feature_{i}_drift_p_value", p_val)
            
            # Log artifacts
            mlflow.log_artifacts(str(run_dir))
            
            # Log model with signature and input example
            mlflow.keras.log_model(
                self.model,
                "model",
                registered_model_name="fashion_mnist_model"
            )

            # # Save drift report in MLflow
            # mlflow.log_artifact(self.output_dir / 'drift_report.json')
        
        self.logger.info(f"Model and artifacts saved to {run_dir}")
        self.logger.info(f"Detected drift in {len(drifted_features_tabular)} features")
        
        return {
            'train_accuracy': train_score,
            'val_accuracy': val_score,
            'drift_report': drift_report
        }

def main():
    # Initialize pipeline
    pipeline = MLOpsPipeline()
    
    # Load and preprocess data
    pipeline.load_and_preprocess_data()
    
    # M1: Perform EDA
    pipeline.perform_eda()
    
    # M2: Feature Engineering & Explainability
    pipeline.engineer_features()
    best_params, best_score = pipeline.optimize_hyperparameters()
    
    # M3: Train model with best parameters
    pipeline.train_model(best_params)
    
    # M2: Feature Engineering & Explainability
    pipeline.explain_features()
    
    # M4: Monitor performance
    performance_metrics = pipeline.monitor_performance()
    
    # Log final results
    with open(pipeline.output_dir / 'results.json', 'w') as f:
        json.dump({
            'best_parameters': best_params,
            'best_score': best_score,
            'performance_metrics': performance_metrics
        }, f, indent=4)

if __name__ == "__main__":
    main()
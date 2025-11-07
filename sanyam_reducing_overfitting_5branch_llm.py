# === Enhanced Supply Chain Forecasting with DPO Integration - PRODUCTION READY ===

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import warnings
import requests
import json
import time
import pickle
from requests.exceptions import Timeout, ConnectionError


warnings.filterwarnings('ignore')

# Core ML Libraries
from sklearn.model_selection import train_test_split, TimeSeriesSplit, KFold
from sklearn.preprocessing import OneHotEncoder, StandardScaler, RobustScaler
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import BayesianRidge, Ridge, Lasso, ElasticNet
from sklearn.feature_selection import SelectKBest, f_regression, RFE
import xgboost as xgb

# Deep Learning Libraries
import tensorflow as tf
from tensorflow.keras.models import Model, Sequential, load_model
from tensorflow.keras.layers import (Input, Dense, Conv1D, LSTM, GRU, Bidirectional,
                                     GlobalAveragePooling1D, Dropout, Concatenate, Multiply,
                                     LayerNormalization, MultiHeadAttention, Layer, Lambda,
                                     BatchNormalization, Add)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.regularizers import l1, l2, l1_l2

print("Enhanced Supply Chain Forecasting System with DPO - By: Sanyam Kathed and Hith Rahil Nidhan")
print("=" * 80)

# ========== DEBUG: Verify Imports ==========
print("\n" + "="*80)
print("VERIFYING IMPORTS")
print("="*80)

try:
    import tensorflow as tf
    print(f"✅ TensorFlow: {tf.__version__}")
except Exception as e:
    print(f"❌ TensorFlow: {e}")

try:
    import pandas as pd
    print(f"✅ Pandas: {pd.__version__}")
except Exception as e:
    print(f"❌ Pandas: {e}")

try:
    import numpy as np
    print(f"✅ NumPy: {np.__version__}")
except Exception as e:
    print(f"❌ NumPy: {e}")

try:
    import sklearn
    print(f"✅ Scikit-learn: {sklearn.__version__}")
except Exception as e:
    print(f"❌ Scikit-learn: {e}")

try:
    import xgboost as xgb
    print(f"✅ XGBoost: {xgb.__version__}")
except Exception as e:
    print(f"❌ XGBoost: {e}")

print("="*80 + "\n")


# --- 1. ENHANCED LLAMA2 INTEGRATION CLASS ---
class Llama2Integrator:
    def __init__(self, ollama_url="http://localhost:11434"):
        self.ollama_url = ollama_url
        self.model_name = "llama2"
        self.timeout = 120
        self.max_retries = 3
        self.is_available = self.check_ollama_availability()

    def check_ollama_availability(self):
        """Check if Ollama is running and Llama2 is available"""
        try:
            response = requests.get(f"{self.ollama_url}/api/tags", timeout=10)
            if response.status_code == 200:
                models = response.json().get('models', [])
                available_models = [model['name'] for model in models]
                if any('llama2' in model for model in available_models):
                    print("✓ Llama2 model detected in Ollama")
                    return True
                else:
                    print("⚠ Llama2 model not found. Available models:", available_models[:3])
                    return False
            else:
                print("⚠ Ollama server not responding")
                return False
        except Exception as e:
            print(f"⚠ Error connecting to Ollama: {e}")
            return False

    def query_llama2_with_retry(self, prompt, max_tokens=300):
        """Send query to Llama2 via Ollama with retry logic"""
        if not self.is_available:
            return "Llama2 not available. Please ensure Ollama is running with Llama2 model."

        truncated_prompt = prompt[:800] + "..." if len(prompt) > 800 else prompt

        for attempt in range(self.max_retries):
            try:
                payload = {
                    "model": self.model_name,
                    "prompt": truncated_prompt,
                    "stream": False,
                    "options": {
                        "temperature": 0.5,
                        "max_tokens": max_tokens
                    }
                }

                response = requests.post(
                    f"{self.ollama_url}/api/generate",
                    json=payload,
                    timeout=self.timeout
                )

                if response.status_code == 200:
                    result = response.json()
                    return result.get('response', 'No response received')
                else:
                    return f"Error: HTTP {response.status_code}"

            except Timeout:
                if attempt < self.max_retries - 1:
                    print(f"⏱ Timeout on attempt {attempt + 1}, retrying...")
                    time.sleep(5)
                    continue
                else:
                    return "Analysis unavailable due to timeout."
            except Exception as e:
                return f"Error querying Llama2: {str(e)}"

    def analyze_forecasting_results(self, metrics, predictions, feature_importance):
        """Generate intelligent analysis of forecasting results"""
        prompt = f"""
As a supply chain expert, briefly analyze these results:
Metrics: R²={metrics.get('r2', 0):.3f}, MAE={metrics.get('mae', 0):.2f}, Resilience={metrics.get('resilience_score', 0):.3f}
Predictions: Mean={np.mean(predictions):.1f}, Range={np.max(predictions) - np.min(predictions):.1f}
Provide: 1) Performance assessment 2) Key insights 3) Recommendations
Keep under 150 words.
"""
        return self.query_llama2_with_retry(prompt, max_tokens=200)

    def generate_business_insights(self, df, top_features):
        """Generate business insights from data patterns"""
        prompt = f"""
Supply chain analysis for {len(df):,} records:
Sales Average: {df['Sales'].mean():.0f}, StdDev: {df['Sales'].std():.0f}
Markets: {df['Market'].nunique() if 'Market' in df.columns else 'N/A'}
Provide: 1) Key patterns 2) Optimization opportunities 3) Risk factors
Keep under 120 words.
"""
        return self.query_llama2_with_retry(prompt, max_tokens=180)

    def explain_model_predictions(self, sample_prediction, uncertainty, feature_values):
        """Explain specific predictions in business terms"""
        confidence = 'High' if uncertainty < 50 else 'Medium' if uncertainty < 100 else 'Low'
        prompt = f"""
Forecast: Sales={sample_prediction:.0f}, Uncertainty={uncertainty:.1f}, Confidence={confidence}
Explain: 1) Business meaning 2) Confidence level 3) Recommended actions
Keep under 100 words.
"""
        return self.query_llama2_with_retry(prompt, max_tokens=150)

    def generate_executive_summary(self, overall_metrics, resilience_score, feature_count):
        """Generate executive summary report"""
        prompt = f"""
Executive Summary - Forecasting Performance:
Accuracy: {overall_metrics.get('r2', 0) * 100:.1f}%
Error: {overall_metrics.get('mae', 0):.1f}
Resilience: {resilience_score:.2f}/1.0
Provide strategic overview for C-level executives in under 100 words.
"""
        return self.query_llama2_with_retry(prompt, max_tokens=150)


# --- 2. ENHANCED ATTENTION MECHANISMS ---
class EnhancedAttention(Layer):
    def __init__(self, **kwargs):
        super(EnhancedAttention, self).__init__(**kwargs)

    def build(self, input_shape):
        self.W = self.add_weight(
            name='att_weight',
            shape=(input_shape[-1], 1),
            initializer='glorot_uniform',
            regularizer=l2(0.01),
            trainable=True
        )
        self.b = self.add_weight(
            name='att_bias',
            shape=(input_shape[1], 1),
            initializer='zeros',
            regularizer=l2(0.01),
            trainable=True
        )
        super(EnhancedAttention, self).build(input_shape)

    def call(self, x):
        e = tf.keras.backend.tanh(tf.keras.backend.dot(x, self.W) + self.b)
        a = tf.keras.backend.softmax(e, axis=1)
        output = x * a
        return tf.keras.backend.sum(output, axis=1)

    def get_config(self):
        config = super(EnhancedAttention, self).get_config()
        return config


# --- 3. COMPREHENSIVE FEATURE ENGINEERING ---
class AdvancedFeatureEngineer:
    def __init__(self):
        self.feature_names = []
        self.feature_selector = None

    def create_interaction_features(self, df, numerical_cols):
        """Create interaction features from existing numerical columns"""
        interaction_features = df.copy()

        for col in numerical_cols:
            if col in df.columns:
                interaction_features[f'{col}_squared'] = df[col] ** 2
                interaction_features[f'{col}_log'] = np.log1p(np.abs(df[col]))

        if 'Sales per customer' in df.columns and 'Order Item Quantity' in df.columns:
            interaction_features['sales_quantity_interaction'] = df['Sales per customer'] * df['Order Item Quantity']

        if 'Days for shipping (real)' in df.columns and 'Benefit per order' in df.columns:
            interaction_features['shipping_benefit_ratio'] = df['Days for shipping (real)'] / (
                        df['Benefit per order'] + 1e-8)

        return interaction_features

    def create_statistical_features(self, df, numerical_cols):
        """Create statistical features from existing data"""
        stat_features = df.copy()

        for col in numerical_cols:
            if col in df.columns:
                stat_features[f'{col}_percentile'] = df[col].rank(pct=True)
                stat_features[f'{col}_zscore'] = (df[col] - df[col].mean()) / (df[col].std() + 1e-8)
                window_size = min(50, len(df) // 20)
                stat_features[f'{col}_ma'] = df[col].rolling(window=window_size, min_periods=1).mean()

        return stat_features

    def select_best_features(self, X, y, k=150):
        """Select the k best features to prevent overfitting"""
        self.feature_selector = SelectKBest(score_func=f_regression, k=min(k, X.shape[1]))
        X_selected = self.feature_selector.fit_transform(X, y)

        selected_features = self.feature_selector.get_support(indices=True)
        print(f"✓ Feature selection: {X.shape[1]} → {X_selected.shape[1]} features")

        return X_selected


# --- 4. REGULARIZED MULTI-CHANNEL DATA FUSION NETWORK ---
def build_regularized_mcdfn_model(input_shape):
    """Build regularized Multi-Channel Data Fusion Network to prevent overfitting"""
    inputs = Input(shape=input_shape, name='main_input')

    # Channel 1: Convolutional Branch with Regularization
    conv_branch = Conv1D(64, 3, activation='relu', padding='same', kernel_regularizer=l2(0.01))(inputs)
    conv_branch = BatchNormalization()(conv_branch)
    conv_branch = Dropout(0.3)(conv_branch)
    conv_branch = Conv1D(32, 3, activation='relu', padding='same', kernel_regularizer=l2(0.01))(conv_branch)
    conv_branch = BatchNormalization()(conv_branch)
    conv_branch = Dropout(0.3)(conv_branch)
    conv_attention = EnhancedAttention()(conv_branch)

    # Channel 2: LSTM Branch with Regularization
    lstm_branch = LSTM(64, return_sequences=True, dropout=0.3, recurrent_dropout=0.3, kernel_regularizer=l2(0.01))(
        inputs)
    lstm_branch = BatchNormalization()(lstm_branch)
    lstm_branch = LSTM(32, return_sequences=True, dropout=0.3, recurrent_dropout=0.3, kernel_regularizer=l2(0.01))(
        lstm_branch)
    lstm_attention = EnhancedAttention()(lstm_branch)

    # Channel 3: GRU Branch with Regularization
    gru_branch = GRU(64, return_sequences=True, dropout=0.3, recurrent_dropout=0.3, kernel_regularizer=l2(0.01))(inputs)
    gru_branch = BatchNormalization()(gru_branch)
    gru_branch = GRU(32, return_sequences=True, dropout=0.3, recurrent_dropout=0.3, kernel_regularizer=l2(0.01))(
        gru_branch)
    gru_attention = EnhancedAttention()(gru_branch)

    # Channel 4: Bidirectional LSTM with Regularization
    bilstm_branch = Bidirectional(
        LSTM(32, return_sequences=True, dropout=0.3, recurrent_dropout=0.3, kernel_regularizer=l2(0.01)))(inputs)
    bilstm_branch = BatchNormalization()(bilstm_branch)
    bilstm_attention = EnhancedAttention()(bilstm_branch)

    # Channel 5: Transformer Branch with Regularization
    transformer_branch = MultiHeadAttention(num_heads=4, key_dim=32)(inputs, inputs)
    transformer_branch = LayerNormalization()(transformer_branch + inputs)
    transformer_branch = Dropout(0.3)(transformer_branch)
    transformer_attention = tf.keras.layers.GlobalAveragePooling1D()(transformer_branch)

    # Fusion Layer with Regularization
    fusion_features = Concatenate()(
        [conv_attention, lstm_attention, gru_attention, bilstm_attention, transformer_attention])

    # Regularized Dense Processing
    fusion_dense = Dense(256, activation='relu', kernel_regularizer=l2(0.01))(fusion_features)
    fusion_dense = BatchNormalization()(fusion_dense)
    fusion_dense = Dropout(0.4)(fusion_dense)
    fusion_dense = Dense(128, activation='relu', kernel_regularizer=l2(0.01))(fusion_dense)
    fusion_dense = BatchNormalization()(fusion_dense)
    fusion_dense = Dropout(0.3)(fusion_dense)
    fusion_dense = Dense(64, activation='relu', kernel_regularizer=l2(0.01))(fusion_dense)
    fusion_dense = Dropout(0.2)(fusion_dense)

    # Probabilistic Output
    mean_output = Dense(1, activation='linear', name='mean_output')(fusion_dense)
    std_output = Dense(1, activation='softplus', name='std_output')(fusion_dense)

    model = Model(inputs=inputs, outputs=[mean_output, std_output], name='RegularizedMCDFN')
    return model


# --- 5. REGULARIZED ENSEMBLE METHODS ---
class RegularizedEnsemble:
    def __init__(self):
        self.models = {}
        self.weights = {}
        self.is_fitted = False

    def add_model(self, name, model, weight=1.0):
        self.models[name] = model
        self.weights[name] = weight

    def fit(self, X, y):
        """Fit all ensemble models with regularization"""
        print("✓ Building Regularized Ensemble Models")

        regularized_models = {
            'RandomForest': RandomForestRegressor(
                n_estimators=200, max_depth=10, min_samples_split=10,
                min_samples_leaf=5, random_state=42, n_jobs=-1
            ),
            'XGBoost': xgb.XGBRegressor(
                n_estimators=200, max_depth=6, learning_rate=0.1,
                subsample=0.8, colsample_bytree=0.8, reg_alpha=0.1,
                reg_lambda=0.1, random_state=42
            ),
            'GradientBoosting': GradientBoostingRegressor(
                n_estimators=200, max_depth=6, learning_rate=0.1,
                subsample=0.8, random_state=42
            ),
            'ElasticNet': ElasticNet(alpha=0.1, l1_ratio=0.5, random_state=42),
            'Ridge': Ridge(alpha=1.0, random_state=42),
            'BayesianRidge': BayesianRidge()
        }

        for name, model in regularized_models.items():
            self.add_model(name, model)

        for name, model in self.models.items():
            print(f"  ✓ Training {name}")
            if hasattr(model, 'fit'):
                model.fit(X, y)

        self.is_fitted = True
        print("✓ Regularized Ensemble models trained successfully")

    def predict(self, X):
        """Make ensemble predictions"""
        if not self.is_fitted:
            raise ValueError("Ensemble must be fitted before prediction")

        predictions = []
        total_weight = sum(self.weights.values())

        for name, model in self.models.items():
            if hasattr(model, 'predict'):
                pred = model.predict(X)
                if len(pred.shape) > 1:
                    pred = pred.flatten()
                weighted_pred = pred * (self.weights[name] / total_weight)
                predictions.append(weighted_pred)

        return np.sum(predictions, axis=0)


# --- 6. CROSS-VALIDATION FOR ROBUST EVALUATION ---
class RobustValidator:
    def __init__(self, cv_folds=5):
        self.cv_folds = cv_folds
        self.cv_scores = []

    def cross_validate_model(self, model, X, y):
        """Perform cross-validation to assess model robustness"""
        kfold = KFold(n_splits=self.cv_folds, shuffle=True, random_state=42)
        scores = []

        for fold, (train_idx, val_idx) in enumerate(kfold.split(X)):
            X_train_fold = X[train_idx]
            X_val_fold = X[val_idx]
            y_train_fold = y.iloc[train_idx] if hasattr(y, 'iloc') else y[train_idx]
            y_val_fold = y.iloc[val_idx] if hasattr(y, 'iloc') else y[val_idx]

            model.fit(X_train_fold, y_train_fold)
            y_pred = model.predict(X_val_fold)
            score = r2_score(y_val_fold, y_pred)
            scores.append(score)
            print(f"  Fold {fold + 1}: R² = {score:.4f}")

        self.cv_scores = scores
        mean_score = np.mean(scores)
        std_score = np.std(scores)
        print(f"✓ Cross-validation results: {mean_score:.4f} ± {std_score:.4f}")

        return mean_score, std_score


# --- 7. SUPPLY CHAIN RESILIENCE METRICS ---
class ResilienceMetrics:
    def __init__(self):
        self.disruption_scenarios = []
        self.recovery_times = []
        self.adaptation_scores = []

    def calculate_resilience_score(self, predictions, actuals):
        """Calculate comprehensive resilience score"""
        base_mae = mean_absolute_error(actuals, predictions)
        base_r2 = r2_score(actuals, predictions)

        prediction_volatility = np.std(predictions)
        actual_volatility = np.std(actuals)
        volatility_ratio = min(prediction_volatility / (actual_volatility + 1e-8), 1.0)

        pred_trend = np.polyfit(range(len(predictions)), predictions, 1)[0]
        actual_trend = np.polyfit(range(len(actuals)), actuals, 1)[0]
        trend_similarity = 1 - abs(pred_trend - actual_trend) / (abs(actual_trend) + 1e-8)

        resilience_score = (
                0.4 * base_r2 +
                0.3 * (1 - base_mae / (np.mean(actuals) + 1e-8)) +
                0.2 * volatility_ratio +
                0.1 * max(0, trend_similarity)
        )

        return {
            'resilience_score': resilience_score,
            'base_r2': base_r2,
            'base_mae': base_mae,
            'volatility_ratio': volatility_ratio,
            'trend_similarity': trend_similarity
        }


# --- 8. DPO (DIRECT PREFERENCE OPTIMIZATION) TRAINER ---
class DPOTrainer:
    """
    Direct Preference Optimization for Supply Chain Forecasting
    Implements reward-free RL using preference pairs
    """

    def __init__(self, base_model, beta=0.1):
        self.reference_model = base_model
        self.policy_model = None
        self.beta = beta
        self.preference_data = []

    def load_preferences(self, preference_file='dpo_preferences.json'):
        """Load collected preference pairs from Streamlit"""
        try:
            if os.path.exists(preference_file):
                with open(preference_file, 'r') as f:
                    self.preference_data = json.load(f)
                print(f"✓ Loaded {len(self.preference_data)} preference pairs")
                return self.preference_data
            else:
                print("⚠ No preference file found")
                return []
        except Exception as e:
            print(f"⚠ Error loading preferences: {e}")
            return []

    def prepare_dpo_dataset(self, preferences, preprocessor):
        """Convert preference data to training format"""
        X = []
        chosen_targets = []
        rejected_targets = []

        for pref in preferences:
            try:
                features = pref['scenario_features']
                feature_df = pd.DataFrame([features])
                feature_array = preprocessor.transform(feature_df)

                if hasattr(feature_array, 'toarray'):
                    feature_array = feature_array.toarray()

                X.append(feature_array[0])
                chosen_targets.append(pref['chosen_prediction'])
                rejected_targets.append(pref['rejected_prediction'])
            except Exception as e:
                print(f"⚠ Skipping preference due to error: {e}")
                continue

        return np.array(X), np.array(chosen_targets), np.array(rejected_targets)

    def build_policy_model(self, input_shape):
        """Create policy model (clone of MCDFN architecture)"""
        self.policy_model = build_regularized_mcdfn_model(input_shape)
        self.policy_model.set_weights(self.reference_model.get_weights())
        print("✓ Policy model initialized from reference model")

    def train_dpo(self, X, chosen_targets, rejected_targets, epochs=50, batch_size=16, validation_split=0.2):
        """Train policy model using DPO"""
        print("\n" + "=" * 80)
        print("TRAINING WITH DIRECT PREFERENCE OPTIMIZATION (DPO)")
        print("=" * 80)

        if len(X) < 10:
            print("⚠ Insufficient preferences for DPO training (need at least 10)")
            return self.reference_model

        optimizer = Adam(learning_rate=1e-4)

        split_idx = int(len(X) * (1 - validation_split))
        X_train, X_val = X[:split_idx], X[split_idx:]
        chosen_train, chosen_val = chosen_targets[:split_idx], chosen_targets[split_idx:]
        rejected_train, rejected_val = rejected_targets[:split_idx], rejected_targets[split_idx:]

        best_loss = float('inf')
        patience_counter = 0

        for epoch in range(epochs):
            epoch_losses = []

            for i in range(0, len(X_train), batch_size):
                batch_X = X_train[i:i + batch_size]
                batch_chosen = chosen_train[i:i + batch_size]
                batch_rejected = rejected_train[i:i + batch_size]

                if len(batch_X) == 0:
                    continue

                with tf.GradientTape() as tape:
                    batch_X_reshaped = batch_X.reshape(batch_X.shape[0], batch_X.shape[1], 1)

                    policy_pred = self.policy_model(batch_X_reshaped, training=True)
                    ref_pred = self.reference_model(batch_X_reshaped, training=False)

                    policy_chosen_reward = -tf.reduce_mean((policy_pred[0] - batch_chosen.reshape(-1, 1)) ** 2)
                    policy_rejected_reward = -tf.reduce_mean((policy_pred[0] - batch_rejected.reshape(-1, 1)) ** 2)
                    ref_chosen_reward = -tf.reduce_mean((ref_pred[0] - batch_chosen.reshape(-1, 1)) ** 2)
                    ref_rejected_reward = -tf.reduce_mean((ref_pred[0] - batch_rejected.reshape(-1, 1)) ** 2)

                    logits = self.beta * ((policy_chosen_reward - ref_chosen_reward) - (
                                policy_rejected_reward - ref_rejected_reward))
                    loss = -tf.math.log(tf.nn.sigmoid(logits) + 1e-8)

                    epoch_losses.append(loss.numpy())

                gradients = tape.gradient(loss, self.policy_model.trainable_variables)
                optimizer.apply_gradients(zip(gradients, self.policy_model.trainable_variables))

            avg_train_loss = np.mean(epoch_losses) if epoch_losses else float('inf')

            if epoch % 10 == 0:
                print(f"Epoch {epoch + 1}/{epochs} - Train Loss: {avg_train_loss:.4f}")

            if avg_train_loss < best_loss:
                best_loss = avg_train_loss
                patience_counter = 0
                self.policy_model.save('dpo_optimized_model.keras')  # FIXED
            else:
                patience_counter += 1
                if patience_counter >= 15:
                    print("⏹ Early stopping triggered")
                    break

        print("=" * 80)
        print("DPO TRAINING COMPLETED")
        print("=" * 80)

        try:
            self.policy_model = load_model('dpo_optimized_model.keras',
                                           custom_objects={'EnhancedAttention': EnhancedAttention})  # FIXED
        except:
            print("⚠ Could not load DPO model, using trained policy")

        return self.policy_model

    def evaluate_dpo_improvement(self, X_test, y_test):
        """Compare reference model vs DPO-optimized model"""
        X_test_reshaped = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

        ref_pred = self.reference_model.predict(X_test_reshaped, verbose=0)[0].flatten()
        ref_mae = mean_absolute_error(y_test, ref_pred)
        ref_r2 = r2_score(y_test, ref_pred)

        dpo_pred = self.policy_model.predict(X_test_reshaped, verbose=0)[0].flatten()
        dpo_mae = mean_absolute_error(y_test, dpo_pred)
        dpo_r2 = r2_score(y_test, dpo_pred)

        print("\n" + "=" * 80)
        print("DPO IMPROVEMENT EVALUATION")
        print("=" * 80)
        print(f"Reference Model - MAE: {ref_mae:.4f}, R²: {ref_r2:.6f}")
        print(f"DPO Model       - MAE: {dpo_mae:.4f}, R²: {dpo_r2:.6f}")
        print(
            f"Improvement     - MAE: {((ref_mae - dpo_mae) / ref_mae) * 100:.2f}%, R²: {((dpo_r2 - ref_r2) / ref_r2) * 100:.2f}%")
        print("=" * 80)

        return {
            'ref_mae': ref_mae, 'ref_r2': ref_r2,
            'dpo_mae': dpo_mae, 'dpo_r2': dpo_r2,
            'mae_improvement': ((ref_mae - dpo_mae) / ref_mae) * 100,
            'r2_improvement': ((dpo_r2 - ref_r2) / ref_r2) * 100
        }


# --- 9. MAIN ENHANCED FORECASTING SYSTEM ---
class EnhancedSupplyChainForecaster:
    def __init__(self):
        self.preprocessor = None
        self.mcdfn_model = None
        self.ensemble = RegularizedEnsemble()
        self.resilience_metrics = ResilienceMetrics()
        self.feature_engineer = AdvancedFeatureEngineer()
        self.validator = RobustValidator()
        self.feature_importance = None
        self.llm_integrator = Llama2Integrator()
        self.feature_names = None

    def predict_with_uncertainty(self, X):
        """Predict with MCDFN + ensemble, returning uncertainties."""
        # MCDFN branch
        X_reshaped = X.reshape(X.shape[0], X.shape[1], 1)
        mean_pred, std_pred = self.mcdfn_model.predict(X_reshaped)
        mean_pred = mean_pred.flatten()
        std_pred = std_pred.flatten()
        # Ensemble branch
        ensemble_pred = self.ensemble.predict(X)
        # Combine your predictions
        combined = 0.6 * mean_pred + 0.4 * ensemble_pred
        lower_bound = mean_pred - 1.96 * std_pred
        upper_bound = mean_pred + 1.96 * std_pred
        return {
            'prediction': combined,
            'uncertainty': std_pred,
            'ensemble_pred': ensemble_pred,
            'mcdfn_pred': mean_pred,
            'lower_bound': lower_bound,
            'upper_bound': upper_bound
        }

    def prepare_data(self, df, features, target):
        """Enhanced data preparation with feature selection"""
        print("✓ Preparing enhanced dataset with regularization")

        # Validate target column
        if target not in df.columns:
            raise KeyError(f"Target column '{target}' not found in dataset!")

        # Validate features
        missing_features = [f for f in features if f not in df.columns]
        if missing_features:
            print(f"\n⚠️ WARNING: {len(missing_features)} features missing from dataset")
            features = [f for f in features if f in df.columns]
            print(f"✓ Using {len(features)} available features")

            if len(features) == 0:
                raise ValueError("No valid features found!")

        # Define categorical and numerical columns based on YOUR dataset
        categorical_cols = [
            'Type', 'Delivery Status', 'Category Name', 'Department Name',
            'Market', 'Shipping Mode', 'Order Region', 'customer_country', 'order_country'
        ]

        # Only keep categorical cols that are in features
        categorical_cols = [col for col in categorical_cols if col in features]

        numerical_cols = [
            'Order Item Quantity', 'Product Price', 'Sales per customer', 'Benefit per order',
            'Late_delivery_risk', 'delivery_delay_days', 'supply_chain_risk_score',
            'estimated_distance_to_port_km', 'port_congestion_score', 'high_port_traffic',
            'teu_volume', 'cargo_tonnage', 'storm_count', 'storm_binary_flag',
            'storm_severity_score', 'total_storm_damage', 'olist_avg_price',
            'olist_avg_freight', 'olist_market_presence', 'olist_regional_volume',
            'product_avg_sales', 'region_avg_sales', 'order_year', 'order_month',
            'order_quarter', 'order_day', 'order_dayofweek', 'order_week',
            'is_holiday_season', 'is_summer', 'days_since_epoch', 'month_sin', 'month_cos'
        ]

        # Only keep numerical cols that are in features
        numerical_cols = [col for col in numerical_cols if col in features]

        print(f"✓ Categorical features: {len(categorical_cols)}")
        print(f"✓ Numerical features: {len(numerical_cols)}")

        # Clean data
        initial_rows = len(df)
        df = df.dropna(subset=[target])
        dropped_rows = initial_rows - len(df)
        if dropped_rows > 0:
            print(f"ℹ️ Dropped {dropped_rows} rows with missing target values")

        X = df[features]
        y = df[target]

        print(f"✓ Data shape: {X.shape[0]} samples × {X.shape[1]} features")

        # Feature engineering
        print("✓ Creating enhanced features")
        X_enhanced = self.feature_engineer.create_interaction_features(X, numerical_cols)
        X_enhanced = self.feature_engineer.create_statistical_features(X_enhanced, numerical_cols)

        new_numerical_cols = [col for col in X_enhanced.columns if col not in categorical_cols]

        # Build preprocessing pipeline
        transformers = []

        if len(categorical_cols) > 0:
            transformers.append(
                ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_cols)
            )

        if len(new_numerical_cols) > 0:
            transformers.append(
                ('num', Pipeline([
                    ('imputer', SimpleImputer(strategy='median')),
                    ('scaler', RobustScaler())
                ]), new_numerical_cols)
            )

        if len(transformers) == 0:
            raise ValueError("No valid features for preprocessing!")

        self.preprocessor = ColumnTransformer(transformers=transformers)

        X_processed = self.preprocessor.fit_transform(X_enhanced)
        X_dense = X_processed if not hasattr(X_processed, "toarray") else X_processed.toarray()

        X_selected = self.feature_engineer.select_best_features(X_dense, y, k=150)

        print(f"✓ Final dataset: {X_selected.shape[1]} selected features")

        return X_selected, y

    def build_ensemble_models(self, X_train, y_train):
        """Build and train ensemble models"""
        self.ensemble.fit(X_train, y_train)

    def train_mcdfn(self, X_train, y_train, X_val, y_val):
        """Train regularized Multi-Channel Data Fusion Network"""
        print("✓ Training Regularized MCDFN model...")

        X_train_reshaped = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
        X_val_reshaped = X_val.reshape(X_val.shape[0], X_val.shape[1], 1)

        self.mcdfn_model = build_regularized_mcdfn_model(X_train_reshaped.shape[1:])

        self.mcdfn_model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss={'mean_output': 'mse', 'std_output': 'mse'},
            loss_weights={'mean_output': 1.0, 'std_output': 0.1},
            metrics={'mean_output': ['mae'], 'std_output': ['mae']}
        )

        callbacks = [
            EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True, verbose=1),
            ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=10, min_lr=1e-6, verbose=1),
            ModelCheckpoint('best_regularized_model.keras', save_best_only=True, monitor='val_loss', verbose=1)  # FIXED
        ]

        history = self.mcdfn_model.fit(
            X_train_reshaped, [y_train, np.ones_like(y_train)],
            validation_data=(X_val_reshaped, [y_val, np.ones_like(y_val)]),
            epochs=1,
            batch_size=64,
            callbacks=callbacks,
            verbose=1
        )

        print("✓ Regularized MCDFN model trained successfully")
        return history

    def calculate_feature_importance(self, X_train, y_train):
        """Calculate feature importance using multiple methods"""
        print("✓ Calculating feature importance")

        rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
        rf_model.fit(X_train, y_train)
        rf_importance = rf_model.feature_importances_

        xgb_model = xgb.XGBRegressor(n_estimators=100, random_state=42)
        xgb_model.fit(X_train, y_train)
        xgb_importance = xgb_model.feature_importances_

        combined_importance = (rf_importance + xgb_importance) / 2

        self.feature_importance = {
            'rf_importance': rf_importance,
            'xgb_importance': xgb_importance,
            'combined_importance': combined_importance
        }

        print("✓ Feature importance calculated")
        return self.feature_importance

    def generate_llm_insights(self, df, metrics, predictions, feature_importance):
        """Generate LLM-powered insights and reports"""
        print("\n" + "=" * 60)
        print("GENERATING LLAMA2-POWERED INSIGHTS")
        print("=" * 60)

        top_feature_indices = np.argsort(feature_importance['combined_importance'])[-10:]
        top_features = [f"Feature_{i}" for i in top_feature_indices]

        print("\n✓ FORECASTING RESULTS ANALYSIS:")
        print("-" * 40)
        analysis = self.llm_integrator.analyze_forecasting_results(metrics, predictions['prediction'],
                                                                   feature_importance)
        print(analysis)

        print("\n✓ BUSINESS INSIGHTS:")
        print("-" * 40)
        business_insights = self.llm_integrator.generate_business_insights(df, top_features)
        print(business_insights)

        print("\n✓ SAMPLE PREDICTION EXPLANATION:")
        print("-" * 40)
        sample_idx = np.random.choice(len(predictions['prediction']))
        sample_pred = predictions['prediction'][sample_idx]
        sample_uncertainty = predictions['uncertainty'][sample_idx]
        sample_features = f"Sales prediction: {sample_pred:.2f}, Uncertainty: {sample_uncertainty:.2f}"
        explanation = self.llm_integrator.explain_model_predictions(sample_pred, sample_uncertainty, sample_features)
        print(explanation)

        print("\n✓ EXECUTIVE SUMMARY:")
        print("-" * 40)
        executive_summary = self.llm_integrator.generate_executive_summary(
            metrics, metrics.get('resilience_score', 0), len(feature_importance['combined_importance'])
        )
        print(executive_summary)

        return {
            'analysis': analysis,
            'business_insights': business_insights,
            'prediction_explanation': explanation,
            'executive_summary': executive_summary
        }


# --- 10. MODEL SAVING FOR STREAMLIT DEPLOYMENT ---
def save_models_for_streamlit(forecaster, X_enhanced, y_enhanced, save_dir='./'):
    """Save all trained models and preprocessors for Streamlit deployment"""
    print("\n" + "=" * 80)
    print("SAVING MODELS FOR STREAMLIT DEPLOYMENT")
    print("=" * 80)

    os.makedirs(save_dir, exist_ok=True)

    try:
        # 1. Save MCDFN Model - FIXED
        if hasattr(forecaster, 'mcdfn_model') and forecaster.mcdfn_model is not None:
            model_path = os.path.join(save_dir, 'best_regularized_model.keras')  # FIXED
            forecaster.mcdfn_model.save(model_path)
            print(f"✓ MCDFN model saved: {model_path}")

        # 2. Save Ensemble Models
        if hasattr(forecaster, 'ensemble') and hasattr(forecaster.ensemble, 'models'):
            for name, model in forecaster.ensemble.models.items():
                clean_name = name.lower().replace(' ', '_').replace('-', '_')
                model_path = os.path.join(save_dir, f'{clean_name}_model.pkl')
                with open(model_path, 'wb') as f:
                    pickle.dump(model, f)
                print(f"✓ {name} model saved: {model_path}")

            ensemble_path = os.path.join(save_dir, 'ensemble_model.pkl')
            with open(ensemble_path, 'wb') as f:
                pickle.dump(forecaster.ensemble, f)
            print(f"✓ Complete ensemble saved: {ensemble_path}")

        # 3. Save Preprocessor
        if hasattr(forecaster, 'preprocessor') and forecaster.preprocessor is not None:
            preprocessor_path = os.path.join(save_dir, 'preprocessor.pkl')
            with open(preprocessor_path, 'wb') as f:
                pickle.dump(forecaster.preprocessor, f)
            print(f"✓ Preprocessor saved: {preprocessor_path}")

        # 4. Save Feature Names
        if hasattr(X_enhanced, 'columns'):
            feature_names = X_enhanced.columns.tolist()
            features_path = os.path.join(save_dir, 'feature_names.pkl')
            with open(features_path, 'wb') as f:
                pickle.dump(feature_names, f)
            print(f"✓ Feature names saved ({len(feature_names)} features): {features_path}")

        # 5. Save Feature Engineer
        if hasattr(forecaster, 'feature_engineer'):
            fe_path = os.path.join(save_dir, 'feature_engineer.pkl')
            with open(fe_path, 'wb') as f:
                pickle.dump(forecaster.feature_engineer, f)
            print(f"✓ Feature engineer saved: {fe_path}")

        # 6. Save Model Metadata
        metadata = {
            'model_version': '1.0_DPO',
            'training_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'n_samples': len(X_enhanced) if hasattr(X_enhanced, '__len__') else 0,
            'n_features': len(X_enhanced.columns) if hasattr(X_enhanced, 'columns') else 0,
            'target_column': 'sales_amount',
            'python_version': '3.11',
            'tensorflow_version': tf.__version__,
            'framework': 'TensorFlow + Scikit-learn + XGBoost + DPO',
            'dpo_enabled': True,
            'model_format': '.keras'  # NEW
        }

        metadata_path = os.path.join(save_dir, 'model_metadata.json')
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=4)
        print(f"✓ Model metadata saved: {metadata_path}")

        print("=" * 80)
        print("ALL MODELS SAVED SUCCESSFULLY!")
        print("=" * 80)
        print(f"\nFiles saved in: {os.path.abspath(save_dir)}")
        print("\nNext Steps:")
        print("1. Ensure Ollama is running: `ollama run llama2`")
        print("2. Run the Streamlit app: `streamlit run streamlit_app_dpo.py`")
        print("3. Collect expert preferences to enable DPO training")
        print("=" * 80)

        return True

    except Exception as e:
        print(f"✗ Error saving models: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


# --- 11. MAIN EXECUTION ---
# --- 11. MAIN EXECUTION WITH DEBUG OUTPUT ---
def main():
    print("\n✓ Starting Enhanced Supply Chain Forecasting System with DPO")

    # ========== STEP 1: LOAD DATA ==========
    print("\n" + "=" * 80)
    print("STEP 1: LOADING DATASET")
    print("=" * 80)

    file_path = 'supply_chain_synthesized_dataset.csv'
    print(f"Looking for file: {file_path}")
    print(f"Current directory: {os.getcwd()}")
    print(f"File exists: {os.path.exists(file_path)}")

    if not os.path.exists(file_path):
        print(f"\n❌ ERROR: File not found at {file_path}")
        print("\nSearching for CSV files in current directory...")
        csv_files = [f for f in os.listdir('.') if f.endswith('.csv')]
        if csv_files:
            print(f"Found CSV files: {csv_files}")
            print(f"\n💡 TIP: Update file_path to one of these files")
        else:
            print("No CSV files found in current directory")
        return None, None, None, None

    try:
        print("Attempting to load dataset...")
        df = pd.read_csv(file_path, encoding='latin1')
        print(f"✅ Dataset loaded successfully!")
        print(f"   Shape: {df.shape[0]} rows × {df.shape[1]} columns")
        print(f"   Memory: {df.memory_usage(deep=True).sum() / 1024 ** 2:.2f} MB")
    except Exception as e:
        print(f"❌ Error loading dataset: {e}")
        import traceback
        traceback.print_exc()
        return None, None, None, None

    # ========== STEP 2: FEATURE SELECTION ==========
    print("\n" + "=" * 80)
    print("STEP 2: FEATURE SELECTION")
    print("=" * 80)

    features = [
        # Categorical features
        'Type', 'Delivery Status', 'Category Name', 'Department Name',
        'Market', 'Shipping Mode', 'Order Region', 'customer_country', 'order_country',

        # Numerical features - Order details
        'Order Item Quantity', 'Product Price', 'Sales per customer', 'Benefit per order',

        # Numerical features - Delivery & Risk
        'Late_delivery_risk', 'delivery_delay_days', 'supply_chain_risk_score',

        # Numerical features - Port & Logistics
        'estimated_distance_to_port_km', 'port_congestion_score', 'high_port_traffic',
        'teu_volume', 'cargo_tonnage',

        # Numerical features - Weather/Storm
        'storm_count', 'storm_binary_flag', 'storm_severity_score', 'total_storm_damage',

        # Numerical features - Market intelligence
        'olist_avg_price', 'olist_avg_freight', 'olist_market_presence',
        'olist_regional_volume', 'product_avg_sales', 'region_avg_sales',

        # Temporal features
        'order_year', 'order_month', 'order_quarter', 'order_day',
        'order_dayofweek', 'order_week', 'is_holiday_season', 'is_summer',
        'days_since_epoch', 'month_sin', 'month_cos'
    ]

    target = 'sales_amount'

    print(f"Requested {len(features)} features")
    print(f"Target: '{target}'")

    # Validate features
    missing_features = [f for f in features if f not in df.columns]
    if missing_features:
        print(f"\n⚠️ WARNING: {len(missing_features)} features not found in dataset")
        for f in missing_features[:5]:  # Show first 5
            print(f"  ❌ '{f}'")
        if len(missing_features) > 5:
            print(f"  ... and {len(missing_features) - 5} more")

        features = [f for f in features if f in df.columns]
        print(f"\n✓ Continuing with {len(features)} available features")

    if target not in df.columns:
        print(f"\n❌ ERROR: Target column '{target}' not found!")
        print(f"Available columns: {list(df.columns)[:10]}... (showing first 10)")
        return None, None, None, None

    print(f"✅ All validations passed - using {len(features)} features")

    # ========== STEP 3: INITIALIZE FORECASTER ==========
    print("\n" + "=" * 80)
    print("STEP 3: INITIALIZING FORECASTER")
    print("=" * 80)

    try:
        print("Creating EnhancedSupplyChainForecaster instance...")
        forecaster = EnhancedSupplyChainForecaster()
        print("✅ Forecaster initialized successfully")
    except Exception as e:
        print(f"❌ Error initializing forecaster: {e}")
        import traceback
        traceback.print_exc()
        return None, None, None, None

    # ========== STEP 4: PREPARE DATA ==========
    print("\n" + "=" * 80)
    print("STEP 4: DATA PREPARATION")
    print("=" * 80)

    try:
        print("Starting data preparation pipeline...")
        X_enhanced, y = forecaster.prepare_data(df, features, target)
        print(f"✅ Data prepared successfully")
        print(f"   X shape: {X_enhanced.shape}")
        print(f"   y shape: {y.shape}")
    except Exception as e:
        print(f"❌ Error in data preparation: {e}")
        import traceback
        traceback.print_exc()
        return None, None, None, None

    # ========== STEP 5: SPLIT DATA ==========
    print("\n" + "=" * 80)
    print("STEP 5: SPLITTING DATA")
    print("=" * 80)

    try:
        X_train, X_temp, y_train, y_temp = train_test_split(X_enhanced, y, test_size=0.4, random_state=42)
        X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)
        print(f"✅ Data split complete")
        print(f"   Train: {X_train.shape}")
        print(f"   Val:   {X_val.shape}")
        print(f"   Test:  {X_test.shape}")
    except Exception as e:
        print(f"❌ Error splitting data: {e}")
        import traceback
        traceback.print_exc()
        return None, None, None, None

    # ========== STEP 6: TRAIN ENSEMBLE ==========
    print("\n" + "=" * 80)
    print("STEP 6: TRAINING ENSEMBLE MODELS")
    print("=" * 80)

    try:
        forecaster.build_ensemble_models(X_train, y_train)
        print("✅ Ensemble models trained")
    except Exception as e:
        print(f"❌ Error training ensemble: {e}")
        import traceback
        traceback.print_exc()
        return None, None, None, None

    # ========== STEP 7: TRAIN MCDFN ==========
    print("\n" + "=" * 80)
    print("STEP 7: TRAINING MCDFN MODEL")
    print("=" * 80)

    try:
        mcdfn_history = forecaster.train_mcdfn(X_train, y_train, X_val, y_val)
        print("✅ MCDFN model trained")
    except Exception as e:
        print(f"❌ Error training MCDFN: {e}")
        import traceback
        traceback.print_exc()
        return None, None, None, None

    # ========== STEP 8: CROSS-VALIDATION ==========
    print("\n" + "=" * 80)
    print("STEP 8: CROSS-VALIDATION")
    print("=" * 80)

    try:
        cv_mean, cv_std = forecaster.validator.cross_validate_model(
            RandomForestRegressor(n_estimators=100, random_state=42), X_train, y_train
        )
        print(f"✅ Cross-validation complete: {cv_mean:.4f} ± {cv_std:.4f}")
    except Exception as e:
        print(f"❌ Error in cross-validation: {e}")
        cv_mean, cv_std = 0, 0

    # ========== STEP 9: FEATURE IMPORTANCE ==========
    print("\n" + "=" * 80)
    print("STEP 9: CALCULATING FEATURE IMPORTANCE")
    print("=" * 80)

    try:
        feature_importance = forecaster.calculate_feature_importance(X_train, y_train)
        print("✅ Feature importance calculated")
    except Exception as e:
        print(f"❌ Error calculating feature importance: {e}")
        feature_importance = {'combined_importance': np.zeros(X_train.shape[1])}

    # ========== STEP 10: MAKE PREDICTIONS ==========
    print("\n" + "=" * 80)
    print("STEP 10: GENERATING PREDICTIONS")
    print("=" * 80)

    try:
        predictions = forecaster.predict_with_uncertainty(X_test)
        print("✅ Predictions generated")
    except Exception as e:
        print(f"❌ Error making predictions: {e}")
        import traceback
        traceback.print_exc()
        return None, None, None, None

    # ========== STEP 11: CALCULATE METRICS ==========
    print("\n" + "=" * 80)
    print("STEP 11: CALCULATING METRICS")
    print("=" * 80)

    try:
        resilience_metrics = forecaster.resilience_metrics.calculate_resilience_score(
            predictions['prediction'], y_test
        )

        mae = mean_absolute_error(y_test, predictions['prediction'])
        mse = mean_squared_error(y_test, predictions['prediction'])
        r2 = r2_score(y_test, predictions['prediction'])
        rmse = np.sqrt(mse)
        coverage_95 = np.mean((y_test >= predictions['lower_bound']) & (y_test <= predictions['upper_bound']))

        all_metrics = {
            'mae': mae, 'mse': mse, 'r2': r2, 'rmse': rmse,
            'resilience_score': resilience_metrics['resilience_score'],
            'coverage_95': coverage_95, 'cv_mean': cv_mean, 'cv_std': cv_std
        }

        print("✅ Metrics calculated")
    except Exception as e:
        print(f"❌ Error calculating metrics: {e}")
        import traceback
        traceback.print_exc()
        return None, None, None, None

    # ========== STEP 12: DISPLAY RESULTS ==========
    print("\n" + "=" * 80)
    print("FINAL RESULTS")
    print("=" * 80)
    print(f"Features used: {len(features)}")
    print(f"MAE: {mae:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print(f"R² Score: {r2:.6f} ({r2 * 100:.2f}%)")
    print(f"Cross-validation: {cv_mean:.4f} ± {cv_std:.4f}")
    print(f"Resilience Score: {resilience_metrics['resilience_score']:.4f}")
    print(f"95% Coverage: {coverage_95 * 100:.2f}%")
    print("=" * 80)

    # ========== STEP 13: SAVE MODELS ==========
    print("\n" + "=" * 80)
    print("STEP 13: SAVING MODELS")
    print("=" * 80)

    try:
        with open('used_features.json', 'w') as f:
            json.dump({'features': features, 'target': target}, f, indent=2)
        print("✓ Feature list saved")

        save_success = save_models_for_streamlit(forecaster, df[features], y, save_dir='./')

        if save_success:
            print("✅ All models saved successfully")
        else:
            print("⚠️ Some models may not have saved properly")
    except Exception as e:
        print(f"⚠️ Error saving models: {e}")

    # ========== STEP 14: CHECK DPO ==========
    print("\n" + "=" * 80)
    print("STEP 14: DPO STATUS CHECK")
    print("=" * 80)

    if os.path.exists('dpo_preferences.json'):
        try:
            with open('dpo_preferences.json', 'r') as f:
                preferences = json.load(f)
            print(f"✓ Found {len(preferences)} preference pairs")

            if len(preferences) >= 50:
                print("✓ DPO training available - run training manually if needed")
            else:
                print(f"ℹ️ Need {50 - len(preferences)} more preferences for DPO training")
        except:
            print("⚠️ Could not read preference file")
    else:
        print("ℹ️ No preference data yet - use Streamlit to collect")

    # ========== STEP 15: LLM INSIGHTS (OPTIONAL) ==========
    print("\n" + "=" * 80)
    print("STEP 15: GENERATING LLM INSIGHTS (Optional)")
    print("=" * 80)

    try:
        print("Attempting to generate LLM insights...")
        llm_insights = forecaster.generate_llm_insights(df, all_metrics, predictions, feature_importance)
        print("✅ LLM insights generated")
    except Exception as e:
        print(f"⚠️ LLM insights skipped: {e}")
        llm_insights = None

    print("\n" + "=" * 80)
    print("✅✅✅ TRAINING COMPLETED SUCCESSFULLY ✅✅✅")
    print("=" * 80)
    print("\nNext steps:")
    print("1. Run: streamlit run streamlit_app_dpo.py")
    print("2. Collect expert preferences (need 50+)")
    print("3. Re-run this script to train DPO model")
    print("=" * 80)

    return forecaster, predictions, resilience_metrics, llm_insights


# Execute with error catching
if __name__ == "__main__":
    print("=" * 80)
    print("SUPPLY CHAIN FORECASTING WITH DPO - TRAINING SCRIPT")
    print("=" * 80)

    try:
        forecaster, predictions, resilience_metrics, llm_insights = main()

        if forecaster is not None:
            print("\n✅ Script completed successfully!")
        else:
            print("\n❌ Script encountered errors - check output above")

    except KeyboardInterrupt:
        print("\n\n⚠️ Training interrupted by user (Ctrl+C)")

    except Exception as e:
        print("\n" + "=" * 80)
        print("CRITICAL ERROR")
        print("=" * 80)
        print(f"Error: {e}")
        print("\nFull traceback:")
        import traceback

        traceback.print_exc()
        print("=" * 80)

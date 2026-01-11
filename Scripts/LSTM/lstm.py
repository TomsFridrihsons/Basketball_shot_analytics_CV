import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import pickle
import json
import os


class ShotPredictionLSTM:
    def __init__(self, data_path):
        """
        Initialize LSTM model for basketball shot prediction
        
        Args:
            data_path: Path to cleaned shot data CSV
        """
        self.data_path = data_path
        self.scaler = StandardScaler()
        self.model = None
        self.max_len = None
        self.history = None
        
        # Features to use for prediction
        self.feature_cols = [
            'ball_x', 'ball_y', 'ball_w', 'ball_h',
            'ball_velocity_x', 'ball_velocity_y',
            'player_x', 'player_y', 'player_w', 'player_h',
            'basket_x', 'basket_y', 'basket_w', 'basket_h',
            'ball_player_dist', 'ball_basket_dist', 'ball_basket_angle',
            'ball_above_player'
        ]
        
        # Initialize config storage
        self.seq_stats = {}
        self.training_config = {}
        self.model_config = {}
        self.train_config = {}
        self.eval_metrics = {}
    
    def load_and_prepare_data(self):
        """Load data and prepare sequences for LSTM"""
        print("=== Loading Data ===")
        df = pd.read_csv(self.data_path)
        
        print(f"Total frames: {len(df)}")
        print(f"Total shots: {df['shot_number'].nunique()}")
        
        # Get shot results
        shot_results = df.groupby('shot_number')['shot_result'].first()
        print(f"Made shots: {(shot_results == 1).sum()}")
        print(f"Missed shots: {(shot_results == 0).sum()}")
        
        # Prepare sequences - one sequence per shot
        X_sequences = []
        y_labels = []
        shot_ids = []
        
        for shot_num in df['shot_number'].unique():
            shot_data = df[df['shot_number'] == shot_num].copy()
            
            # Get features and fill any remaining NaN with 0
            features = shot_data[self.feature_cols].fillna(0).values
            
            # Get label (same for all frames in a shot)
            label = shot_data['shot_result'].iloc[0]
            
            X_sequences.append(features)
            y_labels.append(label)
            shot_ids.append(shot_num)
        
        self.X_raw = X_sequences
        self.y = np.array(y_labels)
        self.shot_ids = shot_ids
        
        print(f"\n=== Sequence Statistics ===")
        seq_lengths = [len(seq) for seq in X_sequences]
        print(f"Min sequence length: {min(seq_lengths)} frames")
        print(f"Max sequence length: {max(seq_lengths)} frames")
        print(f"Mean sequence length: {np.mean(seq_lengths):.1f} frames")
        
        # Store statistics for config
        self.seq_stats = {
            'min_len': int(min(seq_lengths)),
            'max_len': int(max(seq_lengths)),
            'mean_len': float(np.mean(seq_lengths)),
            'total_shots': len(X_sequences),
            'made_shots': int((self.y == 1).sum()),
            'missed_shots': int((self.y == 0).sum())
        }
        
        return X_sequences, y_labels
    
    def pad_sequences(self, X_sequences, max_len=None):
        """Pad sequences to same length for LSTM input"""
        if max_len is None:
            max_len = max(len(seq) for seq in X_sequences)
        
        n_features = X_sequences[0].shape[1]
        X_padded = np.zeros((len(X_sequences), max_len, n_features))
        
        for i, seq in enumerate(X_sequences):
            seq_len = min(len(seq), max_len)
            X_padded[i, :seq_len, :] = seq[:seq_len]
        
        return X_padded, max_len
    
    def prepare_train_test_split(self, test_size=0.2, random_state=42, augment=True):
        """Split data into train and test sets with optional augmentation"""
        print("\n=== Preparing Train/Test Split ===")
        
        # Pad sequences
        X_padded, self.max_len = self.pad_sequences(self.X_raw)
        
        # Data augmentation for training set
        if augment:
            print("Applying data augmentation...")
            X_augmented = []
            y_augmented = []
            
            for i, (seq, label) in enumerate(zip(X_padded, self.y)):
                # Original
                X_augmented.append(seq)
                y_augmented.append(label)
                
                # Add noise (small random variations)
                noise = np.random.normal(0, 0.02, seq.shape)
                X_augmented.append(seq + noise)
                y_augmented.append(label)
                
                # Time shift (shift sequence by 1-3 frames)
                shift = np.random.randint(1, 4)
                shifted = np.roll(seq, shift, axis=0)
                X_augmented.append(shifted)
                y_augmented.append(label)
            
            X_padded = np.array(X_augmented)
            self.y = np.array(y_augmented)
            print(f"Augmented to {len(X_padded)} sequences (3x original)")
        
        # Normalize features
        n_samples, n_timesteps, n_features = X_padded.shape
        X_reshaped = X_padded.reshape(-1, n_features)
        X_scaled = self.scaler.fit_transform(X_reshaped)
        X_scaled = X_scaled.reshape(n_samples, n_timesteps, n_features)
        
        # Split data
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X_scaled, self.y, test_size=test_size, random_state=random_state, stratify=self.y
        )
        
        print(f"Training samples: {len(self.X_train)}")
        print(f"Test samples: {len(self.X_test)}")
        print(f"Sequence length: {self.max_len} frames")
        print(f"Features per frame: {n_features}")
        
        # Store training config
        self.training_config = {
            'test_size': test_size,
            'random_state': random_state,
            'augment': augment,
            'n_train_samples': len(self.X_train),
            'n_test_samples': len(self.X_test)
        }
        
        return self.X_train, self.X_test, self.y_train, self.y_test
    
    def build_model(self, lstm_units=128, dropout_rate=0.4):
        """Build LSTM model"""
        print("\n=== Building LSTM Model ===")
        
        model = keras.Sequential([
            # First LSTM layer
            layers.LSTM(lstm_units, return_sequences=True, 
                       input_shape=(self.max_len, len(self.feature_cols))),
            layers.BatchNormalization(),
            layers.Dropout(dropout_rate),
            
            # Second LSTM layer
            layers.LSTM(lstm_units // 2, return_sequences=True),
            layers.BatchNormalization(),
            layers.Dropout(dropout_rate),
            
            # Third LSTM layer
            layers.LSTM(lstm_units // 4, return_sequences=False),
            layers.BatchNormalization(),
            layers.Dropout(dropout_rate),
            
            # Dense layers
            layers.Dense(64, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(dropout_rate),
            
            layers.Dense(32, activation='relu'),
            layers.Dropout(dropout_rate / 2),
            
            # Output layer (binary classification)
            layers.Dense(1, activation='sigmoid')
        ])
        
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss='binary_crossentropy',
            metrics=['accuracy', keras.metrics.Precision(), keras.metrics.Recall()]
        )
        
        self.model = model
        
        # Store model config
        self.model_config = {
            'lstm_units': lstm_units,
            'dropout_rate': dropout_rate,
            'learning_rate': 0.001,
            'optimizer': 'Adam',
            'loss': 'binary_crossentropy'
        }
        
        print(model.summary())
        return model
    
    def train(self, epochs=100, batch_size=16, patience=15):
        """Train the LSTM model"""
        print("\n=== Training Model ===")
        
        # Callbacks
        early_stop = EarlyStopping(
            monitor='val_loss',
            patience=patience,
            restore_best_weights=True,
            verbose=1
        )
        
        checkpoint = ModelCheckpoint(
            'best_shot_model.h5',
            monitor='val_accuracy',
            save_best_only=True,
            verbose=1
        )
        
        # Train
        history = self.model.fit(
            self.X_train, self.y_train,
            validation_split=0.2,
            epochs=epochs,
            batch_size=batch_size,
            callbacks=[early_stop, checkpoint],
            verbose=1
        )
        
        self.history = history
        
        # Store training hyperparameters
        self.train_config = {
            'epochs_requested': epochs,
            'epochs_trained': len(history.history['loss']),
            'batch_size': batch_size,
            'patience': patience,
            'best_val_accuracy': float(max(history.history['val_accuracy'])),
            'best_val_loss': float(min(history.history['val_loss'])),
            'final_train_accuracy': float(history.history['accuracy'][-1]),
            'final_train_loss': float(history.history['loss'][-1])
        }
        
        return history
    
    def evaluate(self):
        """Evaluate model on test set"""
        print("\n=== Evaluating Model ===")
        
        # Predictions
        y_pred_proba = self.model.predict(self.X_test)
        y_pred = (y_pred_proba > 0.5).astype(int).flatten()
        
        # Metrics
        accuracy = accuracy_score(self.y_test, y_pred)
        print(f"\nTest Accuracy: {accuracy:.4f}")
        
        print("\nClassification Report:")
        print(classification_report(self.y_test, y_pred, 
                                   target_names=['Missed', 'Made']))
        
        print("\nConfusion Matrix:")
        cm = confusion_matrix(self.y_test, y_pred)
        print(cm)
        print(f"\nTrue Negatives (Correctly predicted misses): {cm[0,0]}")
        print(f"False Positives (Predicted made, actually missed): {cm[0,1]}")
        print(f"False Negatives (Predicted missed, actually made): {cm[1,0]}")
        print(f"True Positives (Correctly predicted makes): {cm[1,1]}")
        
        # Calculate precision and recall safely
        precision = float(cm[1, 1] / (cm[1, 1] + cm[0, 1])) if (cm[1, 1] + cm[0, 1]) > 0 else 0.0
        recall = float(cm[1, 1] / (cm[1, 1] + cm[1, 0])) if (cm[1, 1] + cm[1, 0]) > 0 else 0.0
        
        # Store evaluation metrics
        self.eval_metrics = {
            'test_accuracy': float(accuracy),
            'true_negatives': int(cm[0, 0]),
            'false_positives': int(cm[0, 1]),
            'false_negatives': int(cm[1, 0]),
            'true_positives': int(cm[1, 1]),
            'precision': precision,
            'recall': recall
        }
        
        return accuracy, y_pred, y_pred_proba
    
    def plot_training_history(self, save_path='training_history.png'):
        """Plot training history"""
        if self.history is None:
            print("No training history available. Train the model first.")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Accuracy
        axes[0, 0].plot(self.history.history['accuracy'], label='Train')
        axes[0, 0].plot(self.history.history['val_accuracy'], label='Validation')
        axes[0, 0].set_title('Model Accuracy')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Accuracy')
        axes[0, 0].legend()
        axes[0, 0].grid(True)
        
        # Loss
        axes[0, 1].plot(self.history.history['loss'], label='Train')
        axes[0, 1].plot(self.history.history['val_loss'], label='Validation')
        axes[0, 1].set_title('Model Loss')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Loss')
        axes[0, 1].legend()
        axes[0, 1].grid(True)
        
        # Precision
        axes[1, 0].plot(self.history.history['precision'], label='Train')
        axes[1, 0].plot(self.history.history['val_precision'], label='Validation')
        axes[1, 0].set_title('Model Precision')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Precision')
        axes[1, 0].legend()
        axes[1, 0].grid(True)
        
        # Recall
        axes[1, 1].plot(self.history.history['recall'], label='Train')
        axes[1, 1].plot(self.history.history['val_recall'], label='Validation')
        axes[1, 1].set_title('Model Recall')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Recall')
        axes[1, 1].legend()
        axes[1, 1].grid(True)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"\nTraining history saved to '{save_path}'")
        plt.show()
    
    def save_model(self, path='shot_prediction_model.h5'):
        """Save trained model, scaler, and config"""
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        # Save model
        self.model.save(path)
        print(f"\nModel saved to {path}")
        
        # Save scaler for inference
        scaler_path = path.replace('.h5', '_scaler.pkl')
        with open(scaler_path, 'wb') as f:
            pickle.dump(self.scaler, f)
        print(f"Scaler saved to {scaler_path}")
        
        # Build comprehensive config
        config = {
            # Essential for inference
            'max_len': self.max_len,
            'feature_cols': self.feature_cols,
            'n_features': len(self.feature_cols),
            
            # Model architecture info
            'model_config': self.model_config,
            
            # Training info
            'training_config': self.training_config,
            'train_config': self.train_config,
            
            # Data statistics
            'seq_stats': self.seq_stats,
            
            # Evaluation metrics
            'eval_metrics': self.eval_metrics,
            
            # Paths
            'model_path': path,
            'scaler_path': scaler_path
        }
        
        # Save config as pickle
        config_path = path.replace('.h5', '_config.pkl')
        with open(config_path, 'wb') as f:
            pickle.dump(config, f)
        print(f"Config (pickle) saved to {config_path}")
        
        # Also save config as JSON for human readability
        json_config_path = path.replace('.h5', '_config.json')
        with open(json_config_path, 'w') as f:
            json.dump(config, f, indent=4)
        print(f"Config (JSON) saved to {json_config_path}")
        
        # Print config summary
        print("\n=== Config Summary ===")
        print(f"  max_len: {config['max_len']}")
        print(f"  n_features: {config['n_features']}")
        print(f"  feature_cols: {config['feature_cols']}")
        if config['eval_metrics']:
            print(f"  test_accuracy: {config['eval_metrics'].get('test_accuracy', 'N/A'):.4f}")
        if config['train_config']:
            print(f"  epochs_trained: {config['train_config'].get('epochs_trained', 'N/A')}")
            print(f"  best_val_accuracy: {config['train_config'].get('best_val_accuracy', 'N/A'):.4f}")
        
        return config


# =============================================================================
# MAIN EXECUTION
# =============================================================================
if __name__ == "__main__":
    # Configuration - UPDATE THESE PATHS
    DATA_PATH = r"C:\Users\fridr\Documents\HooperAI\data\processed\annotations\basketball_shot_data_clean.csv"
    MODEL_SAVE_PATH = r"C:\Users\fridr\Documents\HooperAI\data\model\basketball_shot_lstm.h5"
    
    # Check if data file exists
    if not os.path.exists(DATA_PATH):
        print(f"ERROR: Data file not found: {DATA_PATH}")
        exit(1)
    
    # Initialize
    print("=" * 60)
    print("BASKETBALL SHOT PREDICTION - LSTM TRAINING")
    print("=" * 60)
    
    shot_lstm = ShotPredictionLSTM(DATA_PATH)
    
    # Step 1: Load and prepare data
    X_sequences, y_labels = shot_lstm.load_and_prepare_data()
    
    # Step 2: Prepare train/test split with augmentation
    X_train, X_test, y_train, y_test = shot_lstm.prepare_train_test_split(
        test_size=0.2, 
        random_state=42,
        augment=True
    )
    
    # Step 3: Build model
    model = shot_lstm.build_model(
        lstm_units=128,
        dropout_rate=0.4
    )
    
    # Step 4: Train model
    history = shot_lstm.train(
        epochs=200,
        batch_size=8,
        patience=30
    )
    
    # Step 5: Evaluate model
    accuracy, y_pred, y_pred_proba = shot_lstm.evaluate()
    
    # Step 6: Plot training history
    shot_lstm.plot_training_history()
    
    # Step 7: Save model, scaler, and config
    config = shot_lstm.save_model(MODEL_SAVE_PATH)
    
    # Final summary
    print("\n" + "=" * 60)
    print("TRAINING COMPLETE")
    print("=" * 60)
    print(f"\nFiles saved:")
    print(f"  - Model:        {MODEL_SAVE_PATH}")
    print(f"  - Scaler:       {MODEL_SAVE_PATH.replace('.h5', '_scaler.pkl')}")
    print(f"  - Config (pkl): {MODEL_SAVE_PATH.replace('.h5', '_config.pkl')}")
    print(f"  - Config (json):{MODEL_SAVE_PATH.replace('.h5', '_config.json')}")
    print(f"\nTest Accuracy: {accuracy:.4f}")
    print("=" * 60)
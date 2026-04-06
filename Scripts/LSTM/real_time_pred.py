import cv2
import numpy as np
import pandas as pd
from ultralytics import YOLO
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, regularizers
import pickle
import os
import json
from datetime import datetime


# Custom layer to replace Lambda (avoids serialization issues)
@keras.utils.register_keras_serializable()
class SumAlongAxis(layers.Layer):
    """Custom layer to sum along axis 1 - replaces Lambda layer"""
    def __init__(self, axis=1, **kwargs):
        super().__init__(**kwargs)
        self.axis = axis
    
    def call(self, x):
        return tf.reduce_sum(x, axis=self.axis)
    
    def get_config(self):
        config = super().get_config()
        config.update({"axis": self.axis})
        return config


def focal_loss(gamma=2.0, alpha=0.25):
    """Focal Loss - needed for model compilation"""
    def focal_loss_fixed(y_true, y_pred):
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.clip_by_value(y_pred, 1e-7, 1 - 1e-7)
        cross_entropy = -y_true * tf.math.log(y_pred) - (1 - y_true) * tf.math.log(1 - y_pred)
        p_t = y_true * y_pred + (1 - y_true) * (1 - y_pred)
        focal_weight = tf.pow(1 - p_t, gamma)
        alpha_weight = y_true * alpha + (1 - y_true) * (1 - alpha)
        focal_loss = alpha_weight * focal_weight * cross_entropy
        return tf.reduce_mean(focal_loss)
    return focal_loss_fixed


def rebuild_lstm_model(max_len, n_features, lstm_units=128, dropout_rate=0.4, 
                       use_attention=True, use_bidirectional=True, l2_reg=0.001):
    """
    Rebuild the LSTM model architecture to match training
    """
    inputs = layers.Input(shape=(max_len, n_features))
    
    # Masking layer for padded sequences
    x = layers.Masking(mask_value=0.0)(inputs)
    
    # First LSTM layer (bidirectional)
    if use_bidirectional:
        x = layers.Bidirectional(
            layers.LSTM(lstm_units, return_sequences=True,
                       kernel_regularizer=regularizers.l2(l2_reg))
        )(x)
    else:
        x = layers.LSTM(lstm_units, return_sequences=True,
                       kernel_regularizer=regularizers.l2(l2_reg))(x)
    
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(dropout_rate)(x)
    
    # Second LSTM layer
    if use_bidirectional:
        x = layers.Bidirectional(
            layers.LSTM(lstm_units // 2, return_sequences=True,
                       kernel_regularizer=regularizers.l2(l2_reg))
        )(x)
    else:
        x = layers.LSTM(lstm_units // 2, return_sequences=True,
                       kernel_regularizer=regularizers.l2(l2_reg))(x)
    
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(dropout_rate)(x)
    
    # Attention mechanism
    if use_attention:
        attention_units = lstm_units // 2 * (2 if use_bidirectional else 1)  # 128 for bidirectional
        
        attention = layers.Dense(1, activation='tanh')(x)
        attention = layers.Flatten()(attention)
        attention = layers.Activation('softmax')(attention)
        attention = layers.RepeatVector(attention_units)(attention)
        attention = layers.Permute([2, 1])(attention)
        
        x = layers.Multiply()([x, attention])
        # Use custom layer instead of Lambda
        x = SumAlongAxis(axis=1)(x)
    else:
        if use_bidirectional:
            x = layers.Bidirectional(
                layers.LSTM(lstm_units // 4, return_sequences=False,
                           kernel_regularizer=regularizers.l2(l2_reg))
            )(x)
        else:
            x = layers.LSTM(lstm_units // 4, return_sequences=False,
                           kernel_regularizer=regularizers.l2(l2_reg))(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(dropout_rate)(x)
    
    # Dense layers
    dense1 = layers.Dense(64, activation='relu',
                         kernel_regularizer=regularizers.l2(l2_reg))(x)
    dense1 = layers.BatchNormalization()(dense1)
    dense1 = layers.Dropout(dropout_rate)(dense1)
    
    dense2 = layers.Dense(32, activation='relu',
                         kernel_regularizer=regularizers.l2(l2_reg))(dense1)
    dense2 = layers.Dropout(dropout_rate / 2)(dense2)
    
    # Output layer
    outputs = layers.Dense(1, activation='sigmoid')(dense2)
    
    model = keras.Model(inputs=inputs, outputs=outputs)
    
    return model


def load_lstm_model_with_weights(model_path, config_path):
    """
    Load LSTM model by rebuilding architecture and loading weights
    """
    print("Loading model configuration...")
    
    # Load config to get model parameters
    with open(config_path, 'rb') as f:
        config = pickle.load(f)
    
    max_len = config.get('max_len', 68)
    n_features = config.get('n_features', 18)
    model_config = config.get('model_config', {})
    
    lstm_units = model_config.get('lstm_units', 128)
    dropout_rate = model_config.get('dropout_rate', 0.4)
    use_attention = model_config.get('use_attention', True)
    use_bidirectional = model_config.get('use_bidirectional', True)
    l2_reg = model_config.get('l2_reg', 0.001)
    
    print(f"  max_len: {max_len}")
    print(f"  n_features: {n_features}")
    print(f"  lstm_units: {lstm_units}")
    print(f"  use_attention: {use_attention}")
    print(f"  use_bidirectional: {use_bidirectional}")
    
    # Rebuild model
    print("\nRebuilding model architecture...")
    model = rebuild_lstm_model(
        max_len=max_len,
        n_features=n_features,
        lstm_units=lstm_units,
        dropout_rate=dropout_rate,
        use_attention=use_attention,
        use_bidirectional=use_bidirectional,
        l2_reg=l2_reg
    )
    
    # Try to load weights from the .keras file
    print("\nLoading weights...")
    
    # Enable unsafe deserialization
    keras.config.enable_unsafe_deserialization()
    
    # Extract weights from the saved model
    import zipfile
    import tempfile
    
    with tempfile.TemporaryDirectory() as tmp_dir:
        # .keras files are zip archives
        with zipfile.ZipFile(model_path, 'r') as zip_ref:
            zip_ref.extractall(tmp_dir)
        
        # Load weights from the extracted files
        weights_path = os.path.join(tmp_dir, 'model.weights.h5')
        if os.path.exists(weights_path):
            try:
                model.load_weights(weights_path)
                print("  ✅ Weights loaded successfully!")
            except Exception as e:
                print(f"  ⚠️ Could not load weights directly: {e}")
                print("  Attempting alternative weight loading...")
                
                # Try loading with by_name
                try:
                    model.load_weights(weights_path, by_name=True, skip_mismatch=True)
                    print("  ✅ Weights loaded with by_name=True")
                except Exception as e2:
                    print(f"  ❌ Weight loading failed: {e2}")
                    raise
        else:
            print(f"  ❌ Weights file not found in archive")
            # List contents for debugging
            print(f"  Archive contents: {os.listdir(tmp_dir)}")
            raise FileNotFoundError("model.weights.h5 not found in .keras archive")
    
    # Compile model (optional, only needed for training/evaluation)
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss=focal_loss(gamma=2.0, alpha=0.3),
        metrics=['accuracy']
    )
    
    return model, config


class RealTimeShotPredictor:
    """
    Real-time basketball shot prediction using YOLO + LSTM
    """
    
    def __init__(self, yolo_model_path, lstm_model_path, scaler_path=None, config_path=None, output_dir=None):
        print("=" * 60)
        print("INITIALIZING SHOT PREDICTOR")
        print("=" * 60)
        
        # Validate paths
        if not os.path.exists(yolo_model_path):
            raise FileNotFoundError(f"YOLO model not found: {yolo_model_path}")
        if not os.path.exists(lstm_model_path):
            raise FileNotFoundError(f"LSTM model not found: {lstm_model_path}")
        
        # Load YOLO model
        print("\nLoading YOLO model...")
        self.yolo_model = YOLO(yolo_model_path)
        print(f"  ✅ YOLO loaded: {yolo_model_path}")
        
        # Initialize defaults
        self.scaler = None
        self.max_sequence_length = 68
        self.optimal_threshold = 0.5
        self.feature_cols = [
            'ball_x', 'ball_y', 'ball_w', 'ball_h',
            'ball_velocity_x', 'ball_velocity_y',
            'player_x', 'player_y', 'player_w', 'player_h',
            'basket_x', 'basket_y', 'basket_w', 'basket_h',
            'ball_player_dist', 'ball_basket_dist', 'ball_basket_angle',
            'ball_above_player'
        ]
        
        # Load LSTM model using weight reconstruction
        print("\nLoading LSTM model...")
        
        if config_path and os.path.exists(config_path):
            try:
                self.lstm_model, config = load_lstm_model_with_weights(lstm_model_path, config_path)
                
                self.max_sequence_length = config.get('max_len', 68)
                self.feature_cols = config.get('feature_cols', self.feature_cols)
                self.optimal_threshold = config.get('optimal_threshold', 0.5)
                
                print(f"\n  ✅ LSTM model loaded successfully")
                print(f"  ✅ max_len: {self.max_sequence_length}")
                print(f"  ✅ n_features: {len(self.feature_cols)}")
                print(f"  ✅ optimal_threshold: {self.optimal_threshold}")
                
                if 'eval_metrics' in config:
                    metrics = config['eval_metrics']
                    print(f"\n  Model Training Metrics:")
                    print(f"    Accuracy:  {metrics.get('test_accuracy', 0)*100:.1f}%")
                    print(f"    Precision: {metrics.get('precision', 0)*100:.1f}%")
                    print(f"    Recall:    {metrics.get('recall', 0)*100:.1f}%")
                    print(f"    F1 Score:  {metrics.get('f1_score', 0)*100:.1f}%")
                    
            except Exception as e:
                print(f"  ❌ Failed to load model with weights: {e}")
                raise RuntimeError(f"Failed to load LSTM model: {e}")
        else:
            raise FileNotFoundError(f"Config file required for model loading: {config_path}")
        
        # Load scaler
        if scaler_path and os.path.exists(scaler_path):
            print(f"\nLoading scaler from {scaler_path}...")
            with open(scaler_path, 'rb') as f:
                self.scaler = pickle.load(f)
            print(f"  ✅ Scaler type: {type(self.scaler).__name__}")
            
            if hasattr(self.scaler, 'center_'):
                print(f"  ✅ RobustScaler fitted with {len(self.scaler.center_)} features")
            elif hasattr(self.scaler, 'mean_'):
                print(f"  ✅ StandardScaler fitted with {len(self.scaler.mean_)} features")
        else:
            from sklearn.preprocessing import RobustScaler
            self.scaler = RobustScaler()
            print("⚠️ WARNING: No scaler file found. Predictions will be inaccurate!")
        
        # Output directory
        self.output_base_dir = output_dir or os.path.join(os.getcwd(), "processed_videos")
        os.makedirs(self.output_base_dir, exist_ok=True)
        
        # Shot tracking
        self.reset_shot_state()
        
        # Statistics tracking
        self.reset_statistics()
        
        # Last known basket position
        self.last_basket_x = None
        self.last_basket_y = None
        self.last_basket_w = None
        self.last_basket_h = None
        
        # Display settings
        self.display_width = 1280
        self.display_max_height = 900
        
        print(f"\n{'=' * 60}")
        print("INITIALIZATION COMPLETE")
        print(f"{'=' * 60}")
        print(f"  Prediction threshold: {self.optimal_threshold}")
        print(f"  Max sequence length: {self.max_sequence_length}")
        print(f"  Output directory: {self.output_base_dir}")

    def reset_statistics(self):
        """Reset statistics tracking"""
        self.statistics = {
            'total_shots_detected': 0,
            'predicted_made': 0,
            'predicted_missed': 0,
            'high_confidence_predictions': 0,
            'low_confidence_predictions': 0,
            'shots': [],
            'processing_start_time': None,
            'processing_end_time': None,
            'video_info': {},
            'model_info': {
                'max_sequence_length': self.max_sequence_length,
                'n_features': len(self.feature_cols),
                'optimal_threshold': self.optimal_threshold
            }
        }
        
    def reset_shot_state(self):
        """Reset shot tracking state"""
        self.possible_shot = False
        self.actual_shot = False
        self.possible_shot_frame_count = 0
        self.pre_recorded_data = []
        self.current_shot_data = []
        self.frame_count = 0
        self.last_prediction = None
        self.last_probability = None
        self.current_shot_start_frame = None
        
    def extract_features(self, ball_boxes, player_boxes, basket_boxes, fps):
        """Extract features from detections"""
        # Ball data
        if len(ball_boxes) > 0:
            ball = ball_boxes[0]
            ball_x = (ball[0] + ball[2]) / 2
            ball_y = (ball[1] + ball[3]) / 2
            ball_w = ball[2] - ball[0]
            ball_h = ball[3] - ball[1]
        else:
            ball_x = ball_y = ball_w = ball_h = None
        
        # Player data (closest to ball)
        if len(player_boxes) > 0 and ball_x is not None:
            closest_player = self._find_closest_player(ball_boxes[0], player_boxes)
            player_x = (closest_player[0] + closest_player[2]) / 2
            player_y = (closest_player[1] + closest_player[3]) / 2
            player_w = closest_player[2] - closest_player[0]
            player_h = closest_player[3] - closest_player[1]
            player_top = closest_player[1]
        else:
            player_x = player_y = player_w = player_h = None
            player_top = None
        
        # Basket data with failsafe
        if len(basket_boxes) > 0:
            basket = basket_boxes[0]
            basket_x = (basket[0] + basket[2]) / 2
            basket_y = (basket[1] + basket[3]) / 2
            basket_w = basket[2] - basket[0]
            basket_h = basket[3] - basket[1]
            
            self.last_basket_x = basket_x
            self.last_basket_y = basket_y
            self.last_basket_w = basket_w
            self.last_basket_h = basket_h
        else:
            basket_x = self.last_basket_x
            basket_y = self.last_basket_y
            basket_w = self.last_basket_w
            basket_h = self.last_basket_h
        
        # Calculate derived features
        if ball_x is not None and player_x is not None:
            ball_player_dist = np.sqrt((ball_x - player_x)**2 + (ball_y - player_y)**2)
            ball_above_player = 1 if ball_y < player_top else 0
        else:
            ball_player_dist = None
            ball_above_player = None
        
        if ball_x is not None and basket_x is not None:
            ball_basket_dist = np.sqrt((ball_x - basket_x)**2 + (ball_y - basket_y)**2)
            ball_basket_angle = np.arctan2(basket_y - ball_y, basket_x - ball_x)
        else:
            ball_basket_dist = None
            ball_basket_angle = None
        
        # Calculate velocity
        ball_velocity_x = ball_velocity_y = None
        prev_data = None
        
        if len(self.current_shot_data) > 0:
            prev_data = self.current_shot_data[-1]
        elif len(self.pre_recorded_data) > 0:
            prev_data = self.pre_recorded_data[-1]
            
        if prev_data is not None and ball_x is not None:
            if prev_data[0] is not None:
                dt = 1.0 / fps if fps > 0 else 1/30
                ball_velocity_x = (ball_x - prev_data[0]) / dt
                ball_velocity_y = (ball_y - prev_data[1]) / dt
        
        features = [
            ball_x, ball_y, ball_w, ball_h,
            ball_velocity_x, ball_velocity_y,
            player_x, player_y, player_w, player_h,
            basket_x, basket_y, basket_w, basket_h,
            ball_player_dist, ball_basket_dist, ball_basket_angle,
            ball_above_player
        ]
        
        return features
    
    def _find_closest_player(self, ball_box, player_boxes):
        """Find player closest to ball"""
        ball_center = np.array([
            (ball_box[0] + ball_box[2]) / 2,
            (ball_box[1] + ball_box[3]) / 2
        ])
        
        min_dist = float('inf')
        closest_player = player_boxes[0]
        
        for player in player_boxes:
            player_center = np.array([
                (player[0] + player[2]) / 2,
                (player[1] + player[3]) / 2
            ])
            dist = np.linalg.norm(ball_center - player_center)
            if dist < min_dist:
                min_dist = dist
                closest_player = player
        
        return closest_player
    
    def detect_shot(self, ball_boxes, player_boxes):
        """Detect if shot is in progress with two-stage detection"""
        if len(ball_boxes) == 0 or len(player_boxes) == 0:
            if self.possible_shot and not self.actual_shot:
                self.possible_shot_frame_count += 1
                if self.possible_shot_frame_count > 20:
                    print("[POSSIBLE SHOT CANCELLED - No detections]")
                    self.possible_shot = False
                    self.possible_shot_frame_count = 0
                    self.pre_recorded_data = []
            return False, False, False
        
        ball = ball_boxes[0]
        ball_y_bottom = ball[3]
        ball_y_center = (ball[1] + ball[3]) / 2
        
        player_tops = [p[1] for p in player_boxes]
        highest_player_top = min(player_tops)
        
        ball_above_players = ball_y_bottom < highest_player_top
        ball_below_player_top = ball_y_center > highest_player_top
        
        possible_shot_started = False
        actual_shot_started = False
        shot_ended = False
        
        if not self.possible_shot and not self.actual_shot and ball_above_players:
            self.possible_shot = True
            self.possible_shot_frame_count = 0
            self.pre_recorded_data = []
            possible_shot_started = True
            self.current_shot_start_frame = self.frame_count
            print("\n[POSSIBLE SHOT DETECTED - Monitoring...]")
        
        if self.possible_shot and not self.actual_shot:
            self.possible_shot_frame_count += 1
            
            if ball_above_players:
                if self.possible_shot_frame_count >= 20:
                    self.actual_shot = True
                    self.possible_shot = False
                    self.current_shot_data = self.pre_recorded_data.copy()
                    self.pre_recorded_data = []
                    actual_shot_started = True
                    print(f"[ACTUAL SHOT CONFIRMED - {len(self.current_shot_data)} frames pre-recorded]")
            else:
                print(f"[POSSIBLE SHOT CANCELLED - Only {self.possible_shot_frame_count} frames]")
                self.possible_shot = False
                self.possible_shot_frame_count = 0
                self.pre_recorded_data = []
        
        if self.actual_shot and ball_below_player_top:
            self.actual_shot = False
            shot_ended = True
        
        return possible_shot_started, actual_shot_started, shot_ended
    
    def predict_shot_outcome(self):
        """Make prediction using optimal threshold from training"""
        if len(self.current_shot_data) < 5:
            print(f"[WARNING] Not enough frames for prediction: {len(self.current_shot_data)}")
            return None, None
        
        sequence = np.array(self.current_shot_data, dtype=np.float32)
        sequence = np.nan_to_num(sequence, nan=0.0)
        
        print(f"\n[DEBUG] Sequence shape: {sequence.shape}")
        
        max_len = self.max_sequence_length
        n_features = len(self.feature_cols)
        
        padded_sequence = np.zeros((1, max_len, n_features), dtype=np.float32)
        seq_len = min(len(sequence), max_len)
        padded_sequence[0, :seq_len, :] = sequence[:seq_len]
        
        n_samples, n_timesteps, n_feats = padded_sequence.shape
        X_reshaped = padded_sequence.reshape(-1, n_feats)
        
        if self.scaler is None:
            print("[ERROR] Scaler is None!")
            return None, None
        
        is_fitted = hasattr(self.scaler, 'center_') or hasattr(self.scaler, 'mean_')
        
        if not is_fitted:
            print("[ERROR] Scaler is not fitted!")
            X_scaled = padded_sequence
        else:
            try:
                X_scaled = self.scaler.transform(X_reshaped)
                X_scaled = X_scaled.reshape(n_samples, n_timesteps, n_feats)
            except Exception as e:
                print(f"[ERROR] Scaler transform failed: {e}")
                X_scaled = padded_sequence
        
        prediction_proba = self.lstm_model.predict(X_scaled, verbose=0)[0][0]
        prediction = 1 if prediction_proba >= self.optimal_threshold else 0
        
        print(f"[DEBUG] Raw probability: {prediction_proba:.4f}")
        print(f"[DEBUG] Threshold: {self.optimal_threshold:.3f}")
        print(f"[DEBUG] Prediction: {'MADE' if prediction == 1 else 'MISSED'}")
        
        return prediction, float(prediction_proba)
    
    def get_confidence_level(self, probability):
        """Categorize confidence level"""
        distance_from_threshold = abs(probability - self.optimal_threshold)
        
        if distance_from_threshold > 0.3:
            return 'HIGH', min(probability, 1 - probability) + 0.5
        elif distance_from_threshold > 0.15:
            return 'MEDIUM', 0.5 + distance_from_threshold
        else:
            return 'LOW', 0.5 + distance_from_threshold * 0.5
    
    def create_output_folder(self, video_path):
        """Create timestamped output folder"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        video_name = os.path.splitext(os.path.basename(video_path))[0]
        folder_name = f"{timestamp}_{video_name}"
        
        output_folder = os.path.join(self.output_base_dir, folder_name)
        os.makedirs(output_folder, exist_ok=True)
        
        return output_folder, timestamp
    
    def save_statistics(self, output_folder, video_path):
        """Save statistics to files"""
        total_shots = self.statistics['total_shots_detected']
        made_shots = self.statistics['predicted_made']
        missed_shots = self.statistics['predicted_missed']
        
        make_percentage = (made_shots / total_shots * 100) if total_shots > 0 else 0.0
        
        self.statistics['summary'] = {
            'total_shots': total_shots,
            'predicted_made': made_shots,
            'predicted_missed': missed_shots,
            'predicted_make_percentage': round(make_percentage, 2),
            'high_confidence_predictions': self.statistics['high_confidence_predictions'],
            'low_confidence_predictions': self.statistics['low_confidence_predictions'],
            'input_video': video_path,
            'threshold_used': self.optimal_threshold,
            'processing_duration_seconds': None
        }
        
        if self.statistics['processing_start_time'] and self.statistics['processing_end_time']:
            start = datetime.fromisoformat(self.statistics['processing_start_time'])
            end = datetime.fromisoformat(self.statistics['processing_end_time'])
            duration = (end - start).total_seconds()
            self.statistics['summary']['processing_duration_seconds'] = round(duration, 2)
        
        json_path = os.path.join(output_folder, "statistics.json")
        with open(json_path, 'w') as f:
            json.dump(self.statistics, f, indent=4, default=str)
        print(f"Statistics saved to: {json_path}")
        
        if len(self.statistics['shots']) > 0:
            csv_path = os.path.join(output_folder, "shots.csv")
            df = pd.DataFrame(self.statistics['shots'])
            df.to_csv(csv_path, index=False)
            print(f"Shots CSV saved to: {csv_path}")
        
        summary_path = os.path.join(output_folder, "summary.txt")
        with open(summary_path, 'w') as f:
            f.write("=" * 50 + "\n")
            f.write("BASKETBALL SHOT PREDICTION - SUMMARY\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"Input Video: {os.path.basename(video_path)}\n")
            f.write(f"Processing Time: {self.statistics['summary']['processing_duration_seconds']} seconds\n")
            f.write(f"Prediction Threshold: {self.optimal_threshold:.3f}\n\n")
            f.write("-" * 50 + "\n")
            f.write("RESULTS\n")
            f.write("-" * 50 + "\n")
            f.write(f"Total Shots Detected: {total_shots}\n")
            f.write(f"Predicted Made: {made_shots}\n")
            f.write(f"Predicted Missed: {missed_shots}\n")
            f.write(f"Predicted Make %: {make_percentage:.1f}%\n\n")
        
        print(f"Summary saved to: {summary_path}")
        return json_path, summary_path
    
    def process_video(self, video_path, display=True, save_video=True, verbose=True):
        """Process video with real-time shot prediction"""
        self.reset_statistics()
        self.reset_shot_state()
        
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            raise ValueError(f"Could not open video: {video_path}")
        
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        duration_seconds = total_frames / fps if fps > 0 else 0
        
        self.statistics['video_info'] = {
            'input_path': video_path,
            'fps': fps,
            'total_frames': total_frames,
            'width': frame_width,
            'height': frame_height,
            'duration_seconds': round(duration_seconds, 2)
        }
        
        output_folder, timestamp = self.create_output_folder(video_path)
        
        video_writer = None
        if save_video:
            output_video_path = os.path.join(output_folder, "processed_video.mp4")
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            video_writer = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))
        
        self.statistics['processing_start_time'] = datetime.now().isoformat()
        
        print(f"\n{'=' * 60}")
        print("PROCESSING VIDEO")
        print(f"{'=' * 60}")
        print(f"Input: {video_path}")
        print(f"Output folder: {output_folder}")
        print(f"FPS: {fps}, Total frames: {total_frames}")
        print(f"Prediction threshold: {self.optimal_threshold:.3f}")
        
        if display:
            print("\nPress 'q' to quit, 'r' to reset")
        
        last_progress = -1
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            self.frame_count += 1
            
            progress = int((self.frame_count / total_frames) * 100)
            if progress % 10 == 0 and progress != last_progress:
                print(f"[PROGRESS] {progress}%")
                last_progress = progress
            
            results = self.yolo_model(frame, conf=0.70, verbose=False)[0]
            
            ball_boxes = []
            player_boxes = []
            basket_boxes = []
            
            for box in results.boxes:
                cls = int(box.cls[0])
                xyxy = box.xyxy[0].cpu().numpy()
                
                if cls == 0:
                    ball_boxes.append(xyxy)
                elif cls == 1:
                    basket_boxes.append(xyxy)
                elif cls == 2:
                    player_boxes.append(xyxy)
            
            possible_shot_started, actual_shot_started, shot_ended = self.detect_shot(ball_boxes, player_boxes)
            
            if possible_shot_started:
                self.pre_recorded_data = []
            
            if actual_shot_started:
                self.last_prediction = None
                self.last_probability = None
            
            if self.possible_shot or self.actual_shot:
                features = self.extract_features(ball_boxes, player_boxes, basket_boxes, fps)
                
                if self.possible_shot:
                    self.pre_recorded_data.append(features)
                elif self.actual_shot:
                    self.current_shot_data.append(features)
            
            if shot_ended and len(self.current_shot_data) > 0:
                prediction, probability = self.predict_shot_outcome()
                self.last_prediction = prediction
                self.last_probability = probability
                
                if prediction is not None:
                    result_text = "MADE" if prediction == 1 else "MISSED"
                    conf_level, _ = self.get_confidence_level(probability)
                    
                    if verbose:
                        print(f"[PREDICTION] {result_text} (prob: {probability:.2%}, conf: {conf_level})")
                    
                    self.statistics['total_shots_detected'] += 1
                    if prediction == 1:
                        self.statistics['predicted_made'] += 1
                    else:
                        self.statistics['predicted_missed'] += 1
                    
                    if conf_level == 'HIGH':
                        self.statistics['high_confidence_predictions'] += 1
                    else:
                        self.statistics['low_confidence_predictions'] += 1
                    
                    shot_info = {
                        'shot_number': self.statistics['total_shots_detected'],
                        'prediction': int(prediction),
                        'prediction_text': result_text,
                        'confidence': float(probability),
                        'confidence_level': conf_level,
                        'start_frame': self.current_shot_start_frame,
                        'end_frame': self.frame_count,
                        'duration_frames': len(self.current_shot_data)
                    }
                    self.statistics['shots'].append(shot_info)
                
                self.current_shot_data = []
            
            # Draw overlays
            annotated_frame = results.plot()
            
            if self.possible_shot:
                cv2.rectangle(annotated_frame, (10, 10), (700, 60), (0, 165, 255), -1)
                cv2.putText(annotated_frame, f"POSSIBLE SHOT... ({self.possible_shot_frame_count}/10)", 
                           (20, 45), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
            elif self.actual_shot:
                cv2.rectangle(annotated_frame, (10, 10), (700, 60), (0, 255, 0), -1)
                cv2.putText(annotated_frame, f"TRACKING ({len(self.current_shot_data)} frames)", 
                           (20, 45), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
            elif self.last_prediction is not None:
                color = (0, 255, 0) if self.last_prediction == 1 else (0, 0, 255)
                text = "MADE" if self.last_prediction == 1 else "MISSED"
                cv2.rectangle(annotated_frame, (10, 10), (400, 60), color, -1)
                cv2.putText(annotated_frame, f"{text} ({self.last_probability:.1%})", 
                           (20, 45), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            
            # Counter
            counter = f"Shots: {self.statistics['total_shots_detected']} | Made: {self.statistics['predicted_made']}"
            cv2.rectangle(annotated_frame, (10, frame_height - 40), (400, frame_height - 5), (0, 0, 0), -1)
            cv2.putText(annotated_frame, counter, (20, frame_height - 15),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            if video_writer is not None:
                video_writer.write(annotated_frame)
            
            if display:
                h, w = annotated_frame.shape[:2]
                scale = min(self.display_width / w, self.display_max_height / h)
                resized = cv2.resize(annotated_frame, (int(w * scale), int(h * scale)))
                cv2.imshow('Shot Prediction', resized)
                
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('r'):
                    self.reset_shot_state()
        
        cap.release()
        if video_writer:
            video_writer.release()
        if display:
            cv2.destroyAllWindows()
        
        self.statistics['processing_end_time'] = datetime.now().isoformat()
        self.save_statistics(output_folder, video_path)
        
        print(f"\n{'=' * 60}")
        print("COMPLETE")
        print(f"{'=' * 60}")
        print(f"Shots: {self.statistics['total_shots_detected']}")
        print(f"Made: {self.statistics['predicted_made']}")
        print(f"Missed: {self.statistics['predicted_missed']}")
        print(f"Output: {output_folder}")
        
        return output_folder, self.statistics


# =============================================================================
# MAIN
# =============================================================================
if __name__ == "__main__":
    
    YOLO_MODEL = r"C:\Users\fridr\Documents\HooperAI\data\models\yolo26m_best.pt"
    LSTM_MODEL = r"C:\Users\fridr\Documents\HooperAI\data\models\basketball_shot_lstm_v2.keras"
    SCALER_PATH = r"C:\Users\fridr\Documents\HooperAI\data\models\basketball_shot_lstm_v2_scaler.pkl"
    CONFIG_PATH = r"C:\Users\fridr\Documents\HooperAI\data\models\basketball_shot_lstm_v2_config.pkl"
    VIDEO_PATH = r"C:\Users\fridr\Documents\HooperAI\data\raw\videos\20250711_171647.mp4"
    OUTPUT_DIR = r"C:\Users\fridr\Documents\HooperAI\data\processed\predictions\new_model"
    
    print("=" * 60)
    print("BASKETBALL SHOT PREDICTION")
    print("=" * 60)
    
    print("\nChecking files...")
    for name, path in [('YOLO', YOLO_MODEL), ('LSTM', LSTM_MODEL), 
                       ('Scaler', SCALER_PATH), ('Config', CONFIG_PATH), ('Video', VIDEO_PATH)]:
        status = "✅" if os.path.exists(path) else "❌"
        print(f"  {status} {name}")
    
    predictor = RealTimeShotPredictor(
        yolo_model_path=YOLO_MODEL,
        lstm_model_path=LSTM_MODEL,
        scaler_path=SCALER_PATH,
        config_path=CONFIG_PATH,
        output_dir=OUTPUT_DIR
    )
    
    predictor.process_video(VIDEO_PATH, display=True, save_video=True, verbose=True)
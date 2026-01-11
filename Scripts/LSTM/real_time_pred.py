import cv2
import numpy as np
import pandas as pd
from ultralytics import YOLO
import tensorflow as tf
from tensorflow import keras
import pickle
import os
import json
from datetime import datetime


class RealTimeShotPredictor:
    def __init__(self, yolo_model_path, lstm_model_path, scaler_path=None, config_path=None, output_dir=None):
        """
        Real-time shot prediction on video
        
        Args:
            yolo_model_path: Path to YOLO detection model
            lstm_model_path: Path to trained LSTM model
            scaler_path: Path to saved scaler (optional)
            config_path: Path to saved config with max_len (optional)
            output_dir: Base directory for saving outputs (optional)
        """
        # Load models
        print("Loading models...")
        self.yolo_model = YOLO(yolo_model_path)
        self.lstm_model = keras.models.load_model(lstm_model_path)
        
        # Initialize defaults
        self.scaler = None
        self.max_sequence_length = 100
        self.feature_cols = [
            'ball_x', 'ball_y', 'ball_w', 'ball_h',
            'ball_velocity_x', 'ball_velocity_y',
            'player_x', 'player_y', 'player_w', 'player_h',
            'basket_x', 'basket_y', 'basket_w', 'basket_h',
            'ball_player_dist', 'ball_basket_dist', 'ball_basket_angle',
            'ball_above_player'
        ]
        
        # Load scaler
        if scaler_path and os.path.exists(scaler_path):
            print(f"Loading scaler from {scaler_path}...")
            with open(scaler_path, 'rb') as f:
                self.scaler = pickle.load(f)
            print(f"  Scaler loaded: {type(self.scaler).__name__}")
            if hasattr(self.scaler, 'mean_'):
                print(f"  Scaler fitted with {len(self.scaler.mean_)} features")
        else:
            from sklearn.preprocessing import StandardScaler
            self.scaler = StandardScaler()
            print("WARNING: No scaler file found. Using unfitted scaler - predictions WILL be wrong!")
        
        # Load config
        if config_path and os.path.exists(config_path):
            print(f"Loading config from {config_path}...")
            with open(config_path, 'rb') as f:
                config = pickle.load(f)
            self.max_sequence_length = config.get('max_len', 100)
            self.feature_cols = config.get('feature_cols', self.feature_cols)
            print(f"  max_len: {self.max_sequence_length}")
            print(f"  n_features: {len(self.feature_cols)}")
        else:
            print(f"WARNING: No config file found. Using defaults.")
        
        # Output directory
        self.output_base_dir = output_dir or os.path.join(os.getcwd(), "processed_videos")
        
        # Shot tracking
        self.reset_shot_state()
        
        # Statistics tracking
        self.reset_statistics()
        
        # Last known basket position (failsafe)
        self.last_basket_x = None
        self.last_basket_y = None
        self.last_basket_w = None
        self.last_basket_h = None
        
        # Display settings
        self.display_width = 1280
        self.display_max_height = 900
        
        print("\n=== Initialization Complete ===")
        print(f"Scaler: {type(self.scaler).__name__}")
        print(f"Max sequence length: {self.max_sequence_length}")
        print(f"Number of features: {len(self.feature_cols)}")
        print(f"Output directory: {self.output_base_dir}")
    
    def reset_statistics(self):
        """Reset statistics tracking"""
        self.statistics = {
            'total_shots_detected': 0,
            'predicted_made': 0,
            'predicted_missed': 0,
            'shots': [],  # Detailed info for each shot
            'processing_start_time': None,
            'processing_end_time': None,
            'video_info': {},
            'model_info': {
                'max_sequence_length': self.max_sequence_length,
                'n_features': len(self.feature_cols)
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
        """Extract features from detections (same as training)"""
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
            
            # Update last known position
            self.last_basket_x = basket_x
            self.last_basket_y = basket_y
            self.last_basket_w = basket_w
            self.last_basket_h = basket_h
        else:
            # Use last known basket position (basket doesn't move)
            basket_x = self.last_basket_x
            basket_y = self.last_basket_y
            basket_w = self.last_basket_w
            basket_h = self.last_basket_h
        
        # Calculate additional features
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
        
        # Calculate velocity - check pre_recorded_data first, then current_shot_data
        ball_velocity_x = ball_velocity_y = None
        prev_data = None
        
        if len(self.current_shot_data) > 0:
            prev_data = self.current_shot_data[-1]
        elif len(self.pre_recorded_data) > 0:
            prev_data = self.pre_recorded_data[-1]
            
        if prev_data is not None and ball_x is not None:
            if prev_data[0] is not None:  # prev ball_x
                dt = 1.0 / fps
                ball_velocity_x = (ball_x - prev_data[0]) / dt
                ball_velocity_y = (ball_y - prev_data[1]) / dt
        
        # Return features in correct order
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
        """
        Detect if shot is in progress with two-stage detection:
        1. Possible shot: Ball above players
        2. Actual shot: Possible shot confirmed after 10 frames
        """
        if len(ball_boxes) == 0 or len(player_boxes) == 0:
            # No detections - check if we should cancel possible shot
            if self.possible_shot and not self.actual_shot:
                self.possible_shot_frame_count += 1
                if self.possible_shot_frame_count > 20:  # Too long without detections
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
        
        # Stage 1: Detect possible shot
        if not self.possible_shot and not self.actual_shot and ball_above_players:
            self.possible_shot = True
            self.possible_shot_frame_count = 0
            self.pre_recorded_data = []
            possible_shot_started = True
            self.current_shot_start_frame = self.frame_count
            print("\n[POSSIBLE SHOT DETECTED - Monitoring...]")
        
        # Stage 2: Confirm actual shot after 10 frames
        if self.possible_shot and not self.actual_shot:
            self.possible_shot_frame_count += 1
            
            # Check if still above players
            if ball_above_players:
                if self.possible_shot_frame_count >= 10:
                    # Confirm as actual shot
                    self.actual_shot = True
                    self.possible_shot = False
                    # Transfer pre-recorded data to current shot
                    self.current_shot_data = self.pre_recorded_data.copy()
                    self.pre_recorded_data = []
                    actual_shot_started = True
                    print(f"[ACTUAL SHOT CONFIRMED - {len(self.current_shot_data)} frames pre-recorded]")
            else:
                # Ball dropped before confirmation - discard
                print(f"[POSSIBLE SHOT CANCELLED - Only {self.possible_shot_frame_count} frames]")
                self.possible_shot = False
                self.possible_shot_frame_count = 0
                self.pre_recorded_data = []
        
        # End actual shot
        if self.actual_shot and ball_below_player_top:
            self.actual_shot = False
            shot_ended = True
        
        return possible_shot_started, actual_shot_started, shot_ended
    
    def predict_shot_outcome(self):
        """Make prediction on current shot data"""
        if len(self.current_shot_data) < 5:  # Need minimum frames
            print(f"[WARNING] Not enough frames for prediction: {len(self.current_shot_data)}")
            return None, None
        
        # Convert to numpy array and fill NaN with 0
        sequence = np.array(self.current_shot_data, dtype=np.float32)
        sequence = np.nan_to_num(sequence, nan=0.0)
        
        print(f"\n[DEBUG] Sequence shape before padding: {sequence.shape}")
        print(f"[DEBUG] First frame features (sample): {sequence[0][:5]}")
        
        # Pad sequence to match training length
        max_len = self.max_sequence_length
        n_features = len(self.feature_cols)
        
        padded_sequence = np.zeros((1, max_len, n_features), dtype=np.float32)
        seq_len = min(len(sequence), max_len)
        padded_sequence[0, :seq_len, :] = sequence[:seq_len]
        
        print(f"[DEBUG] Padded sequence shape: {padded_sequence.shape}")
        
        # Scale features
        n_samples, n_timesteps, n_features = padded_sequence.shape
        X_reshaped = padded_sequence.reshape(-1, n_features)
        
        # Check if scaler is fitted
        if self.scaler is None:
            print("[ERROR] Scaler is None!")
            return None, None
            
        if not hasattr(self.scaler, 'mean_'):
            print("[ERROR] Scaler is not fitted (no mean_ attribute)!")
            print("[WARNING] Using unscaled data - predictions will be inaccurate")
            X_scaled = padded_sequence
        else:
            try:
                X_scaled = self.scaler.transform(X_reshaped)
                X_scaled = X_scaled.reshape(n_samples, n_timesteps, n_features)
                print(f"[DEBUG] Scaling successful")
                print(f"[DEBUG] Scaled features (sample): {X_scaled[0, 0, :5]}")
            except Exception as e:
                print(f"[ERROR] Scaler transform failed: {e}")
                print("[WARNING] Using unscaled data - predictions will be inaccurate")
                X_scaled = padded_sequence
        
        # Predict
        prediction_proba = self.lstm_model.predict(X_scaled, verbose=0)[0][0]
        prediction = 1 if prediction_proba > 0.5 else 0
        
        print(f"[DEBUG] Raw prediction probability: {prediction_proba}")
        
        return prediction, prediction_proba
    
    def create_output_folder(self, video_path):
        """Create timestamped output folder for this run"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        video_name = os.path.splitext(os.path.basename(video_path))[0]
        folder_name = f"{timestamp}_{video_name}"
        
        output_folder = os.path.join(self.output_base_dir, folder_name)
        os.makedirs(output_folder, exist_ok=True)
        
        return output_folder, timestamp
    
    def save_statistics(self, output_folder, video_path):
        """Save statistics to JSON and CSV files"""
        # Calculate summary statistics
        total_shots = self.statistics['total_shots_detected']
        made_shots = self.statistics['predicted_made']
        missed_shots = self.statistics['predicted_missed']
        
        if total_shots > 0:
            make_percentage = (made_shots / total_shots) * 100
        else:
            make_percentage = 0.0
        
        # Add summary to statistics
        self.statistics['summary'] = {
            'total_shots': total_shots,
            'predicted_made': made_shots,
            'predicted_missed': missed_shots,
            'predicted_make_percentage': round(make_percentage, 2),
            'input_video': video_path,
            'processing_duration_seconds': None
        }
        
        # Calculate processing duration
        if self.statistics['processing_start_time'] and self.statistics['processing_end_time']:
            start = datetime.fromisoformat(self.statistics['processing_start_time'])
            end = datetime.fromisoformat(self.statistics['processing_end_time'])
            duration = (end - start).total_seconds()
            self.statistics['summary']['processing_duration_seconds'] = round(duration, 2)
        
        # Save as JSON
        json_path = os.path.join(output_folder, "statistics.json")
        with open(json_path, 'w') as f:
            json.dump(self.statistics, f, indent=4, default=str)
        print(f"Statistics saved to: {json_path}")
        
        # Save shots as CSV for easy analysis
        if len(self.statistics['shots']) > 0:
            csv_path = os.path.join(output_folder, "shots.csv")
            df = pd.DataFrame(self.statistics['shots'])
            df.to_csv(csv_path, index=False)
            print(f"Shots CSV saved to: {csv_path}")
        
        # Save summary as text file for quick viewing
        summary_path = os.path.join(output_folder, "summary.txt")
        with open(summary_path, 'w') as f:
            f.write("=" * 50 + "\n")
            f.write("BASKETBALL SHOT PREDICTION - SUMMARY\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"Input Video: {os.path.basename(video_path)}\n")
            f.write(f"Processing Time: {self.statistics['summary']['processing_duration_seconds']} seconds\n\n")
            f.write("-" * 50 + "\n")
            f.write("RESULTS\n")
            f.write("-" * 50 + "\n")
            f.write(f"Total Shots Detected: {total_shots}\n")
            f.write(f"Predicted Made: {made_shots}\n")
            f.write(f"Predicted Missed: {missed_shots}\n")
            f.write(f"Predicted Make %: {make_percentage:.1f}%\n\n")
            f.write("-" * 50 + "\n")
            f.write("VIDEO INFO\n")
            f.write("-" * 50 + "\n")
            for key, value in self.statistics['video_info'].items():
                f.write(f"{key}: {value}\n")
            f.write("\n")
            f.write("-" * 50 + "\n")
            f.write("SHOT DETAILS\n")
            f.write("-" * 50 + "\n")
            for i, shot in enumerate(self.statistics['shots'], 1):
                result = "MADE" if shot['prediction'] == 1 else "MISSED"
                f.write(f"Shot {i}: {result} (confidence: {shot['confidence']:.1%}, "
                       f"frames: {shot['start_frame']}-{shot['end_frame']})\n")
        print(f"Summary saved to: {summary_path}")
        
        return json_path, summary_path
    
    def process_video(self, video_path, display=True, save_video=True, verbose=True):
        """
        Process video with real-time shot prediction
        
        Args:
            video_path: Path to input video
            display: Whether to display the video while processing (default: True)
            save_video: Whether to save the processed video (default: True)
            verbose: Whether to print detailed logs (default: True)
        
        Returns:
            output_folder: Path to folder containing outputs
            statistics: Dictionary of statistics
        """
        # Reset statistics for new run
        self.reset_statistics()
        self.reset_shot_state()
        
        # Open video
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            raise ValueError(f"Could not open video: {video_path}")
        
        # Get video properties
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        duration_seconds = total_frames / fps if fps > 0 else 0
        
        # Store video info
        self.statistics['video_info'] = {
            'input_path': video_path,
            'fps': fps,
            'total_frames': total_frames,
            'width': frame_width,
            'height': frame_height,
            'duration_seconds': round(duration_seconds, 2)
        }
        
        # Create output folder
        output_folder, timestamp = self.create_output_folder(video_path)
        
        # Initialize video writer if saving
        video_writer = None
        output_video_path = None
        if save_video:
            output_video_path = os.path.join(output_folder, "processed_video.mp4")
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            video_writer = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))
        
        # Record start time
        self.statistics['processing_start_time'] = datetime.now().isoformat()
        
        print(f"\n{'=' * 60}")
        print("PROCESSING VIDEO")
        print(f"{'=' * 60}")
        print(f"Input: {video_path}")
        print(f"Output folder: {output_folder}")
        print(f"FPS: {fps}, Total frames: {total_frames}")
        print(f"Resolution: {frame_width}x{frame_height}")
        print(f"Duration: {duration_seconds:.1f} seconds")
        print(f"Display: {display}, Save video: {save_video}")
        print(f"Max sequence length: {self.max_sequence_length}")
        
        if display:
            print("\nPress 'q' to quit, 'r' to reset shot manually")
        print(f"{'=' * 60}\n")
        
        # Progress tracking
        last_progress = -1
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            self.frame_count += 1
            
            # Print progress every 10%
            progress = int((self.frame_count / total_frames) * 100)
            if progress % 10 == 0 and progress != last_progress:
                print(f"[PROGRESS] {progress}% ({self.frame_count}/{total_frames} frames)")
                last_progress = progress
            
            # Run YOLO detection
            results = self.yolo_model(frame, conf=0.75, verbose=False)[0]
            
            # Extract detections
            ball_boxes = []
            player_boxes = []
            basket_boxes = []
            
            for box in results.boxes:
                cls = int(box.cls[0])
                conf = float(box.conf[0])
                xyxy = box.xyxy[0].cpu().numpy()
                
                if cls == 0:  # Ball
                    ball_boxes.append(xyxy)
                elif cls == 1:  # Basket
                    basket_boxes.append(xyxy)
                elif cls == 2:  # Player
                    player_boxes.append(xyxy)
            
            # Detect shot start/end with two-stage detection
            possible_shot_started, actual_shot_started, shot_ended = self.detect_shot(ball_boxes, player_boxes)
            
            if possible_shot_started:
                self.pre_recorded_data = []
            
            if actual_shot_started:
                self.last_prediction = None
                self.last_probability = None
            
            # Collect features
            if self.possible_shot or self.actual_shot:
                features = self.extract_features(ball_boxes, player_boxes, basket_boxes, fps)
                
                # Store in pre-recorded buffer if possible shot
                if self.possible_shot:
                    self.pre_recorded_data.append(features)
                # Store in current shot data if actual shot
                elif self.actual_shot:
                    self.current_shot_data.append(features)
            
            # Make prediction when shot ends
            if shot_ended and len(self.current_shot_data) > 0:
                prediction, probability = self.predict_shot_outcome()
                self.last_prediction = prediction
                self.last_probability = probability
                
                if prediction is not None:
                    result_text = "MADE" if prediction == 1 else "MISSED"
                    
                    if verbose:
                        print(f"[PREDICTION] {result_text} (confidence: {probability:.2%}, frames: {len(self.current_shot_data)})")
                    
                    # Update statistics
                    self.statistics['total_shots_detected'] += 1
                    if prediction == 1:
                        self.statistics['predicted_made'] += 1
                    else:
                        self.statistics['predicted_missed'] += 1
                    
                    # Record shot details
                    shot_info = {
                        'shot_number': self.statistics['total_shots_detected'],
                        'prediction': int(prediction),
                        'prediction_text': result_text,
                        'confidence': float(probability),
                        'start_frame': self.current_shot_start_frame,
                        'end_frame': self.frame_count,
                        'duration_frames': len(self.current_shot_data),
                        'timestamp_seconds': round(self.frame_count / fps, 2) if fps > 0 else 0
                    }
                    self.statistics['shots'].append(shot_info)
                
                # Reset for next shot
                self.current_shot_data = []
            
            # Draw annotated frame
            annotated_frame = results.plot()
            
            # Add prediction overlay
            if self.possible_shot:
                status_text = f"POSSIBLE SHOT... ({self.possible_shot_frame_count}/10 frames)"
                cv2.rectangle(annotated_frame, (10, 10), (700, 60), (0, 165, 255), -1)  # Orange
                cv2.putText(annotated_frame, status_text, (20, 45), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
            
            elif self.actual_shot:
                status_text = f"TRACKING SHOT... ({len(self.current_shot_data)} frames)"
                cv2.rectangle(annotated_frame, (10, 10), (700, 60), (0, 255, 0), -1)  # Green
                cv2.putText(annotated_frame, status_text, (20, 45), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
            
            elif self.last_prediction is not None:
                # Show last prediction
                if self.last_prediction == 1:
                    color = (0, 255, 0)  # Green for made
                    result_text = "PREDICTED: MADE"
                else:
                    color = (0, 0, 255)  # Red for missed
                    result_text = "PREDICTED: MISSED"
                
                conf_text = f"Confidence: {self.last_probability:.1%}"
                
                cv2.rectangle(annotated_frame, (10, 10), (700, 100), color, -1)
                cv2.putText(annotated_frame, result_text, (20, 50), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 3)
                cv2.putText(annotated_frame, conf_text, (20, 85), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            
            # Add shot counter overlay
            counter_text = f"Shots: {self.statistics['total_shots_detected']} | Made: {self.statistics['predicted_made']} | Missed: {self.statistics['predicted_missed']}"
            cv2.rectangle(annotated_frame, (10, frame_height - 40), (600, frame_height - 5), (0, 0, 0), -1)
            cv2.putText(annotated_frame, counter_text, (20, frame_height - 15),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # Save frame to video
            if video_writer is not None:
                video_writer.write(annotated_frame)
            
            # Display frame if enabled
            if display:
                # Resize for display
                height, width = annotated_frame.shape[:2]
                aspect_ratio = height / width
                
                display_width = self.display_width
                display_height = int(display_width * aspect_ratio)
                
                if display_height > self.display_max_height:
                    display_height = self.display_max_height
                    display_width = int(display_height / aspect_ratio)
                
                resized_frame = cv2.resize(annotated_frame, (display_width, display_height))
                cv2.imshow('Shot Prediction', resized_frame)
                
                # Keyboard controls
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    print("\n[USER QUIT]")
                    break
                elif key == ord('r'):
                    print("[MANUAL RESET]")
                    self.reset_shot_state()
        
        # Cleanup
        cap.release()
        if video_writer is not None:
            video_writer.release()
        if display:
            cv2.destroyAllWindows()
        
        # Record end time
        self.statistics['processing_end_time'] = datetime.now().isoformat()
        
        # Save statistics
        self.save_statistics(output_folder, video_path)
        
        # Print final summary
        print(f"\n{'=' * 60}")
        print("PROCESSING COMPLETE")
        print(f"{'=' * 60}")
        print(f"Total Shots Detected: {self.statistics['total_shots_detected']}")
        print(f"Predicted Made: {self.statistics['predicted_made']}")
        print(f"Predicted Missed: {self.statistics['predicted_missed']}")
        if self.statistics['total_shots_detected'] > 0:
            make_pct = (self.statistics['predicted_made'] / self.statistics['total_shots_detected']) * 100
            print(f"Predicted Make %: {make_pct:.1f}%")
        print(f"\nOutput folder: {output_folder}")
        if output_video_path:
            print(f"Processed video: {output_video_path}")
        print(f"{'=' * 60}")
        
        return output_folder, self.statistics


# =============================================================================
# MAIN EXECUTION
# =============================================================================
if __name__ == "__main__":
    
    # Paths - UPDATE THESE
    YOLO_MODEL = r"C:\Users\fridr\Documents\HooperAI\data\model\basketball-detection-colab-yolo11s-best.pt"
    LSTM_MODEL = r"C:\Users\fridr\Documents\HooperAI\data\model\basketball_shot_lstm.h5"
    SCALER_PATH = r"C:\Users\fridr\Documents\HooperAI\data\model\basketball_shot_lstm_scaler.pkl"
    CONFIG_PATH = r"C:\Users\fridr\Documents\HooperAI\data\model\basketball_shot_lstm_config.pkl"
    VIDEO_PATH = r"C:\Users\fridr\Documents\HooperAI\data\raw\videos\20250711_171647.mp4"
    OUTPUT_DIR = r"C:\Users\fridr\Documents\HooperAI\data\processed\predictions"
    
    # Processing options
    DISPLAY = False          # Set to False to run in background
    SAVE_VIDEO = True       # Set to False to skip saving video
    VERBOSE = True          # Set to False to reduce console output
    
    # Check if required files exist
    print("=" * 60)
    print("BASKETBALL SHOT PREDICTION - REAL-TIME INFERENCE")
    print("=" * 60)
    
    print("\n=== Checking Files ===")
    files_to_check = {
        'YOLO Model': YOLO_MODEL,
        'LSTM Model': LSTM_MODEL,
        'Scaler': SCALER_PATH,
        'Config': CONFIG_PATH,
        'Video': VIDEO_PATH
    }
    
    all_exist = True
    for name, path in files_to_check.items():
        exists = os.path.exists(path)
        status = "OK" if exists else "MISSING"
        print(f"  [{status}] {name}: {path}")
        if not exists:
            all_exist = False
    
    if not all_exist:
        print("\nERROR: Some required files are missing!")
        exit(1)
    
    # Debug: Verify scaler and config contents
    print("\n=== Verifying Saved Files ===")
    
    print("\n--- Scaler ---")
    with open(SCALER_PATH, 'rb') as f:
        test_scaler = pickle.load(f)
    print(f"  Type: {type(test_scaler).__name__}")
    print(f"  Is fitted: {hasattr(test_scaler, 'mean_')}")
    if hasattr(test_scaler, 'mean_'):
        print(f"  Number of features: {len(test_scaler.mean_)}")
    
    print("\n--- Config ---")
    with open(CONFIG_PATH, 'rb') as f:
        test_config = pickle.load(f)
    print(f"  Keys: {list(test_config.keys())}")
    print(f"  max_len: {test_config.get('max_len')}")
    print(f"  n_features: {test_config.get('n_features')}")
    
    print("\n" + "=" * 60)
    
    # Initialize predictor
    print("\n=== Initializing Predictor ===")
    predictor = RealTimeShotPredictor(
        yolo_model_path=YOLO_MODEL,
        lstm_model_path=LSTM_MODEL,
        scaler_path=SCALER_PATH,
        config_path=CONFIG_PATH,
        output_dir=OUTPUT_DIR
    )
    
    # Process video
    output_folder, statistics = predictor.process_video(
        video_path=VIDEO_PATH,
        display=DISPLAY,
        save_video=SAVE_VIDEO,
        verbose=VERBOSE
    )
    
    print(f"\nAll outputs saved to: {output_folder}")
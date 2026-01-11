import pandas as pd
import numpy as np
import json
import re
import os
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)
import matplotlib.pyplot as plt
import seaborn as sns


class ShotPredictionEvaluator:
    """Compare predicted shots against ground truth labels"""
    
    def __init__(self, ground_truth_csv, predictions_source, csv_delimiter=';'):
        """
        Initialize evaluator
        
        Args:
            ground_truth_csv: Path to CSV with labeled shot data
            predictions_source: Path to predictions (JSON, CSV, or TXT file) or string
            csv_delimiter: Delimiter used in ground truth CSV (default: ';')
        """
        self.ground_truth_csv = ground_truth_csv
        self.predictions_source = predictions_source
        self.csv_delimiter = csv_delimiter
        
        # Load data
        self.gt_shots = self._load_ground_truth()
        self.pred_shots = self._load_predictions()
        
        # Matched shots for evaluation
        self.matched_shots = None
        self.evaluation_results = None
        
    def _load_ground_truth(self):
        """Load and process ground truth CSV with semicolon delimiter"""
        print("Loading ground truth data...")
        print(f"  Path: {self.ground_truth_csv}")
        print(f"  Delimiter: '{self.csv_delimiter}'")
        
        # Read CSV with semicolon delimiter
        df = pd.read_csv(self.ground_truth_csv, sep=self.csv_delimiter)
        
        print(f"  Columns found: {list(df.columns)}")
        print(f"  Total rows: {len(df)}")
        
        # Group by shot number to get shot-level info
        shots = []
        for shot_num in df['shot_number'].unique():
            shot_data = df[df['shot_number'] == shot_num]
            
            # Get frame range
            start_frame = int(shot_data['frame_count'].min())
            end_frame = int(shot_data['frame_count'].max())
            
            # Get shot result (should be same for all frames in shot)
            shot_result = shot_data['shot_result'].iloc[0]
            
            # Handle NaN or missing shot_result
            if pd.isna(shot_result):
                # Try to get from any row that has a value
                non_null_results = shot_data['shot_result'].dropna()
                if len(non_null_results) > 0:
                    shot_result = int(non_null_results.iloc[0])
                else:
                    shot_result = 0  # Default to missed if unknown
            else:
                shot_result = int(shot_result)
            
            shot_info = {
                'shot_number': shot_num,
                'start_frame': start_frame,
                'end_frame': end_frame,
                'duration_frames': len(shot_data),
                'actual_result': shot_result,
                'actual_result_text': 'MADE' if shot_result == 1 else 'MISSED'
            }
            shots.append(shot_info)
        
        shots_df = pd.DataFrame(shots)
        shots_df = shots_df.sort_values('start_frame').reset_index(drop=True)
        
        print(f"\n  Loaded {len(shots_df)} ground truth shots")
        print(f"  Made: {(shots_df['actual_result'] == 1).sum()}")
        print(f"  Missed: {(shots_df['actual_result'] == 0).sum()}")
        print(f"  Frame range: {shots_df['start_frame'].min()} - {shots_df['end_frame'].max()}")
        
        # Print shot summary
        print(f"\n  Ground Truth Shots:")
        for _, shot in shots_df.iterrows():
            print(f"    Shot {shot['shot_number']}: {shot['actual_result_text']:6} (frames {shot['start_frame']}-{shot['end_frame']})")
        
        return shots_df
    
    def _load_predictions(self):
        """Load predictions from various formats"""
        print("\nLoading predictions...")
        
        # Check if it's a string (inline predictions)
        if isinstance(self.predictions_source, str) and not os.path.exists(self.predictions_source):
            # Might be inline predictions string
            if 'Shot' in self.predictions_source and 'frames:' in self.predictions_source:
                print("  Parsing inline predictions string...")
                return self._parse_predictions_string(self.predictions_source)
        
        print(f"  Path: {self.predictions_source}")
        
        ext = os.path.splitext(self.predictions_source)[1].lower()
        
        if ext == '.json':
            return self._load_predictions_json()
        elif ext == '.csv':
            return self._load_predictions_csv()
        elif ext == '.txt':
            return self._load_predictions_txt()
        else:
            raise ValueError(f"Unsupported predictions format: {ext}")
    
    def _load_predictions_json(self):
        """Load predictions from JSON file"""
        with open(self.predictions_source, 'r') as f:
            data = json.load(f)
        
        shots = []
        for shot in data.get('shots', []):
            shot_info = {
                'pred_shot_number': shot['shot_number'],
                'start_frame': shot['start_frame'],
                'end_frame': shot['end_frame'],
                'predicted_result': shot['prediction'],
                'predicted_result_text': shot['prediction_text'],
                'confidence': shot['confidence']
            }
            shots.append(shot_info)
        
        shots_df = pd.DataFrame(shots)
        print(f"  Loaded {len(shots_df)} predicted shots from JSON")
        return shots_df
    
    def _load_predictions_csv(self):
        """Load predictions from CSV file"""
        df = pd.read_csv(self.predictions_source)
        
        # Rename columns to standard format if needed
        column_mapping = {
            'shot_number': 'pred_shot_number',
            'prediction': 'predicted_result',
            'prediction_text': 'predicted_result_text',
        }
        
        for old_col, new_col in column_mapping.items():
            if old_col in df.columns and new_col not in df.columns:
                df = df.rename(columns={old_col: new_col})
        
        print(f"  Loaded {len(df)} predicted shots from CSV")
        return df
    
    def _load_predictions_txt(self):
        """Load predictions from summary TXT file"""
        with open(self.predictions_source, 'r') as f:
            content = f.read()
        
        return self._parse_predictions_string(content)
    
    def _parse_predictions_string(self, predictions_string):
        """Parse predictions from a string"""
        pattern = r'Shot (\d+): (MADE|MISSED) $confidence: ([\d.]+)%, frames: (\d+)-(\d+)$'
        matches = re.findall(pattern, predictions_string)
        
        shots = []
        for match in matches:
            shot_num, result, confidence, start_frame, end_frame = match
            shot_info = {
                'pred_shot_number': int(shot_num),
                'start_frame': int(start_frame),
                'end_frame': int(end_frame),
                'predicted_result': 1 if result == 'MADE' else 0,
                'predicted_result_text': result,
                'confidence': float(confidence) / 100
            }
            shots.append(shot_info)
        
        shots_df = pd.DataFrame(shots)
        print(f"  Parsed {len(shots_df)} predicted shots")
        
        # Print predicted shots summary
        print(f"\n  Predicted Shots:")
        for _, shot in shots_df.iterrows():
            print(f"    Pred {shot['pred_shot_number']}: {shot['predicted_result_text']:6} "
                  f"(frames {shot['start_frame']}-{shot['end_frame']}, conf: {shot['confidence']*100:.1f}%)")
        
        return shots_df
    
    def match_shots(self, iou_threshold=0.3, frame_tolerance=50):
        """
        Match predicted shots to ground truth shots based on frame overlap
        
        Args:
            iou_threshold: Minimum IoU for frame ranges to match
            frame_tolerance: Maximum frame distance between centers to consider a match
        """
        print(f"\n{'='*60}")
        print("MATCHING PREDICTED SHOTS TO GROUND TRUTH")
        print(f"{'='*60}")
        print(f"IoU threshold: {iou_threshold}")
        print(f"Frame tolerance: {frame_tolerance}")
        
        matched = []
        used_pred_indices = set()
        
        # For each ground truth shot, find best matching prediction
        for gt_idx, gt_shot in self.gt_shots.iterrows():
            gt_start = gt_shot['start_frame']
            gt_end = gt_shot['end_frame']
            gt_center = (gt_start + gt_end) / 2
            
            best_match = None
            best_score = -1
            best_pred_idx = None
            
            for pred_idx, pred_shot in self.pred_shots.iterrows():
                if pred_idx in used_pred_indices:
                    continue
                    
                pred_start = pred_shot['start_frame']
                pred_end = pred_shot['end_frame']
                pred_center = (pred_start + pred_end) / 2
                
                # Calculate frame overlap (IoU)
                intersection_start = max(gt_start, pred_start)
                intersection_end = min(gt_end, pred_end)
                intersection = max(0, intersection_end - intersection_start)
                
                union_start = min(gt_start, pred_start)
                union_end = max(gt_end, pred_end)
                union = union_end - union_start
                
                iou = intersection / union if union > 0 else 0
                
                # Calculate center distance
                center_dist = abs(gt_center - pred_center)
                
                # Score: prioritize IoU, but also consider proximity
                if iou >= iou_threshold:
                    score = iou
                elif center_dist < frame_tolerance:
                    score = 0.1 + (1 - center_dist / frame_tolerance) * 0.3
                else:
                    score = 0
                
                if score > best_score:
                    best_score = score
                    best_match = pred_shot
                    best_pred_idx = pred_idx
                    best_iou = iou
            
            if best_match is not None and best_score > 0:
                match_info = {
                    'gt_shot_number': gt_shot['shot_number'],
                    'gt_start_frame': gt_start,
                    'gt_end_frame': gt_end,
                    'actual_result': gt_shot['actual_result'],
                    'actual_result_text': gt_shot['actual_result_text'],
                    'pred_shot_number': best_match['pred_shot_number'],
                    'pred_start_frame': best_match['start_frame'],
                    'pred_end_frame': best_match['end_frame'],
                    'predicted_result': best_match['predicted_result'],
                    'predicted_result_text': best_match['predicted_result_text'],
                    'confidence': best_match['confidence'],
                    'iou': best_iou,
                    'matched': True,
                    'correct': gt_shot['actual_result'] == best_match['predicted_result']
                }
                matched.append(match_info)
                used_pred_indices.add(best_pred_idx)
            else:
                # Ground truth shot not detected
                match_info = {
                    'gt_shot_number': gt_shot['shot_number'],
                    'gt_start_frame': gt_start,
                    'gt_end_frame': gt_end,
                    'actual_result': gt_shot['actual_result'],
                    'actual_result_text': gt_shot['actual_result_text'],
                    'pred_shot_number': None,
                    'pred_start_frame': None,
                    'pred_end_frame': None,
                    'predicted_result': None,
                    'predicted_result_text': 'NOT DETECTED',
                    'confidence': None,
                    'iou': 0,
                    'matched': False,
                    'correct': False
                }
                matched.append(match_info)
        
        # Add unmatched predictions (false detections)
        for pred_idx, pred_shot in self.pred_shots.iterrows():
            if pred_idx not in used_pred_indices:
                match_info = {
                    'gt_shot_number': None,
                    'gt_start_frame': None,
                    'gt_end_frame': None,
                    'actual_result': None,
                    'actual_result_text': 'FALSE DETECTION',
                    'pred_shot_number': pred_shot['pred_shot_number'],
                    'pred_start_frame': pred_shot['start_frame'],
                    'pred_end_frame': pred_shot['end_frame'],
                    'predicted_result': pred_shot['predicted_result'],
                    'predicted_result_text': pred_shot['predicted_result_text'],
                    'confidence': pred_shot['confidence'],
                    'iou': 0,
                    'matched': False,
                    'correct': False
                }
                matched.append(match_info)
        
        self.matched_shots = pd.DataFrame(matched)
        
        # Print matching summary
        n_matched = self.matched_shots['matched'].sum()
        n_gt = len(self.gt_shots)
        n_pred = len(self.pred_shots)
        n_false = len([m for m in matched if m['actual_result_text'] == 'FALSE DETECTION'])
        n_missed = len([m for m in matched if m['predicted_result_text'] == 'NOT DETECTED'])
        
        print(f"\n--- Matching Results ---")
        print(f"Ground truth shots:    {n_gt}")
        print(f"Predicted shots:       {n_pred}")
        print(f"Successfully matched:  {n_matched}")
        print(f"Missed detections:     {n_missed} (GT shots not found by predictor)")
        print(f"False detections:      {n_false} (Predictions without GT)")
        
        return self.matched_shots
    
    def evaluate(self):
        """Calculate evaluation metrics"""
        if self.matched_shots is None:
            self.match_shots()
        
        print(f"\n{'='*60}")
        print("EVALUATION METRICS")
        print(f"{'='*60}")
        
        results = {}
        
        # ===========================================
        # 1. SHOT DETECTION METRICS
        # ===========================================
        print("\n" + "-"*50)
        print("1. SHOT DETECTION PERFORMANCE")
        print("-"*50)
        
        n_gt = len(self.gt_shots)
        n_pred = len(self.pred_shots)
        n_matched = self.matched_shots['matched'].sum()
        n_false = len(self.matched_shots[self.matched_shots['actual_result_text'] == 'FALSE DETECTION'])
        n_missed = len(self.matched_shots[self.matched_shots['predicted_result_text'] == 'NOT DETECTED'])
        
        det_recall = n_matched / n_gt if n_gt > 0 else 0
        det_precision = n_matched / n_pred if n_pred > 0 else 0
        det_f1 = 2 * (det_precision * det_recall) / (det_precision + det_recall) \
                 if (det_precision + det_recall) > 0 else 0
        
        results['detection'] = {
            'total_ground_truth': int(n_gt),
            'total_predicted': int(n_pred),
            'correctly_detected': int(n_matched),
            'missed_detections': int(n_missed),
            'false_detections': int(n_false),
            'detection_recall': float(det_recall),
            'detection_precision': float(det_precision),
            'detection_f1': float(det_f1)
        }
        
        print(f"Ground truth shots:    {n_gt}")
        print(f"Predicted shots:       {n_pred}")
        print(f"Correctly detected:    {n_matched} ({det_recall*100:.1f}% recall)")
        print(f"Missed detections:     {n_missed}")
        print(f"False detections:      {n_false}")
        print(f"Detection Recall:      {det_recall*100:.1f}%")
        print(f"Detection Precision:   {det_precision*100:.1f}%")
        print(f"Detection F1:          {det_f1*100:.1f}%")
        
        # ===========================================
        # 2. SHOT OUTCOME PREDICTION METRICS
        # ===========================================
        print("\n" + "-"*50)
        print("2. SHOT OUTCOME PREDICTION (Matched Shots Only)")
        print("-"*50)
        
        matched_only = self.matched_shots[self.matched_shots['matched'] == True].copy()
        
        if len(matched_only) > 0:
            y_true = matched_only['actual_result'].values.astype(int)
            y_pred = matched_only['predicted_result'].values.astype(int)
            
            accuracy = accuracy_score(y_true, y_pred)
            
            # Handle case where all predictions or all actuals are same class
            try:
                precision = precision_score(y_true, y_pred, zero_division=0)
                recall = recall_score(y_true, y_pred, zero_division=0)
                f1 = f1_score(y_true, y_pred, zero_division=0)
            except:
                precision = recall = f1 = 0.0
            
            # Confusion matrix
            cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
            if cm.shape == (2, 2):
                tn, fp, fn, tp = cm.ravel()
            else:
                # Handle edge cases
                tn = fp = fn = tp = 0
                if cm.shape[0] == 1:
                    if y_true[0] == 0:
                        tn = cm[0, 0]
                    else:
                        tp = cm[0, 0]
            
            results['outcome_prediction'] = {
                'n_matched_shots': int(len(matched_only)),
                'accuracy': float(accuracy),
                'precision': float(precision),
                'recall': float(recall),
                'f1_score': float(f1),
                'true_positives': int(tp),
                'true_negatives': int(tn),
                'false_positives': int(fp),
                'false_negatives': int(fn),
                'confusion_matrix': cm.tolist()
            }
            
            print(f"Matched shots evaluated:   {len(matched_only)}")
            print(f"")
            print(f"Accuracy:                  {accuracy*100:.1f}%")
            print(f"Precision (MADE):          {precision*100:.1f}%")
            print(f"Recall (MADE):             {recall*100:.1f}%")
            print(f"F1 Score:                  {f1*100:.1f}%")
            print(f"")
            print(f"Confusion Matrix:")
            print(f"                    Predicted")
            print(f"                 MISSED    MADE")
            print(f"Actual MISSED      {tn:3}      {fp:3}")
            print(f"Actual MADE        {fn:3}      {tp:3}")
            print(f"")
            print(f"True Positives (Correct MADE):     {tp}")
            print(f"True Negatives (Correct MISSED):   {tn}")
            print(f"False Positives (Wrong MADE):      {fp}")
            print(f"False Negatives (Wrong MISSED):    {fn}")
        else:
            print("No matched shots to evaluate!")
            results['outcome_prediction'] = None
        
        # ===========================================
        # 3. CONFIDENCE ANALYSIS
        # ===========================================
        print("\n" + "-"*50)
        print("3. CONFIDENCE ANALYSIS")
        print("-"*50)
        
        if len(matched_only) > 0:
            correct = matched_only[matched_only['correct'] == True]
            incorrect = matched_only[matched_only['correct'] == False]
            
            avg_conf_correct = correct['confidence'].mean() if len(correct) > 0 else 0
            avg_conf_incorrect = incorrect['confidence'].mean() if len(incorrect) > 0 else 0
            avg_conf_all = matched_only['confidence'].mean()
            
            made_preds = matched_only[matched_only['predicted_result'] == 1]
            missed_preds = matched_only[matched_only['predicted_result'] == 0]
            
            results['confidence'] = {
                'avg_confidence_all': float(avg_conf_all),
                'avg_confidence_correct': float(avg_conf_correct),
                'avg_confidence_incorrect': float(avg_conf_incorrect),
                'avg_confidence_made_predictions': float(made_preds['confidence'].mean()) if len(made_preds) > 0 else 0,
                'avg_confidence_missed_predictions': float(missed_preds['confidence'].mean()) if len(missed_preds) > 0 else 0
            }
            
            print(f"Average confidence (all):       {avg_conf_all*100:.1f}%")
            print(f"Average confidence (correct):   {avg_conf_correct*100:.1f}%")
            print(f"Average confidence (incorrect): {avg_conf_incorrect*100:.1f}%")
            print(f"Avg conf for MADE predictions:  {results['confidence']['avg_confidence_made_predictions']*100:.1f}%")
            print(f"Avg conf for MISSED predictions:{results['confidence']['avg_confidence_missed_predictions']*100:.1f}%")
        else:
            results['confidence'] = None
        
        # ===========================================
        # 4. SHOT-BY-SHOT RESULTS
        # ===========================================
        print("\n" + "-"*50)
        print("4. SHOT-BY-SHOT COMPARISON")
        print("-"*50)
        
        for _, row in self.matched_shots.iterrows():
            if row['matched']:
                status = "✅" if row['correct'] else "❌"
                print(f"{status} GT Shot {row['gt_shot_number']:2}: {row['actual_result_text']:6} "
                      f"(frames {row['gt_start_frame']}-{row['gt_end_frame']}) | "
                      f"Pred: {row['predicted_result_text']:6} (conf: {row['confidence']*100:5.1f}%, "
                      f"frames {row['pred_start_frame']}-{row['pred_end_frame']}) | IoU: {row['iou']:.2f}")
            elif row['predicted_result_text'] == 'NOT DETECTED':
                print(f"⚠️  GT Shot {row['gt_shot_number']:2}: {row['actual_result_text']:6} "
                      f"(frames {row['gt_start_frame']}-{row['gt_end_frame']}) | NOT DETECTED")
            else:
                print(f"🔴 FALSE DETECTION: Pred Shot {row['pred_shot_number']:2} | "
                      f"{row['predicted_result_text']:6} (conf: {row['confidence']*100:5.1f}%, "
                      f"frames {row['pred_start_frame']}-{row['pred_end_frame']})")
        
        self.evaluation_results = results
        return results
    
    def plot_results(self, save_path=None):
        """Generate visualization plots"""
        if self.evaluation_results is None:
            self.evaluate()
        
        fig, axes = plt.subplots(2, 3, figsize=(16, 10))
        fig.suptitle('Shot Prediction Evaluation Results', fontsize=14, fontweight='bold')
        
        # 1. Confusion Matrix
        ax1 = axes[0, 0]
        if self.evaluation_results['outcome_prediction']:
            cm = np.array(self.evaluation_results['outcome_prediction']['confusion_matrix'])
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax1,
                       xticklabels=['MISSED', 'MADE'], yticklabels=['MISSED', 'MADE'],
                       annot_kws={'size': 16})
            ax1.set_xlabel('Predicted', fontsize=12)
            ax1.set_ylabel('Actual', fontsize=12)
            ax1.set_title('Confusion Matrix', fontsize=12)
        else:
            ax1.text(0.5, 0.5, 'No matched shots', ha='center', va='center', fontsize=12)
            ax1.set_title('Confusion Matrix')
        
        # 2. Detection Performance
        ax2 = axes[0, 1]
        detection = self.evaluation_results['detection']
        categories = ['Correctly\nDetected', 'Missed\nDetections', 'False\nDetections']
        values = [detection['correctly_detected'], 
                 detection['missed_detections'],
                 detection['false_detections']]
        colors = ['#2ecc71', '#e74c3c', '#f39c12']
        bars = ax2.bar(categories, values, color=colors, edgecolor='black')
        ax2.set_title('Shot Detection Results', fontsize=12)
        ax2.set_ylabel('Count', fontsize=11)
        for bar, val in zip(bars, values):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                    str(val), ha='center', va='bottom', fontweight='bold', fontsize=12)
        
        # 3. Metrics Comparison
        ax3 = axes[0, 2]
        if self.evaluation_results['outcome_prediction']:
            metrics = ['Accuracy', 'Precision', 'Recall', 'F1 Score']
            outcome = self.evaluation_results['outcome_prediction']
            values = [outcome['accuracy'], outcome['precision'], 
                     outcome['recall'], outcome['f1_score']]
            colors = ['#3498db', '#e67e22', '#27ae60', '#9b59b6']
            bars = ax3.bar(metrics, [v*100 for v in values], color=colors, edgecolor='black')
            ax3.set_title('Prediction Metrics', fontsize=12)
            ax3.set_ylabel('Percentage (%)', fontsize=11)
            ax3.set_ylim([0, 105])
            for bar, val in zip(bars, values):
                ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                        f'{val*100:.1f}%', ha='center', va='bottom', fontsize=10)
        else:
            ax3.text(0.5, 0.5, 'No matched shots', ha='center', va='center', fontsize=12)
            ax3.set_title('Prediction Metrics')
        
        # 4. Confidence Distribution
        ax4 = axes[1, 0]
        matched_only = self.matched_shots[self.matched_shots['matched'] == True]
        if len(matched_only) > 0:
            correct = matched_only[matched_only['correct'] == True]['confidence'] * 100
            incorrect = matched_only[matched_only['correct'] == False]['confidence'] * 100
            
            if len(correct) > 0:
                ax4.hist(correct, bins=10, alpha=0.7, label=f'Correct ({len(correct)})', 
                        color='#2ecc71', edgecolor='black')
            if len(incorrect) > 0:
                ax4.hist(incorrect, bins=10, alpha=0.7, label=f'Incorrect ({len(incorrect)})', 
                        color='#e74c3c', edgecolor='black')
            ax4.set_xlabel('Confidence (%)', fontsize=11)
            ax4.set_ylabel('Count', fontsize=11)
            ax4.set_title('Confidence Distribution', fontsize=12)
            ax4.legend(fontsize=10)
        else:
            ax4.text(0.5, 0.5, 'No matched shots', ha='center', va='center', fontsize=12)
            ax4.set_title('Confidence Distribution')
        
        # 5. Shot Timeline Visualization
        ax5 = axes[1, 1]
        
        # Plot ground truth shots
        gt_y_positions = {}
        for i, (_, row) in enumerate(self.gt_shots.iterrows()):
            y_pos = i * 2
            gt_y_positions[row['shot_number']] = y_pos
            color = '#2ecc71' if row['actual_result'] == 1 else '#e74c3c'
            ax5.barh(y_pos, row['end_frame'] - row['start_frame'],
                    left=row['start_frame'], height=0.8, color=color, alpha=0.7,
                    label='GT' if i == 0 else '', edgecolor='black')
            ax5.text(row['start_frame'], y_pos, f" GT{row['shot_number']}", 
                    va='center', ha='left', fontsize=8)
        
        # Plot predicted shots
        for i, (_, row) in enumerate(self.pred_shots.iterrows()):
            y_pos = i * 2 + 0.9
            ax5.barh(y_pos, row['end_frame'] - row['start_frame'],
                    left=row['start_frame'], height=0.6, color='#3498db', alpha=0.5,
                    label='Predicted' if i == 0 else '', edgecolor='black', linestyle='--')
        
        ax5.set_xlabel('Frame', fontsize=11)
        ax5.set_ylabel('Shot Index', fontsize=11)
        ax5.set_title('Shot Timeline (Green=Made, Red=Missed, Blue=Predicted)', fontsize=10)
        ax5.legend(loc='upper right', fontsize=9)
        
        # 6. Summary Statistics
        ax6 = axes[1, 2]
        ax6.axis('off')
        
        detection = self.evaluation_results['detection']
        outcome = self.evaluation_results['outcome_prediction']
        confidence = self.evaluation_results.get('confidence', {})
        
        summary_text = f"""
══════════════════════════════════════
       EVALUATION SUMMARY
══════════════════════════════════════

SHOT DETECTION
  Ground Truth:        {detection['total_ground_truth']}
  Predicted:           {detection['total_predicted']}
  Correctly Detected:  {detection['correctly_detected']}
  Detection Recall:    {detection['detection_recall']*100:.1f}%
  Detection Precision: {detection['detection_precision']*100:.1f}%
  Detection F1:        {detection['detection_f1']*100:.1f}%
"""
        
        if outcome:
            summary_text += f"""
OUTCOME PREDICTION (n={outcome['n_matched_shots']})
  Accuracy:            {outcome['accuracy']*100:.1f}%
  Precision:           {outcome['precision']*100:.1f}%
  Recall:              {outcome['recall']*100:.1f}%
  F1 Score:            {outcome['f1_score']*100:.1f}%

  Correct MADE:        {outcome['true_positives']}
  Correct MISSED:      {outcome['true_negatives']}
  Wrong (FP):          {outcome['false_positives']}
  Wrong (FN):          {outcome['false_negatives']}
"""
        
        if confidence:
            summary_text += f"""
CONFIDENCE
  Avg (correct):       {confidence['avg_confidence_correct']*100:.1f}%
  Avg (incorrect):     {confidence['avg_confidence_incorrect']*100:.1f}%
"""
        
        ax6.text(0.05, 0.95, summary_text, transform=ax6.transAxes, fontsize=10,
                verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='#ecf0f1', alpha=0.8))
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"\nPlot saved to: {save_path}")
        
        plt.show()
        return fig
    
    def save_results(self, output_dir):
        """Save evaluation results to files"""
        os.makedirs(output_dir, exist_ok=True)
        
        # Save matched shots CSV
        csv_path = os.path.join(output_dir, 'matched_shots.csv')
        self.matched_shots.to_csv(csv_path, index=False)
        print(f"Saved matched shots to: {csv_path}")
        
        # Save evaluation results JSON
        json_path = os.path.join(output_dir, 'evaluation_results.json')
        with open(json_path, 'w') as f:
            json.dump(self.evaluation_results, f, indent=4)
        print(f"Saved evaluation results to: {json_path}")
        
        # Save summary report
        report_path = os.path.join(output_dir, 'evaluation_report.txt')
        self._save_text_report(report_path)
        print(f"Saved evaluation report to: {report_path}")
        
        # Save plot
        plot_path = os.path.join(output_dir, 'evaluation_plots.png')
        self.plot_results(save_path=plot_path)
        
        return output_dir
    
    def _save_text_report(self, report_path):
        """Save detailed text report"""
        with open(report_path, 'w') as f:
            f.write("=" * 60 + "\n")
            f.write("SHOT PREDICTION EVALUATION REPORT\n")
            f.write("=" * 60 + "\n\n")
            
            d = self.evaluation_results['detection']
            f.write("SHOT DETECTION PERFORMANCE\n")
            f.write("-" * 40 + "\n")
            f.write(f"Ground Truth Shots:  {d['total_ground_truth']}\n")
            f.write(f"Predicted Shots:     {d['total_predicted']}\n")
            f.write(f"Correctly Detected:  {d['correctly_detected']}\n")
            f.write(f"Missed Detections:   {d['missed_detections']}\n")
            f.write(f"False Detections:    {d['false_detections']}\n")
            f.write(f"Detection Recall:    {d['detection_recall']*100:.1f}%\n")
            f.write(f"Detection Precision: {d['detection_precision']*100:.1f}%\n")
            f.write(f"Detection F1:        {d['detection_f1']*100:.1f}%\n\n")
            
            if self.evaluation_results['outcome_prediction']:
                o = self.evaluation_results['outcome_prediction']
                f.write("OUTCOME PREDICTION PERFORMANCE\n")
                f.write("-" * 40 + "\n")
                f.write(f"Matched Shots:       {o['n_matched_shots']}\n")
                f.write(f"Accuracy:            {o['accuracy']*100:.1f}%\n")
                f.write(f"Precision:           {o['precision']*100:.1f}%\n")
                f.write(f"Recall:              {o['recall']*100:.1f}%\n")
                f.write(f"F1 Score:            {o['f1_score']*100:.1f}%\n\n")
                f.write(f"True Positives:      {o['true_positives']}\n")
                f.write(f"True Negatives:      {o['true_negatives']}\n")
                f.write(f"False Positives:     {o['false_positives']}\n")
                f.write(f"False Negatives:     {o['false_negatives']}\n\n")
            
            f.write("SHOT-BY-SHOT RESULTS\n")
            f.write("-" * 40 + "\n")
            for _, row in self.matched_shots.iterrows():
                if row['matched']:
                    status = "CORRECT" if row['correct'] else "WRONG"
                    f.write(f"GT Shot {row['gt_shot_number']}: {row['actual_result_text']} -> "
                           f"Pred: {row['predicted_result_text']} ({status}) "
                           f"[conf: {row['confidence']*100:.1f}%]\n")
                elif row['predicted_result_text'] == 'NOT DETECTED':
                    f.write(f"GT Shot {row['gt_shot_number']}: {row['actual_result_text']} -> "
                           f"NOT DETECTED\n")
                else:
                    f.write(f"FALSE DETECTION: Pred Shot {row['pred_shot_number']} "
                           f"[frames: {row['pred_start_frame']}-{row['pred_end_frame']}]\n")


# =============================================================================
# MAIN EXECUTION
# =============================================================================
if __name__ == "__main__":
    
    # =========================================
    # UPDATE THESE PATHS
    # =========================================
    
# Paths to your data
    GROUND_TRUTH_CSV = r"C:\Users\fridr\Documents\HooperAI\data\processed\annotations\basketball_shot_data_labeled.csv"
    
    # Option 1: Load from statistics.json
    # PREDICTIONS = r"C:\Users\fridr\Documents\HooperAI\data\processed\predictions\...\statistics.json"
    
    # Option 2: Load from shots.csv
    PREDICTIONS = r"C:\Users\fridr\Documents\HooperAI\data\processed\predictions\20260111_201443_20250711_171647\shots.csv"
    
    # Option 3: Load from summary.txt
    # PREDICTIONS = r"C:\Users\fridr\Documents\HooperAI\data\pocessed\predictions\...\summary.txt"
    
    # Output directory for results
    OUTPUT_DIR = r"C:\Users\fridr\Documents\HooperAI\data\processed\evaluation\metrics"
    
    # Output directory
    OUTPUT_DIR = r"C:\Users\fridr\Documents\HooperAI\data\processed\evaluation"
    
    # CSV delimiter (use ';' for your format)
    CSV_DELIMITER = ';'
    
    # =========================================
    # RUN EVALUATION
    # =========================================
    
    print("=" * 60)
    print("SHOT PREDICTION EVALUATION")
    print("=" * 60)
    
    # Check if ground truth file exists
    if not os.path.exists(GROUND_TRUTH_CSV):
        print(f"\n❌ Ground truth file not found: {GROUND_TRUTH_CSV}")
        exit(1)
    print(f"\n✅ Ground truth file found")
    
    # Initialize evaluator
    evaluator = ShotPredictionEvaluator(
        ground_truth_csv=GROUND_TRUTH_CSV,
        predictions_source=PREDICTIONS,
        csv_delimiter=CSV_DELIMITER
    )
    
    # Match shots
    evaluator.match_shots(iou_threshold=0.3, frame_tolerance=50)
    
    # Evaluate
    results = evaluator.evaluate()
    
    # Plot results
    evaluator.plot_results()
    
    # Save results
    evaluator.save_results(OUTPUT_DIR)
    
    print("\n" + "=" * 60)
    print("EVALUATION COMPLETE")
    print(f"Results saved to: {OUTPUT_DIR}")
    print("=" * 60)

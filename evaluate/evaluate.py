#!/usr/bin/env python3
"""
WhatsApp NLP Project - Evaluation Module

This script evaluates the existing sentiment and sarcasm models against
a ground-truth labeled dataset without modifying any existing project files.

IMPORTANT: This evaluation uses the EXACT same models and processing pipeline
as the main application to ensure identical behavior.

Usage: python evaluate/evaluate.py

Author: WhatsApp Chat Analyzer Evaluation Module
"""

import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix
)
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Add the parent directory to Python path to import existing modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import EXISTING NLP modules - NO modifications to existing code
import nlp_analyzer
import sarcasm_analyzer

class WhatsAppNLPEvaluator:
    """
    Evaluates existing WhatsApp NLP models on labeled ground-truth data.
    
    Uses the EXACT same models and processing pipeline as the main application:
    - Sentiment: nlp_analyzer.load_sentiment_model() + nlp_analyzer.run_sentiment_analysis()
    - Sarcasm: sarcasm_analyzer.load_sarcasm_model() + sarcasm_analyzer.run_sarcasm_analysis()
    """
    
    def __init__(self, dataset_path="evaluate/dataset/whatsapp_hinglish_messages.csv"):
        self.dataset_path = dataset_path
        self.results_dir = "evaluate/results"
        self.df = None
        self.predictions = None
        self.log_lines = []  # For detailed logging
        
        print("WhatsApp NLP Evaluator initialized")
        print(f"Dataset: {dataset_path}")
        print(f"Results: {self.results_dir}")
    
    def log(self, message):
        """Add message to detailed log."""
        self.log_lines.append(message)
    
    def save_detailed_log(self):
        """Save all detailed logs to file."""
        if self.log_lines:
            with open(f"{self.results_dir}/reports/detailed_evaluation_log.txt", "w", encoding='utf-8') as f:
                f.write("\n".join(self.log_lines))
            print(f"✅ Detailed log saved to: reports/detailed_evaluation_log.txt")
    
    def load_and_validate_dataset(self):
        """Load dataset and perform comprehensive validation."""
        print("\n" + "="*60)
        print("STEP 1: DATASET LOADING AND VALIDATION")
        print("="*60)
        
        # Check if dataset file exists
        if not os.path.exists(self.dataset_path):
            raise FileNotFoundError(
                f"Dataset not found: {self.dataset_path}\n"
                "Please place your dataset file at: evaluate/dataset/whatsapp_hinglish_messages.csv"
            )
        
        # Load dataset
        print(f"Loading dataset: {self.dataset_path}")
        self.df = pd.read_csv(self.dataset_path)
        print(f"Dataset loaded successfully: {len(self.df)} rows")
        
        # Validate required columns
        required_cols = ['message', 'sentiment', 'is_sarcastic']
        missing_cols = [col for col in required_cols if col not in self.df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
        
        # Dataset Statistics
        print(f"\nDataset Statistics:")
        print(f"- Total rows: {len(self.df):,}")
        
        # Check missing values
        missing_messages = self.df['message'].isna().sum()
        missing_sentiment = self.df['sentiment'].isna().sum() 
        missing_sarcasm = self.df['is_sarcastic'].isna().sum()
        
        print(f"- Missing messages: {missing_messages}")
        print(f"- Missing sentiment labels: {missing_sentiment}")
        print(f"- Missing sarcasm labels: {missing_sarcasm}")
        
        if missing_messages > 0 or missing_sentiment > 0 or missing_sarcasm > 0:
            print("WARNING: Missing values detected in dataset!")
        
        # Check duplicates
        duplicate_messages = self.df.duplicated(subset=['message']).sum()
        print(f"- Duplicate messages: {duplicate_messages}")
        
        # Validate sentiment values
        valid_sentiments = {'Positive', 'Neutral', 'Negative'}
        unique_sentiments = set(self.df['sentiment'].dropna().unique())
        invalid_sentiments = unique_sentiments - valid_sentiments
        
        if invalid_sentiments:
            print(f"ERROR: Invalid sentiment values found: {invalid_sentiments}")
            print(f"Valid sentiment values are: {valid_sentiments}")
            raise ValueError("Invalid sentiment labels in dataset")
        
        print(f"- Valid sentiment labels: {unique_sentiments}")
        
        # Validate sarcasm values  
        unique_sarcasm = set(self.df['is_sarcastic'].dropna().unique())
        valid_sarcasm = {True, False, 'TRUE', 'FALSE'}
        
        # Handle different boolean representations
        if not unique_sarcasm.issubset(valid_sarcasm):
            # Try to convert string representations
            if unique_sarcasm.issubset({'True', 'False', 'true', 'false', 'TRUE', 'FALSE'}):
                print("Converting string boolean values to actual booleans...")
                self.df['is_sarcastic'] = self.df['is_sarcastic'].map({
                    'True': True, 'true': True, True: True, 'TRUE': True,
                    'False': False, 'false': False, False: False, 'FALSE': False
                })
                unique_sarcasm = set(self.df['is_sarcastic'].dropna().unique())
            
            if not unique_sarcasm.issubset({True, False}):
                print(f"ERROR: Invalid sarcasm values found: {unique_sarcasm}")
                print(f"Valid sarcasm values are: {True, False}")
                raise ValueError("Invalid sarcasm labels in dataset")
        
        print(f"- Valid sarcasm labels: {unique_sarcasm}")
        
        # Distribution analysis
        print(f"\nSentiment Distribution:")
        sentiment_counts = self.df['sentiment'].value_counts()
        for sentiment, count in sentiment_counts.items():
            pct = count / len(self.df) * 100
            print(f"  {sentiment}: {count:,} ({pct:.1f}%)")
        
        print(f"\nSarcasm Distribution:")
        sarcasm_counts = self.df['is_sarcastic'].value_counts()
        for is_sarc, count in sarcasm_counts.items():
            label = "Sarcastic" if is_sarc else "Not Sarcastic"
            pct = count / len(self.df) * 100
            print(f"  {label}: {count:,} ({pct:.1f}%)")
        
        # Save dataset statistics to file
        stats = {
            'total_rows': len(self.df),
            'missing_messages': missing_messages,
            'missing_sentiment': missing_sentiment,
            'missing_sarcasm': missing_sarcasm,
            'duplicate_messages': duplicate_messages,
            'sentiment_distribution': sentiment_counts.to_dict(),
            'sarcasm_distribution': sarcasm_counts.to_dict()
        }
        
        # Save to results folder
        import json
        with open(f"{self.results_dir}/reports/dataset_statistics.json", "w") as f:
            json.dump(stats, f, indent=2, default=str)
        print(f"\nDataset statistics saved to: {self.results_dir}/reports/dataset_statistics.json")
        
        print("\nDataset validation completed successfully!")
        return True
    
    def run_existing_model_predictions(self):
        """
        Run predictions using the EXACT same models and pipeline as the main application.
        This ensures identical behavior to what users see in the app.
        """
        print("\n" + "="*60)
        print("STEP 2: RUNNING EXISTING MODEL PREDICTIONS")
        print("="*60)
        
        # Create a DataFrame that mimics the structure expected by existing pipeline
        # The existing models expect DataFrame with 'message', 'user', 'date' columns
        print("Preparing data for existing NLP pipeline...")
        
        eval_df = pd.DataFrame({
            'message': self.df['message'].copy(),
            'user': ['eval_user'] * len(self.df),  # Dummy user for processing
            'date': pd.Timestamp.now()  # Dummy timestamp
        })
        
        print(f"Processing {len(eval_df)} messages through existing models...")
        
        # Load EXISTING sentiment model (same function as main app)
        print("\nLoading sentiment model...")
        sentiment_pipe = nlp_analyzer.load_sentiment_model()
        if sentiment_pipe is None:
            raise RuntimeError("Failed to load existing sentiment model")
        print("Sentiment model loaded successfully")
        
        # Load EXISTING sarcasm model (same function as main app)  
        print("\nLoading sarcasm model...")
        sarcasm_pipe = sarcasm_analyzer.load_sarcasm_model()
        if sarcasm_pipe is None:
            raise RuntimeError("Failed to load existing sarcasm model")
        print("Sarcasm model loaded successfully")
        
        # Run EXISTING sentiment analysis pipeline
        print("\nRunning sentiment analysis using existing pipeline...")
        eval_df = nlp_analyzer.run_sentiment_analysis(eval_df, sentiment_pipe)
        print("Sentiment analysis completed")
        
        # Run EXISTING sarcasm analysis pipeline
        print("\nRunning sarcasm analysis using existing pipeline...")
        eval_df = sarcasm_analyzer.run_sarcasm_analysis(eval_df, sarcasm_pipe)
        print("Sarcasm analysis completed")
        
        # Extract predictions and ground truth
        print("\nExtracting predictions and ground truth labels...")
        self.predictions = pd.DataFrame({
            'message': self.df['message'],
            'actual_sentiment': self.df['sentiment'],
            'predicted_sentiment': eval_df['sentiment'],
            'sentiment_confidence': eval_df['sentiment_score'],
            'actual_sarcasm': self.df['is_sarcastic'],
            'predicted_sarcasm': eval_df['is_sarcastic'],
            'sarcasm_confidence': eval_df['sarcasm_score']
        })
        
        # Filter out messages that couldn't be analyzed (NaN predictions)
        initial_count = len(self.predictions)
        self.predictions = self.predictions.dropna(subset=['predicted_sentiment', 'predicted_sarcasm'])
        final_count = len(self.predictions)
        
        if final_count < initial_count:
            filtered_count = initial_count - final_count
            print(f"Filtered out {filtered_count} messages that couldn't be analyzed by existing models")
            print(f"(These were likely media messages, notifications, etc.)")
        
        print(f"\nFinal dataset for evaluation: {final_count:,} messages")
        return True
    
    def evaluate_sentiment_performance(self):
        """Evaluate sentiment prediction performance using existing model."""
        print("\n" + "="*60)
        print("STEP 3: SENTIMENT ANALYSIS EVALUATION")
        print("="*60)
        
        y_true = self.predictions['actual_sentiment']
        y_pred = self.predictions['predicted_sentiment']
        
        # Calculate all required metrics
        accuracy = accuracy_score(y_true, y_pred)
        precision_macro = precision_score(y_true, y_pred, average='macro')
        recall_macro = recall_score(y_true, y_pred, average='macro')
        f1_macro = f1_score(y_true, y_pred, average='macro')
        precision_weighted = precision_score(y_true, y_pred, average='weighted')
        recall_weighted = recall_score(y_true, y_pred, average='weighted')
        f1_weighted = f1_score(y_true, y_pred, average='weighted')
        
        # Display results
        print(f"Sentiment Classification Results:")
        print(f"  Accuracy: {accuracy:.4f}")
        print(f"  Precision (Macro): {precision_macro:.4f}")
        print(f"  Recall (Macro): {recall_macro:.4f}")
        print(f"  F1-Score (Macro): {f1_macro:.4f}")
        print(f"  Precision (Weighted): {precision_weighted:.4f}")
        print(f"  Recall (Weighted): {recall_weighted:.4f}")
        print(f"  F1-Score (Weighted): {f1_weighted:.4f}")
        
        # Detailed classification report
        print(f"\nDetailed Classification Report:")
        print(classification_report(y_true, y_pred))
        
        # Save metrics
        metrics = {
            'accuracy': accuracy,
            'precision_macro': precision_macro,
            'recall_macro': recall_macro,
            'f1_macro': f1_macro,
            'precision_weighted': precision_weighted,
            'recall_weighted': recall_weighted,
            'f1_weighted': f1_weighted
        }
        
        metrics_df = pd.DataFrame([metrics])
        metrics_df.to_csv(f"{self.results_dir}/sentiment/sentiment_metrics.csv", index=False)
        print(f"Metrics saved to: {self.results_dir}/sentiment/sentiment_metrics.csv")
        
        # Save detailed classification report
        report_dict = classification_report(y_true, y_pred, output_dict=True)
        report_df = pd.DataFrame(report_dict).transpose()
        report_df.to_csv(f"{self.results_dir}/sentiment/sentiment_classification_report.csv")
        print(f"Classification report saved to: {self.results_dir}/sentiment/sentiment_classification_report.csv")
        
        # Create and save confusion matrix
        self.create_confusion_matrix(
            y_true, y_pred,
            title="Sentiment Prediction Confusion Matrix",
            save_path=f"{self.results_dir}/sentiment/sentiment_confusion_matrix.png"
        )
        
        # Save predictions
        sentiment_predictions = self.predictions[[
            'message', 'actual_sentiment', 'predicted_sentiment', 'sentiment_confidence'
        ]].copy()
        sentiment_predictions.to_csv(f"{self.results_dir}/sentiment/sentiment_predictions.csv", index=False)
        print(f"All predictions saved to: {self.results_dir}/sentiment/sentiment_predictions.csv")
        
        return metrics
    
    def evaluate_sarcasm_performance(self):
        """Evaluate sarcasm prediction performance using existing model."""
        print("\n" + "="*60)
        print("STEP 4: SARCASM ANALYSIS EVALUATION")
        print("="*60)
        
        # Clean and convert sarcasm data to boolean
        y_true = self.predictions['actual_sarcasm'].astype(bool)
        y_pred = self.predictions['predicted_sarcasm'].astype(bool)
        
        # Remove any NaN values
        mask = ~(pd.isna(self.predictions['actual_sarcasm']) | pd.isna(self.predictions['predicted_sarcasm']))
        y_true = y_true[mask]
        y_pred = y_pred[mask]
        
        print(f"Evaluating sarcasm on {len(y_true)} clean samples...")
        
        # Calculate all required metrics
        accuracy = accuracy_score(y_true, y_pred)
        precision_macro = precision_score(y_true, y_pred, average='macro')
        recall_macro = recall_score(y_true, y_pred, average='macro')
        f1_macro = f1_score(y_true, y_pred, average='macro')
        precision_weighted = precision_score(y_true, y_pred, average='weighted')
        recall_weighted = recall_score(y_true, y_pred, average='weighted')
        f1_weighted = f1_score(y_true, y_pred, average='weighted')
        
        # Display results
        print(f"Sarcasm Detection Results:")
        print(f"  Accuracy: {accuracy:.4f}")
        print(f"  Precision (Macro): {precision_macro:.4f}")
        print(f"  Recall (Macro): {recall_macro:.4f}")
        print(f"  F1-Score (Macro): {f1_macro:.4f}")
        print(f"  Precision (Weighted): {precision_weighted:.4f}")
        print(f"  Recall (Weighted): {recall_weighted:.4f}")
        print(f"  F1-Score (Weighted): {f1_weighted:.4f}")
        
        # Detailed classification report
        class_names = ['Not Sarcastic', 'Sarcastic']
        print(f"\nDetailed Classification Report:")
        class_report = classification_report(y_true, y_pred, target_names=class_names)
        print(class_report)
        
        # Save metrics
        metrics = {
            'accuracy': accuracy,
            'precision_macro': precision_macro,
            'recall_macro': recall_macro,
            'f1_macro': f1_macro,
            'precision_weighted': precision_weighted,
            'recall_weighted': recall_weighted,
            'f1_weighted': f1_weighted
        }
        
        metrics_df = pd.DataFrame([metrics])
        metrics_df.to_csv(f"{self.results_dir}/sarcasm/sarcasm_metrics.csv", index=False)
        print(f"Metrics saved to: {self.results_dir}/sarcasm/sarcasm_metrics.csv")
        
        # Save detailed classification report
        report_dict = classification_report(y_true, y_pred, target_names=class_names, output_dict=True)
        report_df = pd.DataFrame(report_dict).transpose()
        report_df.to_csv(f"{self.results_dir}/sarcasm/sarcasm_classification_report.csv")
        print(f"Classification report saved to: {self.results_dir}/sarcasm/sarcasm_classification_report.csv")
        
        # Create and save sarcasm confusion matrix PNG
        self.create_confusion_matrix(
            y_true, y_pred,
            title="Sarcasm Detection Confusion Matrix",
            labels=['Not Sarcastic', 'Sarcastic'],
            save_path=f"{self.results_dir}/sarcasm/sarcasm_confusion_matrix.png"
        )
        
        # Save predictions
        sarcasm_predictions = self.predictions[[
            'message', 'actual_sarcasm', 'predicted_sarcasm', 'sarcasm_confidence'
        ]].copy()
        sarcasm_predictions.to_csv(f"{self.results_dir}/sarcasm/sarcasm_predictions.csv", index=False)
        print(f"All predictions saved to: {self.results_dir}/sarcasm/sarcasm_predictions.csv")
        
        return metrics
    
    def evaluate_sarcastic_vs_nonsarcastic_sentiment(self):
        """
        Evaluate sentiment performance separately for sarcastic vs non-sarcastic messages.
        This analysis is critical to understand if sarcasm affects sentiment classification.
        """
        print("\n" + "="*60)
        print("STEP 5: SARCASTIC vs NON-SARCASTIC SENTIMENT ANALYSIS")
        print("="*60)
        
        results = {}
        
        # Separate sarcastic and non-sarcastic messages
        sarcastic_mask = self.predictions['actual_sarcasm'] == True
        non_sarcastic_mask = self.predictions['actual_sarcasm'] == False
        
        sarcastic_data = self.predictions[sarcastic_mask]
        non_sarcastic_data = self.predictions[non_sarcastic_mask]
        
        print(f"Total messages: {len(self.predictions):,}")
        print(f"Sarcastic messages: {len(sarcastic_data):,}")
        print(f"Non-sarcastic messages: {len(non_sarcastic_data):,}")
        
        # Evaluate sentiment on NON-SARCASTIC messages
        if len(non_sarcastic_data) > 0:
            print(f"\nEvaluating sentiment performance on NON-SARCASTIC messages:")
            
            y_true_non = non_sarcastic_data['actual_sentiment']
            y_pred_non = non_sarcastic_data['predicted_sentiment']
            
            acc_non = accuracy_score(y_true_non, y_pred_non)
            precision_macro_non = precision_score(y_true_non, y_pred_non, average='macro')
            recall_macro_non = recall_score(y_true_non, y_pred_non, average='macro')
            f1_macro_non = f1_score(y_true_non, y_pred_non, average='macro')
            f1_weighted_non = f1_score(y_true_non, y_pred_non, average='weighted')
            
            print(f"  Accuracy: {acc_non:.4f}")
            print(f"  Precision (Macro): {precision_macro_non:.4f}")
            print(f"  Recall (Macro): {recall_macro_non:.4f}")
            print(f"  F1-Score (Macro): {f1_macro_non:.4f}")
            print(f"  F1-Score (Weighted): {f1_weighted_non:.4f}")
            
            # Save detailed non-sarcastic sentiment analysis
            non_sarc_report = classification_report(y_true_non, y_pred_non, output_dict=True)
            non_sarc_df = pd.DataFrame(non_sarc_report).transpose()
            non_sarc_df.to_csv(f"{self.results_dir}/sentiment/non_sarcastic_sentiment_report.csv")
            print(f"  Non-sarcastic sentiment report saved to: sentiment/non_sarcastic_sentiment_report.csv")
            
            results['non_sarcastic'] = {
                'count': len(non_sarcastic_data),
                'accuracy': acc_non,
                'precision_macro': precision_macro_non,
                'recall_macro': recall_macro_non,
                'f1_macro': f1_macro_non,
                'f1_weighted': f1_weighted_non
            }
        
        # Evaluate sentiment on SARCASTIC messages
        if len(sarcastic_data) > 0:
            print(f"\nEvaluating sentiment performance on SARCASTIC messages:")
            
            y_true_sarc = sarcastic_data['actual_sentiment']
            y_pred_sarc = sarcastic_data['predicted_sentiment']
            
            acc_sarc = accuracy_score(y_true_sarc, y_pred_sarc)
            precision_macro_sarc = precision_score(y_true_sarc, y_pred_sarc, average='macro')
            recall_macro_sarc = recall_score(y_true_sarc, y_pred_sarc, average='macro')
            f1_macro_sarc = f1_score(y_true_sarc, y_pred_sarc, average='macro')
            f1_weighted_sarc = f1_score(y_true_sarc, y_pred_sarc, average='weighted')
            
            print(f"  Accuracy: {acc_sarc:.4f}")
            print(f"  Precision (Macro): {precision_macro_sarc:.4f}")
            print(f"  Recall (Macro): {recall_macro_sarc:.4f}")
            print(f"  F1-Score (Macro): {f1_macro_sarc:.4f}")
            print(f"  F1-Score (Weighted): {f1_weighted_sarc:.4f}")
            
            # Save detailed sarcastic sentiment analysis
            sarc_report = classification_report(y_true_sarc, y_pred_sarc, output_dict=True)
            sarc_df = pd.DataFrame(sarc_report).transpose()
            sarc_df.to_csv(f"{self.results_dir}/sentiment/sarcastic_sentiment_report.csv")
            print(f"  Sarcastic sentiment report saved to: sentiment/sarcastic_sentiment_report.csv")
            
            results['sarcastic'] = {
                'count': len(sarcastic_data),
                'accuracy': acc_sarc,
                'precision_macro': precision_macro_sarc,
                'recall_macro': recall_macro_sarc,
                'f1_macro': f1_macro_sarc,
                'f1_weighted': f1_weighted_sarc
            }
        
        # Calculate performance difference
        if 'non_sarcastic' in results and 'sarcastic' in results:
            acc_difference = results['non_sarcastic']['accuracy'] - results['sarcastic']['accuracy']
            f1_difference = results['non_sarcastic']['f1_macro'] - results['sarcastic']['f1_macro']
            
            print(f"\nPERFORMANCE IMPACT OF SARCASM:")
            print(f"  Accuracy drop due to sarcasm: {acc_difference:.4f} ({acc_difference*100:.1f}%)")
            print(f"  F1-macro drop due to sarcasm: {f1_difference:.4f} ({f1_difference*100:.1f}%)")
            
            results['performance_impact'] = {
                'accuracy_drop': acc_difference,
                'f1_macro_drop': f1_difference
            }
        
        # Save comparative results
        if results:
            results_df = pd.DataFrame(results).transpose()
            results_df.to_csv(f"{self.results_dir}/combined/sarcastic_vs_nonsarcastic_sentiment.csv")
            print(f"\nComparative results saved to: {self.results_dir}/combined/sarcastic_vs_nonsarcastic_sentiment.csv")
            
            # Save detailed breakdown
            detailed_results = {
                'total_messages': len(self.predictions),
                'sarcastic_count': len(sarcastic_data),
                'non_sarcastic_count': len(non_sarcastic_data),
                **results
            }
            
            import json
            with open(f"{self.results_dir}/combined/detailed_sarcastic_vs_nonsarcastic_stats.json", "w") as f:
                json.dump(detailed_results, f, indent=2, default=str)
            print(f"Detailed statistics saved to: combined/detailed_sarcastic_vs_nonsarcastic_stats.json")
        
        return results
    
    def evaluate_combined_performance(self):
        """
        Evaluate combined sentiment + sarcasm prediction accuracy.
        This shows how often BOTH models are correct simultaneously.
        """
        print("\n" + "="*60)
        print("STEP 6: COMBINED PERFORMANCE EVALUATION")
        print("="*60)
        
        # Calculate different combination scenarios
        both_correct = (
            (self.predictions['actual_sentiment'] == self.predictions['predicted_sentiment']) &
            (self.predictions['actual_sarcasm'] == self.predictions['predicted_sarcasm'])
        )
        
        both_incorrect = (
            (self.predictions['actual_sentiment'] != self.predictions['predicted_sentiment']) &
            (self.predictions['actual_sarcasm'] != self.predictions['predicted_sarcasm'])
        )
        
        sentiment_correct_sarcasm_incorrect = (
            (self.predictions['actual_sentiment'] == self.predictions['predicted_sentiment']) &
            (self.predictions['actual_sarcasm'] != self.predictions['predicted_sarcasm'])
        )
        
        sarcasm_correct_sentiment_incorrect = (
            (self.predictions['actual_sentiment'] != self.predictions['predicted_sentiment']) &
            (self.predictions['actual_sarcasm'] == self.predictions['predicted_sarcasm'])
        )
        
        # Count occurrences
        total_samples = len(self.predictions)
        both_correct_count = both_correct.sum()
        both_incorrect_count = both_incorrect.sum()
        sent_correct_sarc_incorrect_count = sentiment_correct_sarcasm_incorrect.sum()
        sarc_correct_sent_incorrect_count = sarcasm_correct_sentiment_incorrect.sum()
        
        # Calculate percentages
        both_correct_pct = both_correct_count / total_samples * 100
        both_incorrect_pct = both_incorrect_count / total_samples * 100
        sent_correct_sarc_incorrect_pct = sent_correct_sarc_incorrect_count / total_samples * 100
        sarc_correct_sent_incorrect_pct = sarc_correct_sent_incorrect_count / total_samples * 100
        
        # Combined accuracy = both must be correct
        combined_accuracy = both_correct_count / total_samples
        
        # Display results
        print(f"Combined Performance Analysis:")
        print(f"  Total samples: {total_samples:,}")
        print(f"  Both correct: {both_correct_count:,} ({both_correct_pct:.1f}%)")
        print(f"  Both incorrect: {both_incorrect_count:,} ({both_incorrect_pct:.1f}%)")
        print(f"  Sentiment correct, sarcasm incorrect: {sent_correct_sarc_incorrect_count:,} ({sent_correct_sarc_incorrect_pct:.1f}%)")
        print(f"  Sarcasm correct, sentiment incorrect: {sarc_correct_sent_incorrect_count:,} ({sarc_correct_sent_incorrect_pct:.1f}%)")
        print(f"\n  Combined Accuracy (both must be correct): {combined_accuracy:.4f}")
        
        # Save results
        combined_results = {
            'total_samples': total_samples,
            'both_correct': both_correct_count,
            'both_correct_percentage': both_correct_pct,
            'both_incorrect': both_incorrect_count,
            'both_incorrect_percentage': both_incorrect_pct,
            'sentiment_correct_sarcasm_incorrect': sent_correct_sarc_incorrect_count,
            'sentiment_correct_sarcasm_incorrect_percentage': sent_correct_sarc_incorrect_pct,
            'sarcasm_correct_sentiment_incorrect': sarc_correct_sent_incorrect_count,
            'sarcasm_correct_sentiment_incorrect_percentage': sarc_correct_sent_incorrect_pct,
            'combined_accuracy': combined_accuracy
        }
        
        combined_df = pd.DataFrame([combined_results])
        combined_df.to_csv(f"{self.results_dir}/combined/combined_performance.csv", index=False)
        print(f"Combined performance saved to: {self.results_dir}/combined/combined_performance.csv")
        
        return combined_results
    
    def create_error_analysis_files(self):
        """Create detailed error analysis CSV files for incorrect predictions."""
        print("\n" + "="*60)
        print("STEP 7: ERROR ANALYSIS")
        print("="*60)
        
        # Sentiment errors only
        sentiment_errors = self.predictions[
            self.predictions['actual_sentiment'] != self.predictions['predicted_sentiment']
        ].copy()
        
        if len(sentiment_errors) > 0:
            sentiment_error_file = f"{self.results_dir}/sentiment/sentiment_errors.csv"
            sentiment_errors[[
                'message', 'actual_sentiment', 'predicted_sentiment', 
                'sentiment_confidence', 'actual_sarcasm'
            ]].to_csv(sentiment_error_file, index=False)
            print(f"Sentiment errors: {len(sentiment_errors):,} saved to {sentiment_error_file}")
        else:
            print("No sentiment errors found (perfect sentiment classification!)")
        
        # Sarcasm errors only
        sarcasm_errors = self.predictions[
            self.predictions['actual_sarcasm'] != self.predictions['predicted_sarcasm']
        ].copy()
        
        if len(sarcasm_errors) > 0:
            sarcasm_error_file = f"{self.results_dir}/sarcasm/sarcasm_errors.csv"
            sarcasm_errors[[
                'message', 'actual_sarcasm', 'predicted_sarcasm',
                'sarcasm_confidence', 'actual_sentiment'
            ]].to_csv(sarcasm_error_file, index=False)
            print(f"Sarcasm errors: {len(sarcasm_errors):,} saved to {sarcasm_error_file}")
        else:
            print("No sarcasm errors found (perfect sarcasm classification!)")
        
        # Combined errors (either sentiment OR sarcasm wrong)
        combined_errors = self.predictions[
            (self.predictions['actual_sentiment'] != self.predictions['predicted_sentiment']) |
            (self.predictions['actual_sarcasm'] != self.predictions['predicted_sarcasm'])
        ].copy()
        
        if len(combined_errors) > 0:
            combined_error_file = f"{self.results_dir}/combined/combined_errors.csv"
            combined_errors.to_csv(combined_error_file, index=False)
            print(f"Combined errors: {len(combined_errors):,} saved to {combined_error_file}")
        else:
            print("No combined errors found (perfect classification on both models!)")
        
        return len(sentiment_errors), len(sarcasm_errors), len(combined_errors)
    
    def create_confusion_matrix(self, y_true, y_pred, title, labels=None, save_path=None):
        """Create and save confusion matrix visualization."""
        cm = confusion_matrix(y_true, y_pred)
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=labels or sorted(set(y_true)),
                    yticklabels=labels or sorted(set(y_true)))
        plt.title(title)
        plt.ylabel('Actual')
        plt.xlabel('Predicted')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Confusion matrix saved: {save_path}")
        
        plt.close()  # Close to free memory
    
    def generate_final_report(self, sentiment_metrics, sarcasm_metrics, 
                             comparative_results, combined_results, error_counts):
        """Generate comprehensive final evaluation report."""
        print("\n" + "="*60)
        print("STEP 8: GENERATING FINAL REPORT")
        print("="*60)
        
        report_lines = []
        
        # Header
        report_lines.append("WhatsApp NLP Project - Evaluation Report")
        report_lines.append("="*60)
        report_lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report_lines.append(f"Dataset: {self.dataset_path}")
        report_lines.append("")
        
        # Models evaluated
        report_lines.append("MODELS EVALUATED:")
        report_lines.append("  Sentiment Model: shae2977/xlm-roberta-hinglish-sentiment-analysis")
        report_lines.append("  Sarcasm Model: amaan00z/sarcasm_xlmr")
        report_lines.append("  Pipeline: EXACT same as main application")
        report_lines.append("")
        
        # Dataset information
        report_lines.append("DATASET INFORMATION:")
        report_lines.append(f"  Total samples: {len(self.df):,}")
        report_lines.append(f"  Analyzed samples: {len(self.predictions):,}")
        
        # Dataset distributions
        sentiment_dist = self.df['sentiment'].value_counts()
        report_lines.append("  Sentiment distribution:")
        for sent, count in sentiment_dist.items():
            pct = count / len(self.df) * 100
            report_lines.append(f"    {sent}: {count:,} ({pct:.1f}%)")
        
        sarcasm_dist = self.df['is_sarcastic'].value_counts()
        report_lines.append("  Sarcasm distribution:")
        for is_sarc, count in sarcasm_dist.items():
            label = "Sarcastic" if is_sarc else "Not Sarcastic"
            pct = count / len(self.df) * 100
            report_lines.append(f"    {label}: {count:,} ({pct:.1f}%)")
        
        report_lines.append("")
        
        # Sentiment performance
        report_lines.append("SENTIMENT ANALYSIS RESULTS:")
        report_lines.append(f"  Accuracy: {sentiment_metrics['accuracy']:.4f}")
        report_lines.append(f"  Precision (Macro): {sentiment_metrics['precision_macro']:.4f}")
        report_lines.append(f"  Recall (Macro): {sentiment_metrics['recall_macro']:.4f}")
        report_lines.append(f"  F1-Score (Macro): {sentiment_metrics['f1_macro']:.4f}")
        report_lines.append(f"  F1-Score (Weighted): {sentiment_metrics['f1_weighted']:.4f}")
        report_lines.append("")
        
        # Sarcasm performance
        report_lines.append("SARCASM DETECTION RESULTS:")
        report_lines.append(f"  Accuracy: {sarcasm_metrics['accuracy']:.4f}")
        report_lines.append(f"  Precision (Macro): {sarcasm_metrics['precision_macro']:.4f}")
        report_lines.append(f"  Recall (Macro): {sarcasm_metrics['recall_macro']:.4f}")
        report_lines.append(f"  F1-Score (Macro): {sarcasm_metrics['f1_macro']:.4f}")
        report_lines.append(f"  F1-Score (Weighted): {sarcasm_metrics['f1_weighted']:.4f}")
        report_lines.append("")
        
        # Sarcastic vs Non-sarcastic sentiment analysis
        if comparative_results:
            report_lines.append("SARCASTIC vs NON-SARCASTIC SENTIMENT PERFORMANCE:")
            if 'non_sarcastic' in comparative_results:
                non_sarc = comparative_results['non_sarcastic']
                report_lines.append(f"  Non-sarcastic messages ({non_sarc['count']:,}):")
                report_lines.append(f"    Accuracy: {non_sarc['accuracy']:.4f}")
                report_lines.append(f"    F1-Score (Macro): {non_sarc['f1_macro']:.4f}")
            
            if 'sarcastic' in comparative_results:
                sarc = comparative_results['sarcastic']
                report_lines.append(f"  Sarcastic messages ({sarc['count']:,}):")
                report_lines.append(f"    Accuracy: {sarc['accuracy']:.4f}")
                report_lines.append(f"    F1-Score (Macro): {sarc['f1_macro']:.4f}")
            report_lines.append("")
        
        # Combined performance
        report_lines.append("COMBINED PERFORMANCE:")
        report_lines.append(f"  Both predictions correct: {combined_results['both_correct']:,} ({combined_results['both_correct_percentage']:.1f}%)")
        report_lines.append(f"  Both predictions incorrect: {combined_results['both_incorrect']:,} ({combined_results['both_incorrect_percentage']:.1f}%)")
        report_lines.append(f"  Only sentiment correct: {combined_results['sentiment_correct_sarcasm_incorrect']:,} ({combined_results['sentiment_correct_sarcasm_incorrect_percentage']:.1f}%)")
        report_lines.append(f"  Only sarcasm correct: {combined_results['sarcasm_correct_sentiment_incorrect']:,} ({combined_results['sarcasm_correct_sentiment_incorrect_percentage']:.1f}%)")
        report_lines.append(f"  Combined accuracy: {combined_results['combined_accuracy']:.4f}")
        report_lines.append("")
        
        # Error analysis
        sent_errors, sarc_errors, comb_errors = error_counts
        report_lines.append("ERROR ANALYSIS:")
        report_lines.append(f"  Sentiment errors: {sent_errors:,}")
        report_lines.append(f"  Sarcasm errors: {sarc_errors:,}")
        report_lines.append(f"  Combined errors: {comb_errors:,}")
        report_lines.append("")
        
        # Output files
        report_lines.append("GENERATED FILES:")
        report_lines.append("  evaluate/results/sentiment/sentiment_metrics.csv")
        report_lines.append("  evaluate/results/sentiment/sentiment_classification_report.csv")
        report_lines.append("  evaluate/results/sentiment/sentiment_confusion_matrix.png")
        report_lines.append("  evaluate/results/sentiment/sentiment_predictions.csv")
        report_lines.append("  evaluate/results/sentiment/sentiment_errors.csv")
        report_lines.append("  evaluate/results/sarcasm/sarcasm_metrics.csv")
        report_lines.append("  evaluate/results/sarcasm/sarcasm_classification_report.csv")
        report_lines.append("  evaluate/results/sarcasm/sarcasm_confusion_matrix.png")
        report_lines.append("  evaluate/results/sarcasm/sarcasm_predictions.csv")
        report_lines.append("  evaluate/results/sarcasm/sarcasm_errors.csv")
        report_lines.append("  evaluate/results/combined/sarcastic_vs_nonsarcastic_sentiment.csv")
        report_lines.append("  evaluate/results/combined/combined_performance.csv")
        report_lines.append("  evaluate/results/combined/combined_errors.csv")
        report_lines.append("  evaluate/results/reports/evaluation_report.txt")
        
        # Save report
        report_text = "\n".join(report_lines)
        report_file = f"{self.results_dir}/reports/evaluation_report.txt"
        with open(report_file, "w", encoding='utf-8') as f:
            f.write(report_text)
        
        print(f"Final evaluation report saved: {report_file}")
        
        return report_text
    
    def run_complete_evaluation(self):
        """Execute the complete evaluation pipeline."""
        print("🚀 WhatsApp NLP Project - Model Evaluation")
        print("Using EXISTING models and pipeline from main application")
        print("Results will be saved to files with minimal console output")
        print("="*60)
        
        # Create log file for detailed output
        self.log_file = f"{self.results_dir}/reports/detailed_evaluation_log.txt"
        self.log_lines = []
        
        try:
            # Step 1: Load and validate dataset
            print("🔄 Step 1/8: Loading and validating dataset...")
            self.load_and_validate_dataset()
            
            # Step 2: Run predictions using existing models
            print("🔄 Step 2/8: Running model predictions...")
            self.run_existing_model_predictions()
            
            # Step 3: Evaluate sentiment performance
            print("🔄 Step 3/8: Evaluating sentiment performance...")
            sentiment_metrics = self.evaluate_sentiment_performance()
            
            # Step 4: Evaluate sarcasm performance
            print("🔄 Step 4/8: Evaluating sarcasm performance...")
            sarcasm_metrics = self.evaluate_sarcasm_performance()
            
            # Step 5: Evaluate sentiment for sarcastic vs non-sarcastic messages
            print("🔄 Step 5/8: Analyzing sarcastic vs non-sarcastic performance...")
            comparative_results = self.evaluate_sarcastic_vs_nonsarcastic_sentiment()
            
            # Step 6: Evaluate combined performance
            print("🔄 Step 6/8: Evaluating combined performance...")
            combined_results = self.evaluate_combined_performance()
            
            # Step 7: Create error analysis files
            print("🔄 Step 7/8: Creating error analysis...")
            error_counts = self.create_error_analysis_files()
            
            # Step 8: Generate final comprehensive report
            print("🔄 Step 8/8: Generating final report...")
            self.generate_final_report(
                sentiment_metrics, sarcasm_metrics, 
                comparative_results, combined_results, error_counts
            )
            
            print("\n✅ EVALUATION COMPLETED SUCCESSFULLY!")
            print("="*60)
            
            # Summary of key results - minimal console output
            print("📊 SUMMARY:")
            print(f"   📝 Messages Evaluated: {len(self.predictions):,}")
            print(f"   😊 Sentiment Accuracy: {sentiment_metrics['accuracy']:.1%}")
            print(f"   🎭 Sarcasm Accuracy: {sarcasm_metrics['accuracy']:.1%}")
            print(f"   🎯 Combined Accuracy: {combined_results['combined_accuracy']:.1%}")
            
            print(f"\n📁 All detailed results saved to: {self.results_dir}/")
            print(f"📋 Main report: {self.results_dir}/reports/evaluation_report.txt")
            print(f"📋 Detailed log: {self.results_dir}/reports/detailed_evaluation_log.txt")
            
        except Exception as e:
            print(f"\n❌ ERROR during evaluation: {e}")
            raise


def main():
    """Main execution function."""
    evaluator = WhatsAppNLPEvaluator()
    evaluator.run_complete_evaluation()


if __name__ == "__main__":
    main()
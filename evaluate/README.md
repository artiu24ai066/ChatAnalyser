# WhatsApp NLP Project - Evaluation Module

This evaluation module provides comprehensive performance assessment for the existing WhatsApp Chat Analyzer NLP models without modifying any existing project files or functionality.

## Overview

The evaluation system tests the **EXACT same models and processing pipeline** as the main application against a ground-truth dataset of labeled Hinglish/WhatsApp-style messages.

**CRITICAL**: This module reuses the existing model loading functions, preprocessing logic, and inference pipeline to ensure identical behavior to what users experience in the main application.

## Models Evaluated

- **Sentiment Analysis**: `shae2977/xlm-roberta-hinglish-sentiment-analysis`
  - Uses: `nlp_analyzer.load_sentiment_model()` and `nlp_analyzer.run_sentiment_analysis()`
  - Labels: Positive, Neutral, Negative

- **Sarcasm Detection**: `amaan00z/sarcasm_xlmr` 
  - Uses: `sarcasm_analyzer.load_sarcasm_model()` and `sarcasm_analyzer.run_sarcasm_analysis()`
  - Labels: True (Sarcastic), False (Not Sarcastic)

## Dataset Requirements

### Dataset Location
Place your dataset at: `evaluate/dataset/whatsapp_hinglish_messages.csv`

### Required Format
The CSV file must contain exactly these columns:

- **`message`** (string): The WhatsApp message text
- **`sentiment`** (string): Ground-truth sentiment label
  - Valid values: "Positive", "Neutral", "Negative" (case-sensitive)
- **`is_sarcastic`** (boolean): Ground-truth sarcasm label  
  - Valid values: `True`, `False` (boolean values, not strings)

### Example Format
```csv
message,sentiment,is_sarcastic
"yaar aaj bahut acha laga",Positive,False
"bhai kya bakwas hai",Negative,False
"kal milte hain",Neutral,False
"haan haan genius hai tu",Positive,True
```

## Directory Structure

```
evaluate/
├── dataset/
│   └── whatsapp_hinglish messages.csv    # Your labeled dataset
├── results/
│   ├── sentiment/                        # Sentiment evaluation results
│   │   ├── sentiment_metrics.csv
│   │   ├── sentiment_classification_report.csv
│   │   ├── sentiment_confusion_matrix.png
│   │   ├── sentiment_predictions.csv
│   │   ├── sentiment_errors.csv
│   │   ├── error_confidence_distribution.png
│   │   └── length_vs_confidence_errors.png
│   ├── sarcasm/                          # Sarcasm evaluation results  
│   │   ├── sarcasm_metrics.csv
│   │   ├── sarcasm_classification_report.csv
│   │   ├── sarcasm_confusion_matrix.png
│   │   ├── sarcasm_predictions.csv
│   │   ├── sarcasm_errors.csv
│   │   └── error_confidence_distribution.png
│   ├── combined/                         # Combined analysis results
│   │   ├── sarcastic_vs_nonsarcastic_sentiment.csv
│   │   ├── combined_performance.csv
│   │   └── combined_errors.csv
│   └── reports/                          # Summary reports
│       ├── evaluation_report.txt
│       └── error_analysis_report.txt
├── evaluate.py                           # Main evaluation script
├── error_analysis.py                     # Advanced error analysis
└── README.md                            # This documentation
```

## Usage

### 1. Basic Evaluation

Run the complete evaluation pipeline:

```bash
python evaluate/evaluate.py
```

This will:
1. **Load and validate** your dataset
2. **Run predictions** using existing models (identical to main app)
3. **Calculate metrics** for sentiment and sarcasm performance
4. **Generate confusion matrices** and visualizations
5. **Perform comparative analysis** (sarcastic vs non-sarcastic sentiment)
6. **Evaluate combined performance** (both models correct simultaneously)
7. **Create error analysis files** with incorrect predictions
8. **Generate comprehensive report** with all results

### 2. Advanced Error Analysis

For detailed error pattern analysis:

```bash
python evaluate/error_analysis.py
```

This provides:
- **Error pattern analysis** (common mistake types)
- **Confidence score distributions** for errors
- **Message length correlations** with errors  
- **False positive/negative breakdown** for sarcasm
- **Error interaction analysis** (when both models fail)
- **Visualizations** of error patterns

## Evaluation Methodology

### 1. Existing Pipeline Reuse
The evaluation uses the **identical** functions as the main application:

```python
# Sentiment Analysis (EXACT same as main app)
sentiment_pipe = nlp_analyzer.load_sentiment_model()
eval_df = nlp_analyzer.run_sentiment_analysis(eval_df, sentiment_pipe)

# Sarcasm Analysis (EXACT same as main app)  
sarcasm_pipe = sarcasm_analyzer.load_sarcasm_model()
eval_df = sarcasm_analyzer.run_sarcasm_analysis(eval_df, sarcasm_pipe)
```

### 2. Ground Truth Separation
- Ground-truth labels are **NEVER** provided to the models
- Models make **independent predictions** on message text only
- Predictions are compared with ground truth **after** inference

### 3. Comprehensive Metrics
For both sentiment and sarcasm:
- **Accuracy**: Overall correct predictions
- **Precision (Macro/Weighted)**: Per-class precision averages
- **Recall (Macro/Weighted)**: Per-class recall averages
- **F1-Score (Macro/Weighted)**: Harmonic mean of precision/recall
- **Classification Reports**: Detailed per-class breakdowns
- **Confusion Matrices**: Visual prediction pattern analysis

### 4. Specialized Analyses

#### Sarcastic vs Non-Sarcastic Sentiment Performance
- Evaluates sentiment accuracy **separately** for sarcastic and non-sarcastic messages
- Determines if sarcasm detection affects sentiment classification accuracy
- Critical for understanding model interactions

#### Combined Performance Analysis
- **Both Correct**: Messages where both sentiment and sarcasm are predicted correctly
- **Both Incorrect**: Messages where both models fail
- **Partial Correct**: Messages where only one model is correct
- **Combined Accuracy**: Percentage where both predictions are simultaneously correct

## Output Files

### Key Results
- **`evaluate/results/reports/evaluation_report.txt`**: Comprehensive summary of all results
- **`evaluate/results/sentiment/sentiment_confusion_matrix.png`**: Sentiment prediction patterns
- **`evaluate/results/sarcasm/sarcasm_confusion_matrix.png`**: Sarcasm prediction patterns

### Detailed Analysis  
- **`evaluate/results/sentiment/sentiment_errors.csv`**: All incorrect sentiment predictions
- **`evaluate/results/sarcasm/sarcasm_errors.csv`**: All incorrect sarcasm predictions
- **`evaluate/results/combined/combined_errors.csv`**: All prediction errors (either model)
- **`evaluate/results/combined/sarcastic_vs_nonsarcastic_sentiment.csv`**: Comparative sentiment performance

### Performance Metrics
- **`evaluate/results/sentiment/sentiment_metrics.csv`**: Numerical sentiment performance metrics
- **`evaluate/results/sarcasm/sarcasm_metrics.csv`**: Numerical sarcasm performance metrics  
- **`evaluate/results/combined/combined_performance.csv`**: Combined model performance metrics

### Predictions
- **`evaluate/results/sentiment/sentiment_predictions.csv`**: All sentiment predictions with confidence scores
- **`evaluate/results/sarcasm/sarcasm_predictions.csv`**: All sarcasm predictions with confidence scores

## Important Notes

### Model Consistency
- **No modifications** are made to existing models, preprocessing, or inference logic
- **Identical behavior** to the main WhatsApp Chat Analyzer application
- **Same message filtering** logic (skips media, notifications, etc.)
- **Same label mappings** and post-processing steps

### Performance Interpretation
- **Confidence scores** represent model certainty (0-1), not emotional intensity
- **Sarcasm detection** is inherently challenging and context-dependent
- **Hinglish text** presents unique challenges for NLP models due to code-mixing
- **Results** should be interpreted as statistical estimates of model performance

### Dataset Quality Impact
- **Ground-truth quality** directly affects evaluation reliability
- **Label consistency** is critical for meaningful metrics
- **Annotation subjectivity** (especially for sarcasm) affects results
- **Data distribution** should match real-world usage patterns

### Limitations
- **Message-level analysis only** (no conversation context)
- **Independent message classification** (no inter-message dependencies)
- **Static evaluation** (no learning or adaptation during evaluation)
- **Single domain** (WhatsApp/Hinglish specific)

## Troubleshooting

### Common Issues

**Dataset not found**
```
FileNotFoundError: Dataset not found: evaluate/dataset/whatsapp_hinglish messages.csv
```
→ Ensure dataset file is placed at exact path with correct filename

**Invalid column names**
```
ValueError: Missing required columns: ['message', 'sentiment', 'is_sarcastic']
```
→ Check CSV headers match exactly (case-sensitive)

**Invalid label values**
```
ValueError: Invalid sentiment labels in dataset
```
→ Ensure sentiment values are exactly "Positive", "Neutral", "Negative"
→ Ensure sarcasm values are boolean True/False (not strings)

**Model loading failures**
```
RuntimeError: Failed to load existing sentiment model
```
→ Check internet connection for model downloads (first run only)
→ Verify transformers and torch are installed
→ Ensure sufficient memory for model loading

### Memory Issues
If evaluation fails due to memory constraints:
1. **Close other applications** to free memory
2. **Process smaller batches** by modifying BATCH_SIZE in existing modules
3. **Use CPU-only inference** if GPU memory is insufficient

### Performance Issues  
For large datasets:
1. **Monitor progress** through console output
2. **Expect longer runtime** for first model download
3. **Subsequent runs** will be faster (models cached)

## Dependencies

All dependencies are inherited from the main project:
- **pandas**: Data manipulation and analysis
- **numpy**: Numerical operations
- **matplotlib**: Visualization and plot generation
- **seaborn**: Statistical data visualization
- **scikit-learn**: Machine learning metrics and evaluation
- **transformers**: Hugging Face transformer models (existing requirement)
- **torch**: PyTorch backend for transformers (existing requirement)

## Contact & Support

This evaluation module is designed to work seamlessly with the existing WhatsApp Chat Analyzer project. 

For issues related to:
- **Main application functionality**: Refer to main project documentation
- **Model behavior or predictions**: These reflect the existing application behavior
- **Evaluation methodology or metrics**: Review this documentation
- **Dataset format or requirements**: See the Dataset Requirements section above

The evaluation results represent the true performance of your deployed NLP system as users experience it in the main application.
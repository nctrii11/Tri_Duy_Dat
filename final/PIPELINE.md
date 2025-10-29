# Titanic Survival Prediction - Complete ML Pipeline

## Project Structure

```
final/
├── main.ipynb                          # Main orchestration notebook
├── PIPELINE.md                         # This documentation
├── src/
│   ├── __init__.py                     # Package initialization
│   ├── data_preprocessing.py           # Data loading and preparation
│   ├── train_models.py                 # Baseline model training with logging
│   ├── tune_models.py                  # Hyperparameter tuning with logging
│   └── visualize_results.py            # Results visualization
├── logs/
│   ├── training_log.txt                # Detailed baseline training logs
│   ├── training_summary.csv            # Baseline metrics summary
│   ├── tuning_log.txt                  # Detailed tuning logs
│   ├── tuning_summary.csv              # Tuning results summary
│   ├── comprehensive_results.csv       # Combined results table
│   └── plots/
│       ├── 01_accuracy_comparison.png
│       ├── 02_detailed_metrics_comparison.png
│       └── 03_f1_score_comparison.png
└── README.md                           # General documentation
```

## Pipeline Overview

### 1. Data Preparation (`data_preprocessing.py`)

- **Load**: Read `train_clean.csv` and `test_clean.csv` from EDA folder
- **Split**: 80/20 train-validation split with stratification
- **Scale**: StandardScaler normalization
- **Output**: Scaled feature matrices and target vectors

### 2. Baseline Model Training (`train_models.py`)

Three models trained with detailed logging:

- **Logistic Regression** (max_iter=1000)
- **Decision Tree** (default parameters)
- **K-Nearest Neighbors** (n_neighbors=5)

**Logged Information:**

- Training start/end time
- Train set size and class distribution
- Model training time
- Validation metrics:
  - Accuracy, Precision, Recall, F1-Score
  - Confusion Matrix
  - Classification Report

**Output:**

- `logs/training_log.txt` - Detailed logs
- `logs/training_summary.csv` - Metrics summary

### 3. Hyperparameter Tuning (`tune_models.py`)

GridSearchCV tuning for each model:

**Logistic Regression:**

- C: [0.001, 0.01, 0.1, 1, 10, 100]
- penalty: ['l2']
- solver: ['lbfgs', 'liblinear']
- max_iter: [1000, 2000]

**Decision Tree:**

- max_depth: [5, 10, 15, 20, None]
- min_samples_split: [2, 5, 10]
- min_samples_leaf: [1, 2, 4]
- criterion: ['gini', 'entropy']

**K-Nearest Neighbors:**

- n_neighbors: [3, 5, 7, 9, 11, 15]
- weights: ['uniform', 'distance']
- metric: ['euclidean', 'manhattan']

**Logged Information:**

- Best parameters found
- Best CV F1-Score
- Validation metrics (on tuned model)
- Tuning time

**Output:**

- `logs/tuning_log.txt` - Detailed logs
- `logs/tuning_summary.csv` - Results summary

### 4. Visualization (`visualize_results.py`)

Three comparison charts generated:

**Chart 1: Accuracy Comparison**

- Grouped bar chart: Baseline vs Tuned
- All 3 models side-by-side
- Value labels on bars

**Chart 2: Detailed Metrics Comparison**

- Subplots for each model
- Metrics: Accuracy, Precision, Recall, F1-Score
- Before/after tuning

**Chart 3: F1-Score Comparison**

- Focused on F1-Score metric
- Baseline vs Tuned
- Value labels for clarity

## Running the Pipeline

### Option 1: Jupyter Notebook (Recommended)

```bash
# Open and run main.ipynb cell by cell
jupyter notebook main.ipynb
```

### Option 2: Python Script (if needed)

Create a `run_pipeline.py`:

```python
import sys
sys.path.insert(0, 'src')

from data_preprocessing import load_data, prepare_data
from train_models import train_all_models, save_results_summary
from tune_models import tune_all_models, save_tuning_summary
from visualize_results import plot_model_comparison, create_results_summary_table

# Execute each step
df_train, df_test = load_data("../EDA")
X_train, X_val, X_test, y_train, y_val, scaler = prepare_data(df_train, df_test)

baseline_results, _ = train_all_models(X_train, X_val, y_train, y_val)
save_results_summary(baseline_results)

tuned_results, _ = tune_all_models(X_train, X_val, y_train, y_val)
save_tuning_summary(tuned_results)

plot_model_comparison(baseline_results, tuned_results)
create_results_summary_table(baseline_results, tuned_results)
```

## Key Features

### ✓ Comprehensive Logging

- **Python logging module** for structured, timestamped logs
- Separate files for training and tuning phases
- Console output for real-time monitoring
- Detailed metrics and confusion matrices

### ✓ Complete Evaluation

- Multiple metrics: Accuracy, Precision, Recall, F1-Score
- Confusion matrices for each model
- Classification reports
- Before/after tuning comparison

### ✓ Professional Visualization

- High-resolution PNG outputs (300 DPI)
- Clear labeling and legends
- Value labels on bars
- Multiple chart types for different insights

### ✓ Reproducibility

- Fixed random seeds (42) throughout
- Stratified cross-validation
- Detailed hyperparameter logs
- CSV export for analysis

## Expected Results

### Baseline Accuracy (Typical)

- Logistic Regression: ~81%
- Decision Tree: ~78%
- KNN: ~80%

### After Tuning (Typical)

- Logistic Regression: ~82-83%
- Decision Tree: ~80-82%
- KNN: ~81-82%

_Actual results may vary based on CV splits and tuning parameter ranges_

## Notes

1. **Training Time**: GridSearchCV can take 5-15 minutes depending on your machine
2. **Stratified Split**: Maintains class distribution in train/val/test splits
3. **Feature Scaling**: StandardScaler applied to all numeric features
4. **F1-Score Focus**: GridSearchCV optimizes F1-Score as default metric (balanced for imbalanced data)

## Troubleshooting

**Issue**: Import errors when running notebook

- **Solution**: Ensure you run `sys.path.insert(0, 'src')` before imports

**Issue**: File not found for EDA data

- **Solution**: Verify `../EDA/train_clean.csv` and `../EDA/test_clean.csv` exist

**Issue**: GridSearchCV taking too long

- **Solution**: Reduce parameter grid sizes or reduce n_jobs to use fewer cores

## Future Enhancements

1. Add cross-validation within tuning process
2. Implement feature importance analysis
3. Add ROC-AUC curves
4. Implement early stopping for tree-based models
5. Add model persistence (pickle/joblib)

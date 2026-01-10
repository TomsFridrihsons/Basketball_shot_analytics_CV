# Evaluations Directory

This directory contains evaluation results, metrics, and analysis outputs.

## Structure

```
evaluations/
├── metrics/              # Evaluation metrics (CSV, JSON)
│   └── *.csv            # Metrics per experiment
├── plots/                # Visualization outputs
│   ├── confusion_matrix/
│   ├── pr_curves/
│   └── loss_curves/
├── reports/              # Evaluation reports
│   └── *.md             # Markdown reports
└── README.md            # This file
```

## Contents

### Metrics
- Precision, Recall, mAP scores
- Per-class metrics
- Training/validation metrics over time

### Plots
- Confusion matrices
- Precision-Recall curves
- Loss curves
- Detection visualizations

### Reports
- Model comparison reports
- Dataset analysis reports
- Performance summaries

## Usage

After training, evaluation results are typically saved here:
1. Metrics exported from training runs
2. Custom evaluation scripts output
3. Model comparison analyses

## Best Practices

- Name files with experiment/run identifiers
- Include timestamps in filenames
- Document evaluation methodology
- Compare results across experiments

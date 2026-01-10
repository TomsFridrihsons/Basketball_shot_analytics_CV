# Datasets Directory

This directory contains training-ready datasets in YOLO format.

## Structure

```
datasets/
├── dataset-name/          # Example: basketball-v1
│   ├── data.yaml         # Dataset configuration file
│   ├── train/            # Training split
│   │   ├── images/       # Training images
│   │   └── labels/       # Training labels (YOLO format)
│   ├── valid/            # Validation split
│   │   ├── images/       # Validation images
│   │   └── labels/       # Validation labels
│   └── test/             # Test split (optional)
│       ├── images/       # Test images
│       └── labels/       # Test labels
└── README.md             # This file
```

## Dataset Naming Convention

Use descriptive names:
- `basketball-v1` - Version 1 of basketball dataset
- `basketball-v2-enhanced` - Enhanced version 2
- `basketball-multi-court` - Multi-court dataset

## Adding a New Dataset

1. Create a new directory: `datasets/your-dataset-name/`
2. Organize into `train/`, `valid/`, and optionally `test/` splits
3. Create `data.yaml` with dataset configuration (see `configs/dataset_template.yaml`)
4. Ensure images and labels are paired correctly
5. Update this README with dataset information

## Dataset Configuration (data.yaml)

Each dataset should have a `data.yaml` file with:
- Path to dataset
- Number of classes
- Class names
- Train/valid/test paths

See `../configs/dataset_template.yaml` for a template.

## Best Practices

- Keep train/valid/test splits consistent
- Use 80/10/10 or 70/15/15 splits typically
- Ensure label files match image files (same base name)
- Document dataset statistics (number of images, classes, etc.)

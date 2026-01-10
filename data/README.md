# Data Directory

This directory contains raw and processed data for the project.

## Structure

```
data/
├── raw/              # Original, unprocessed data
│   ├── videos/      # Raw video files
│   └── images/      # Raw image files
├── processed/        # Processed/preprocessed data
│   ├── frames/      # Extracted video frames
│   └── annotations/ # Manual annotations (before YOLO format)
└── README.md        # This file
```

## Usage

- **Raw Data**: Place original videos and images here before processing
- **Processed Data**: Store intermediate processed data (frames, annotations) here
- **Final Datasets**: Processed datasets ready for training go in `../datasets/`

## Notes

- Large files should be stored outside the repository or in cloud storage
- Use `.gitignore` to exclude large data files from version control
- Document data sources and processing steps in this directory

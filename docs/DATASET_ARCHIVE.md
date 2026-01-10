# Dataset Archive Information

## Archived Datasets

The following datasets have been archived and moved to the `archives/` directory:

### Archived Datasets
- **Hooper-2** - Basketball detection dataset (ball, hoop, player)
  - Training: 84 images
  - Validation: 13 images
  - Source: [Roboflow Universe](https://universe.roboflow.com/basketball-dataset/hooper)

- **HooperAnalytic-1** - Basketball analytics dataset (ball, basket, player)
  - Training: 96 images
  - Validation: 15 images
  - Source: [Roboflow Universe](https://universe.roboflow.com/basketball-dataset/hooperanalytic)

## Archive Location

Archived datasets are stored in:
```
archives/datasets_archive_YYYYMMDD_HHMMSS.zip
```

## Restoring Archived Datasets

To restore a dataset from archive:

```bash
# Extract the archive
unzip archives/datasets_archive_YYYYMMDD_HHMMSS.zip

# Or on Windows PowerShell:
Expand-Archive -Path archives/datasets_archive_YYYYMMDD_HHMMSS.zip -DestinationPath .
```

## Starting Fresh

The repository is now ready for new datasets. When adding new datasets:

1. Place them in a new directory (e.g., `dataset-v2/`)
2. Follow the same structure:
   ```
   dataset-name/
   ├── data.yaml
   ├── train/
   │   ├── images/
   │   └── labels/
   └── valid/
       ├── images/
       └── labels/
   ```
3. Update `README.md` with new dataset information

## Archive Date

Datasets archived on: 2026-01-10

---

**Note**: Archived datasets are preserved but excluded from active development to start fresh.

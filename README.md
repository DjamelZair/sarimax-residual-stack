# SARIMAX Residual Stack

This repository contains the code and results for a SARIMAX residual stacking workflow.

## Data

The `data/` directory is **not included** in the repository to keep the repo lightweight and
within GitHub file size limits. To run the training locally:

1. Place your dataset files inside `data/`.
2. Run the training script:

```bash
python scripts/train.py
```

If you want to include data in version control, consider using Git LFS or an external
storage location and add download instructions here.

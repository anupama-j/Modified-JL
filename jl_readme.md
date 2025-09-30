# Dimensionality Reduction using Johnson-Lindenstrauss Lemma

**Authors:** Jawale A, Magar G  
**Year:** 2025

This project implements and evaluates various dimensionality reduction techniques for time series classification, with a focus on Johnson-Lindenstrauss (JL) projection methods.

## Overview

The code performs feature extraction from time series data and applies multiple dimensionality reduction techniques, comparing their classification performance using Support Vector Machines (SVM).

## Methods Implemented

1. **JL Lemma** - Standard Johnson-Lindenstrauss random projection
2. **Adaptive JL** - Dynamic JL projection with adaptive scaling
3. **Subspace Embedding** - PCA followed by JL projection
4. **PCA** - Principal Component Analysis
5. **UMAP** - Uniform Manifold Approximation and Projection

## Requirements

### R Version
- R 4.0.0 or higher recommended

### Required Packages

```r
install.packages(c(
  "FNN", "RcppRoll", "e1071", "caret", "pracma", 
  "RandPro", "umap", "dimRed", "keras", "dplyr", 
  "zoo", "Matrix", "caTools", "Rtsne", "ggplot2", 
  "cluster", "uwot"
))
```

## File Structure

- **`JL_Code.R`** - Main implementation with core functions and single dataset evaluation
- **`JL1_finl_new2_pca.R`** - Extended version with PCA and UMAP implementations
- **`executing_for_all_datasets.R`** - Batch processing script for multiple datasets (d1.csv through d7.csv)

## Usage

### Single Dataset Analysis

```r
# Load the script
source("JL_Code.R")

# Set your dataset path
mydf_raw <- read.csv("path/to/your/dataset.csv")

# Extract time series features
mydf <- as.data.frame(getTimeFeatures(mydf_raw))

# The script will automatically:
# 1. Apply all dimensionality reduction methods
# 2. Train SVM classifiers
# 3. Evaluate accuracy
# 4. Report runtime
```

### Multiple Dataset Analysis

```r
# Update dataset paths in executing_for_all_datasets.R
dataset_paths <- paste0("your/path/d", 1:7, ".csv")

# Run the batch script
source("executing_for_all_datasets.R")

# Results will be saved to:
# dimensionality_reduction_results.csv
```

## Input Data Format

The input CSV file should have:
- First column: Class labels
- Remaining columns: Time series observations
- Each row represents one time series instance

Example:
```
Class, t1, t2, t3, ..., tn
1, 0.5, 0.6, 0.4, ..., 0.7
2, 0.3, 0.2, 0.5, ..., 0.6
```

## Feature Extraction

The `getTimeFeatures()` function extracts rolling window statistics:
- Mean
- Median
- Standard deviation
- Variance
- Minimum/Maximum
- Energy
- Average power
- Root mean square (RMS)

**Window size:** 10 observations  
**Step size:** 2 (50% overlap)

## Dimensionality Reduction Parameters

Default target dimension: **100 features**

### Method-Specific Parameters

**JL Lemma:**
- Uses random Gaussian projection matrix
- Preserves distances approximately

**Adaptive JL:**
- Dynamic dimension calculation based on epsilon = 0.99
- k = ceil(log(n) / log(1/ε²))

**Subspace Embedding:**
- PCA dimension: 100
- JL dimension: 100 (applied after PCA)

**UMAP:**
- n_neighbors: 15
- min_dist: 0.1
- metric: Euclidean

## Evaluation Metrics

For each method, the following metrics are computed:

- **Accuracy** - Overall classification accuracy
- **Confidence Interval** - 95% CI for accuracy
- **P-Value** - Statistical significance
- **Kappa** - Cohen's Kappa statistic
- **Runtime** - Execution time in seconds

## Output

### Console Output
Displays a data frame with method comparison:

```
         Method Accuracy CI_Lower CI_Upper   P_Value    Kappa
1      Original   0.XXX    0.XXX    0.XXX      0.XXX    0.XXX
2      JL Lemma   0.XXX    0.XXX    0.XXX      0.XXX    0.XXX
3   Adaptive JL   0.XXX    0.XXX    0.XXX      0.XXX    0.XXX
4 Subspace Emb   0.XXX    0.XXX    0.XXX      0.XXX    0.XXX
5           PCA   0.XXX    0.XXX    0.XXX      0.XXX    0.XXX
6          UMAP   0.XXX    0.XXX    0.XXX      0.XXX    0.XXX
```

### Saved Output
When running the batch script, results are saved to:
- `dimensionality_reduction_results.csv`

Contains columns: Method, Accuracy, CI_Lower, CI_Upper, P_Value, Kappa, Dataset, Runtime

## Functions Reference

### Core Functions

- `getTimeFeatures(seriesset)` - Extract time series features
- `jl_lemma(data_matrix, target_dimension)` - Standard JL projection
- `adaptive_jl_incremental_reduction(data, min_dim, max_dim)` - Adaptive JL
- `subspace_embedding(data_matrix, pca_dimension, jl_dimension)` - PCA + JL
- `perform_pca_reduction(data, target_dim)` - PCA reduction
- `perform_umap_reduction(data, target_dim, n_neighbors, min_dist)` - UMAP reduction

### Utility Functions

- `fuzzifier(x, a, b, c)` - Triangular fuzzy membership function
- `replacena(xx)` - Handle missing values with moving average
- `generate_projection_matrix(n_samples, n_features, epsilon)` - Generate JL matrix
- `evaluate_reduced_data(name, reduced_df, original_data)` - Classification evaluation

## Classification Settings

- **Classifier:** Support Vector Machine (C-classification)
- **Train/Test Split:** 70/30
- **Seed:** 12342 (for reproducibility)

## Visualization (Optional)

The code includes commented functions for visualization:
- `plot_tsne()` - t-SNE 2D visualization
- `plot_pca()` - PCA 2D visualization

Uncomment these sections to generate plots for each method.

## Troubleshooting

### Common Issues

**Error: "Not enough features left for UMAP"**
- Solution: Reduce `target_dim` or check for constant columns in data

**Error: Missing package**
- Solution: Install all required packages listed above

**NA/Inf values in data**
- Solution: The `replacena()` function handles missing values automatically

**Memory issues with large datasets**
- Solution: Reduce `target_dim` or process datasets in smaller batches

## Performance Tips

1. **Parallel Processing:** Modify the loop in `executing_for_all_datasets.R` to use parallel computation
2. **Reduce Target Dimension:** Lower values (e.g., 50) for faster computation
3. **Sample Data:** For testing, use a subset of rows

## Citation

If you use this code in your research, please cite:

```
Jawale A, Magar G. (2025). Dimensionality Reduction using Johnson-Lindenstrauss Lemma 
for Time Series Classification.
```

## License

This code is provided as-is without any license restrictions.

## Contact

For questions or issues, please contact the authors: Jawale A, Magar G

## Acknowledgments

This implementation uses the Johnson-Lindenstrauss lemma for efficient dimensionality reduction while preserving pairwise distances in high-dimensional data.
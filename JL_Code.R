# ==============================================================================
# Title: Dimensionality Reduction using JL Lemma, Adaptive JL, and Subspace Embedding
# Author: [Your Name]
# Description: This script performs feature extraction and dimensionality reduction 
# using Johnson–Lindenstrauss (JL) projections, adaptive scaling, and PCA-based subspace 
# embeddings. It also evaluates classification performance and computes runtime.
# ==============================================================================

# ========================
# Load Required Libraries
# ========================
library(FNN)
library(RcppRoll)
library(e1071)
library(caret)
library(pracma)
library(RandPro)
library(dimRed)
library(keras)
library(dplyr)
library(zoo)
library(Matrix)
library(caTools)
library(Rtsne)
library(ggplot2)

# ========================
# Utility Functions
# ========================

# Fuzzification Function
fuzzifier <- function(x, a, b, c) {
  if (is.numeric(x)) {
    return(max(min((x - a)/(b - a), (c - x)/(c - b)), 0))
  } else if (is.data.frame(x)) {
    y <- x
    for (i in 1:nrow(x)) {
      for (j in 1:ncol(x)) {
        p <- x[i, j]
        q <- max(min((p - a)/(b - a), (c - p)/(c - b)), -5)
        y[i, j] <- q
      }
    }
    return(y)
  }
}

# Replace NA values with moving average or previous values
replacena <- function(xx) {
  for (i in 1:length(xx)) {
    if (is.na(xx[i])) {
      xx[i] <- ifelse(is.na(xx[i + 1]), xx[i - 1], mean(xx[(i - 3):(i - 1)], na.rm = TRUE))
    }
  }
  return(xx)
}

# Generate time series features using rolling windows
getTimeFeatures <- function(seriesset) {
  Timefeaturesmatrix <- NA
  
  for (i in 1:nrow(seriesset)) {
    classvar <- seriesset[i, 1]
    series <- as.numeric(seriesset[i, ])
    
    # Extract window-based features
    windowsize <- 10
    mn <- rollapplyr(series, windowsize, by = 2, mean)
    med <- rollapplyr(series, windowsize, by = 2, median)
    sdd <- rollapplyr(series, windowsize, by = 2, sd)
    vr <- rollapplyr(series, windowsize, by = 2, var)
    mnv <- rollapplyr(series, windowsize, by = 2, min)
    mx <- rollapplyr(series, windowsize, by = 2, max)
    energy <- rollapplyr(series^2, windowsize, by = 2, sum)
    avgp <- rollapplyr(series^2, windowsize, by = 2, mean)
    rms <- rollapplyr(avgp, 1, sqrt)
    
    # Combine into a row
    r <- c("Class" = classvar, mn, mx, energy, med, sdd, vr, mnv, avgp, rms)
    Timefeaturesmatrix <- rbind(Timefeaturesmatrix, r)
  }
  
  return(Timefeaturesmatrix[-1, ])  # Remove the initial NA row
}

# ========================
# Dimensionality Reduction Methods
# ========================

# Standard Johnson-Lindenstrauss projection
jl_lemma <- function(data_matrix, target_dimension) {
  n_features <- ncol(data_matrix)
  projection_matrix <- matrix(rnorm(n_features * target_dimension), nrow = n_features)
  projected_data <- as.matrix(data_matrix) %*% projection_matrix
  return(projected_data)
}

# Adaptive JL projection with dynamic scaling
adaptive_jl_incremental_reduction <- function(data, min_dim, max_dim) {
  num_samples <- nrow(data)
  num_features <- ncol(data)
  scaling_factor <- sqrt(num_features)
  target_dimension <- round(min_dim + (max_dim - min_dim))
  
  jl_projection_matrix <- generate_projection_matrix(num_samples, num_features)
  reduced_data <- as.matrix(data) %*% jl_projection_matrix
  return(reduced_data)
}

# PCA followed by JL projection
subspace_embedding <- function(data_matrix, pca_dimension, jl_dimension) {
  constant_columns <- apply(data_matrix, 2, function(x) length(unique(x)) == 1)
  data_matrix <- data_matrix[, !constant_columns]
  
  pca_result <- prcomp(data_matrix, center = TRUE, scale. = TRUE)
  pca_reduced_data <- pca_result$x[, 1:pca_dimension]
  
  jl_projection <- matrix(rnorm(pca_dimension * jl_dimension), nrow = pca_dimension)
  jl_projection <- jl_projection / sqrt(rowSums(jl_projection^2))
  
  final_reduced_data <- pca_reduced_data %*% jl_projection
  return(final_reduced_data)
}

# JL projection matrix generator with scaling
generate_projection_matrix <- function(n_samples, n_features, epsilon = 0.99) {
  k <- ceiling(log(n_samples) / log(1 / epsilon^2))
  projection_matrix <- matrix(rnorm(n_features * k), nrow = n_features, ncol = k)
  projection_matrix <- projection_matrix * sqrt(1 / k)
  return(projection_matrix)
}

# ========================
# Evaluation Functions
# ========================

# Classification Accuracy using SVM
evaluate_reduced_data <- function(name, reduced_df, original_data) {
  split <- sample.split(reduced_df$Class, SplitRatio = 0.70)
  train <- subset(reduced_df, split == TRUE)
  test <- subset(reduced_df, split == FALSE)
  
  model <- svm(Class ~ ., data = train, type = "C-classification")
  pred <- predict(model, test)
  acc <- mean(pred == test$Class)
  
  return(c(Method = name, Accuracy = acc, Stress = 0, Silhouette = 0))  # Optional metrics
}

# ========================
# Main Execution
# ========================

set.seed(12342)

# Load time series dataset and extract features
mydf_raw <- read.csv("Dataset/d7.csv")
mydf <- as.data.frame(getTimeFeatures(mydf_raw))

# Define target dimensions
target_dim <- 100
max_target_dimension <- 100
min_target_dimension <- 0

# Reduce with JL
cat("JL Lemma\n")
d_original <- data.frame(cbind(jl_lemma(mydf[,-1], target_dim), Class = mydf$Class))

# Reduce with Adaptive JL
cat("Adaptive JL\n")
d_adaptive <- data.frame(cbind(adaptive_jl_incremental_reduction(mydf[,-1], min_target_dimension, max_target_dimension), Class = mydf$Class))

# Reduce with Subspace Embedding
cat("Subspace Embedding\n")
d_scaled <- data.frame(cbind(subspace_embedding(mydf[,-1], 100, target_dim), Class = mydf$Class))

# Run evaluation
results <- list()
timings <- numeric(4)

timings[1] <- system.time({ results[[1]] <- evaluate_reduced_data("Original", mydf, mydf[,-1]) })["elapsed"]
timings[2] <- system.time({ results[[2]] <- evaluate_reduced_data("JL Lemma", d_original, mydf[,-1]) })["elapsed"]
timings[3] <- system.time({ results[[3]] <- evaluate_reduced_data("Adaptive JL", d_adaptive, mydf[,-1]) })["elapsed"]
timings[4] <- system.time({ results[[4]] <- evaluate_reduced_data("Subspace Embedding", d_scaled, mydf[,-1]) })["elapsed"]

# Display results
results_df <- as.data.frame(do.call(rbind, results))
runtime_df <- data.frame(Method = c("Original", "JL Lemma", "Adaptive JL", "Subspace Embedding"),
                         Time_sec = timings)

print(results_df)
print(runtime_df)

# ========================
# Optional Visualization
# ========================
# Define your plot_tsne() and plot_pca() functions separately to visualize each method.
# These are not shown here for brevity but should include ggplot2 plots of 2D representations.
# # t-SNE plots
plot_tsne(mydf,"Original")
plot_tsne(d_original, "JL Lemma")
plot_tsne(d_adaptive, "Adaptive Scaled")
plot_tsne(d_scaled, "Subspace Embedding")
# 
# # PCA plots
plot_pca(mydf,"Original")
plot_pca(d_original, "JL Lemma")
plot_pca(d_adaptive, "Adaptive Scaled")
plot_pca(d_scaled, "Subspace Embedding")
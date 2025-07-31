library(FNN)
library(RcppRoll)
library(e1071)
library(caret)
library(pracma)
library(RandPro)
library(umap)
library(dimRed)
library(keras)
library(dplyr)
library(zoo)
library(Matrix)
library(caTools)
library(Rtsne)
library(ggplot2)
library(cluster)

# ========== Existing helper functions go here ==========
# All your functions like: fuzzifier, getTimeFeatures, perform_pca_reduction, perform_umap_reduction, etc.
# Copy-paste all unchanged helper function definitions above this point

# ========== Evaluate all datasets in a loop ==========

dataset_paths <- paste0("D:\\_Anupama Research\\_Anupama Research\\Anupama Research Papers\\JL Paper\\Fuzzy\\d", 1:7, ".csv")

all_results <- list()
all_timings <- list()

for (i in 1:length(dataset_paths)) {
  
  cat("Processing Dataset:", dataset_paths[i], "\n")
  
  mydf1 <- read.csv(dataset_paths[i])
  mydf <- as.data.frame(getTimeFeatures(mydf1))
  mydf$Class <- as.factor(mydf$Class)
  
  # Define reductions
  target_dim <- 100
  max_target_dimension <- 100
  min_target_dimension <- 0
  
  # JL Lemma
  reduced_data <- jl_lemma(mydf[,-1], target_dim)
  d_jl <- data.frame(cbind(reduced_data, "Class" = mydf$Class))
  
  # Adaptive JL
  reduced_data <- adaptive_jl_incremental_reduction(mydf[,-1], min_target_dimension, max_target_dimension)
  d_adaptive <- data.frame(cbind(reduced_data, "Class" = mydf$Class))
  
  # Subspace Embedding
  reduced_data <- subspace_embedding(mydf[,-1], 100, target_dim)
  d_scaled <- data.frame(cbind(reduced_data, "Class" = mydf$Class))
  
  # PCA
  d_pca <- perform_pca_reduction(mydf, target_dim = 100)
  
  # UMAP
  d_umap <- perform_umap_reduction(mydf, target_dim = 100)
  
  # Evaluation
  results <- list()
  timings <- numeric(6)
  
  timings[1] <- system.time({
    results[[1]] <- evaluate_reduced_data("JL Lemma", d_jl, mydf[,-1])
  })["elapsed"]
  
  timings[2] <- system.time({
    results[[2]] <- evaluate_reduced_data("Adaptive JL", d_adaptive, mydf[,-1])
  })["elapsed"]
  
  timings[3] <- system.time({
    results[[3]] <- evaluate_reduced_data("Subspace Embedding", d_scaled, mydf[,-1])
  })["elapsed"]
  
  timings[4] <- system.time({
    results[[4]] <- evaluate_reduced_data("PCA", d_pca, mydf[,-1])
  })["elapsed"]
  
  timings[5] <- system.time({
    results[[5]] <- evaluate_reduced_data("UMAP", d_umap, mydf[,-1])
  })["elapsed"]
  
  timings[6] <- system.time({
    results[[6]] <- evaluate_reduced_data("Original", mydf, mydf[,-1])
  })["elapsed"]
  
  results_df <- as.data.frame(do.call(rbind, results))
  results_df$Dataset <- paste0("d", i)
  results_df$Runtime <- timings
  
  all_results[[i]] <- results_df
  all_timings[[i]] <- timings
}

# =================== Combine All Results ===================

final_results <- do.call(rbind, all_results)
print(final_results)

# Optionally: save to CSV
write.csv(final_results, "dimensionality_reduction_results.csv", row.names = FALSE)

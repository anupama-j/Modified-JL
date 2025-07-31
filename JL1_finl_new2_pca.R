library(FNN)
library(RcppRoll)
library(e1071)
library(caret)
library(pracma)
library(RandPro)
# Function to perform incremental Johnson-Lindenstrauss dimensionality reduction
# Function to perform incremental Johnson-Lindenstrauss dimensionality reduction
# Required packages: keras, dplyr
library(umap)
library(dimRed)
library(keras)
library(dplyr)
library(zoo)
library(Matrix)
library(caTools)

library(Rtsne)
library(ggplot2)

# Load necessary libraries

fuzzifier<-function(x,a,b,c)
 {
  
  if(is.numeric(x))
    return (max( min( (x-a)/(b-a), (c-x)/(c-b) ), 0 )) 
  else
    if(is.data.frame(x))
    {
      y=x
      for(i in c((1:nrow(x)))){ 
        for(j in c((1:ncol(x))))
        {
          p=x[i,j]
          q=max(min((p-a)/(b-a), (c-p)/(c-b) ), -5 )
          y[i,j]=q
        }
      }
      return(y)
      
    }
}



# ========================= PCA and UMAP =============================

perform_pca_reduction <- function(data,target_dim = 100) {
  
  features <- data[, -which(names(data) == "Class")]
  labels <- data$Class
  
  # Remove constant columns (zero variance)
  constant_cols <- sapply(features, function(col) var(col, na.rm = TRUE) == 0)
  features <- features[, !constant_cols]
  pca_result <- prcomp(features, center = TRUE, scale. = TRUE)
  reduced_data <- as.data.frame(pca_result$x[, 1:target_dim])
  reduced_data$Class <- labels
  return(reduced_data)
}


perform_umap_reduction <- function(data, target_dim = 100, n_neighbors = 15, min_dist = 0.1) {
  features <- data[, -which(names(data) == "Class")]
  labels <- data$Class
  constant_cols <- apply(features, 2, function(x) length(unique(x)) <= 1)
  features <- features[, !constant_cols]
  
  if (ncol(features) < target_dim) {
    stop("Not enough features left for UMAP reduction after removing constant columns.")
  }
  
  
  config <- umap::umap.defaults
  config$n_neighbors <- n_neighbors
  config$min_dist <- min_dist
  config$n_components <- target_dim
  cat("Checking for NA, Inf, NaN...\n")


  #umap_result <- umap::umap(features, config = config)
  
  umap_result <- uwot::umap(
    features,
    n_neighbors = n_neighbors,
    min_dist = min_dist,
    n_components = target_dim,
    metric = "euclidean",
    verbose = TRUE
  )
  
  
  reduced_data <- as.data.frame(umap_result)
  colnames(reduced_data) <- paste0("UMAP_", 1:target_dim)
  reduced_data$Class <- labels
  return(reduced_data)
}



subspace_embedding <-function(data_matrix,pca_dimension,jl_dimension) { 
 # if (is.null(jl_projection)) {
    jl_projection <- matrix(rnorm(pca_dimension * jl_dimension), nrow = pca_dimension, ncol = jl_dimension)
    jl_projection <- jl_projection / sqrt(rowSums(jl_projection^2))
 # }
  
  # Perform PCA
    constant_columns <- apply(data_matrix, 2, function(x) length(unique(x)) == 1)
    data_matrix <- data_matrix[, !constant_columns]
    
    # Check for columns with zero variance
   # constant_columns <- apply(data_matrix, 2, var) == 0
    
    # Remove constant columns
#   data_matrix <- data_matrix[, !constant_columns]
    
  pca_result <- prcomp(data_matrix, center = TRUE, scale = TRUE)
  
  
  # Extract the first 'pca_dimension' principal components
  pca_reduced_data <- pca_result$x[, 1:pca_dimension]
  
  # Project PCA-reduced data onto lower-dimensional space using JL projection
  final_reduced_data <- pca_reduced_data %*% jl_projection
  
  return(final_reduced_data)

}




replacena <- function(xx) {
  for(i in 1:length(xx)) if(is.na(xx[i]))
    xx[i] = ifelse(is.na(xx[i+1]), xx[i-1], (xx[i-1]+xx[i-2]+xx[i-3])/3)
  return(xx)
}

getTimeFeatures <-function(seriesset)
{
  framer=c()
  Timefeaturesmatrix=NA
  for(i in c(1:nrow(seriesset)))
  { 
    classvar=seriesset[i,1]  
    
    series=seriesset[i,]
    y=as.numeric(series)
    means=c()
    sds=c()
    maxs=c()
    mins=c()
    peaks=c()
    windowsize=10
    
    mn=rollapplyr(y, windowsize,by=2, mean)
    #md=mode(y)
    med=rollapplyr(y,windowsize , by=2,median)
    sdd=rollapplyr(y,windowsize , by=2,sd)
    vr=rollapplyr(y,windowsize , by=2,stats::var)
    mnv=rollapplyr(y,windowsize , by=2,min)
    mx=rollapplyr(y,windowsize , by=2,max)
    energy=rollapplyr(y*y,windowsize ,by=2, sum)
    avgp=rollapplyr(y*y,windowsize , by=2,mean)
    rms=rollapplyr(avgp,1 , by=2,sqrt)
    
    
    # dff=rollapplyr(df,2,diff)
    
    
    #  Timefeaturesmatrix=(rbind(mn,med,sdd,vr,mn,mx,energy,avgp,rms))
    # r= c("Class"=classvar,mn,med,sdd,vr,mx,mnv,energy, avgp, rms)
    r= c("Class"=classvar,mn,mx,energy,med,sdd,vr,mnv,avgp,rms)
    
    
    Timefeaturesmatrix=rbind(Timefeaturesmatrix,r)
  }
  return(Timefeaturesmatrix[-1,])
}


jl_lemma<-function(data_matrix,target_dimension){
  n_samples <- nrow(data_matrix)   # Number of data points
  n_features <- ncol(data_matrix)
  
  projection_matrix <- matrix(rnorm(n_features * target_dimension), nrow = n_features)
  
  # Apply the JL projection
  projected_data <- as.matrix(data_matrix) %*% projection_matrix
  return(projected_data)
}



adaptive_jl_incremental_reduction <- function(data, min_dim, max_dim) {
  
  num_samples <- nrow(data)
  num_features <- ncol(data)
  
  reduced_data_list <- list()  # Initialize a list to store reduced data
  
  
    scaling_factor <- sqrt(num_features)  # Scaling factor for JL projection
    
    target_dimension <- round(min_dim + (max_dim - min_dim) )
    
    # Generate JL projection matrix with scaling
   # jl_projection_matrix <- generate_jl_projection_matrix(num_features, target_dimension, scaling_factor)
    
    jl_projection_matrix=generate_projection_matrix(num_samples,num_features)
    
    # Normalize the JL projection matrix
    #normalized_jl_projection_matrix <- normalize_projection_matrix(jl_projection_matrix)
    
    # Project current data point using JL projection
    reduced_data <- as.matrix(data) %*% jl_projection_matrix
  
  return(reduced_data)
}


generate_projection_matrix <- function(n_samples, n_features, epsilon = .99) {
  if (epsilon <= 0 || epsilon >= 1) {
    stop("Epsilon must be in the range (0, 1)")
  }
  
  # Compute the required number of random features
  k <- ceiling(log(n_samples) / log(1 / epsilon^2))
  
  # Generate the projection matrix
  projection_matrix <- matrix(rnorm(n_features * k), nrow = n_features, ncol = k)
  
  # Scale the projection matrix to preserve distances
  projection_matrix <- projection_matrix * sqrt(1 / k)
  
  return(projection_matrix)
}


# Function to generate JL projection matrix
generate_jl_projection_matrix <- function(original_dimension, target_dimension,scaling) {
  projection_matrix <- matrix(rnorm(original_dimension * target_dim) * scaling, nrow = original_dimension, ncol = target_dim)
  
  # Apply the projection matrix to the data
 # p_data <- as.matrix(data) %*% projection_matrix
  return(projection_matrix)
}

# Function to normalize the projection matrix
normalize_projection_matrix <- function(matrix) {
  normalized_matrix <- matrix / sqrt(rowSums(matrix^2))
  return(normalized_matrix)
}


#--------------------------------------------------------









#===========================================================================================


set.seed(12342)

mydf1<-read.csv("D:\\_Anupama Research\\_Anupama Research\\Anupama Research Papers\\JL Paper\\Fuzzy\\d6.csv" )

mydf=as.data.frame(getTimeFeatures( mydf1))
dim(mydf)


# Set the target dimension for dimensionality reduction
target_dim <- 100
max_target_dimension=100
min_target_dimension=0


# #--------------------------------------------------------------------------
print("JL Lemma")
reduced_data <-jl_lemma(mydf[,-1],target_dim)
reduced_df<-data.frame(cbind(reduced_data, "Class"=mydf$Class))

d_original=reduced_df



#-------------------------------------------------------------------------------
print("Adaptive")
reduced_data <- adaptive_jl_incremental_reduction(mydf[,-1], min_target_dimension, max_target_dimension)
reduced_df<-data.frame(cbind(reduced_data, "Class"=mydf$Class))

d_adaptive=reduced_df


#--------------------------------------------------------------------------
print("Subspace Embedd")
reduced_data=subspace_embedding((mydf[,-1]), 100,(target_dim))
reduced_df<-data.frame(cbind(reduced_data, "Class"=mydf$Class))

d_scaled=reduced_df


#--------------------------------------------------------------------------
# ---------------- PCA -------------------------------------
print("PCA Reduction")
d_pca <- perform_pca_reduction(mydf, target_dim = 100)

# ---------------- UMAP -------------------------------------
print("UMAP Reduction")
d_umap <- perform_umap_reduction(mydf, target_dim = 100)

#--------------------------------------------------------------------------


perform_pca_reduction <- function(data, target_dim = 2) {
  # Expects data with a column named 'Class'
  features <- data[, -which(names(data) == "Class")]
  labels <- data$Class
  
  # Apply PCA with centering and scaling
  pca_result <- prcomp(features, center = TRUE, scale. = TRUE)
  
  # Extract top principal components
  reduced_data <- as.data.frame(pca_result$x[, 1:target_dim])
  reduced_data$Class <- labels
  
  return(reduced_data)
}




#--------------------------------------------------------------------------------------


# Evaluation functions (assumes compute_stress is already defined)
library(cluster)
library(e1071)
library(caTools)

evaluate_reduced_data <- function(name, reduced_df, original_data) {
  # Classification Accuracy
  split = sample.split(reduced_df$Class, SplitRatio = 0.70)
  train = subset(reduced_df, split == TRUE)
  test = subset(reduced_df, split == FALSE)
  
  model = svm(Class ~ ., data = train, type = "C-classification")
  pred = predict(model, test)
  acc = mean(pred == test$Class)
  levels_union <- union(levels(factor(pred)), levels(factor(test$Class)))
  
  pred <- factor(pred, levels = levels_union)
  truth <- factor(test$Class, levels = levels_union)
  
  
  conf_mat <- confusionMatrix(pred, truth)
  
  # Print the full summary
  acc <- conf_mat$overall['Accuracy']
  kappa <- conf_mat$overall['Kappa']
  pval <- if ("AccuracyPValue" %in% names(conf_mat$overall)) conf_mat$overall['AccuracyPValue'] else NA
  ci_low <- if ("AccuracyLower" %in% names(conf_mat$overall)) conf_mat$overall['AccuracyLower'] else NA
  ci_up  <- if ("AccuracyUpper" %in% names(conf_mat$overall)) conf_mat$overall['AccuracyUpper'] else NA
  
  return(c(Method = name, Accuracy = acc,
           CI_Lower = ci_low, CI_Upper = ci_up,
           P_Value = pval, Kappa = kappa
           ))
  
}

# Evaluate each dataset
results <- list()
timings <- numeric(3)


timings[1] <- system.time({
  results[[1]] <- evaluate_reduced_data("Original", mydf, mydf[,-1])
})["elapsed"]


# JL Lemma
timings[2] <- system.time({
  results[[1]] <- evaluate_reduced_data("JL Lemma",d_original, mydf[,-1])
})["elapsed"]

# Adaptive JL
timings[3] <- system.time({
  results[[2]] <- evaluate_reduced_data("Adaptive JL", d_adaptive, mydf[,-1])
})["elapsed"]

# Subspace Embedding
timings[4] <- system.time({
  results[[3]] <- evaluate_reduced_data("Subspace Embedding", d_scaled, mydf[,-1])
})["elapsed"]


# Extend results and timing to include PCA and UMAP
timings[5] <- system.time({
  results[[4]] <- evaluate_reduced_data("PCA", d_pca, mydf[,-1])
})["elapsed"]

timings[6] <- system.time({
  results[[5]] <- evaluate_reduced_data("UMAP", d_umap, mydf[,-1])
})["elapsed"]



# Show runtime in seconds

method_names <- c( "PCA", "UMAP")

#method_names <- c("Original", "JL Lemma", "Adaptive JL", "Subspace Embedding", "PCA", "UMAP")
data.frame(Method = method_names, Time_sec = timings)

# Convert to data frame for tabular summary
results_df <- as.data.frame(do.call(rbind, results))
print(results_df)



# 
# # t-SNE plots
# plot_tsne(mydf,"Original")
#  plot_tsne(d_original, "JL Lemma")
#  plot_tsne(d_adaptive, "Adaptive Scaled")
#  plot_tsne(d_scaled, "Subspace Embedding")
# # 
# # # PCA plots
#  plot_pca(mydf,"Original")
#  plot_pca(d_original, "JL Lemma")
#  plot_pca(d_adaptive, "Adaptive Scaled")
#  plot_pca(d_scaled, "Subspace Embedding")






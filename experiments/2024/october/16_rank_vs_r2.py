"""
Experiment Setup:
- \( X \): Random feature matrix with varying rank.
- \( Y \): Clean target matrix generated by a low-rank structure (used for evaluation).
- \( Y' \): Noisy version of \( Y \), used for training the linear regression (LR) model.

Prediction:
As the rank of \( X \) increases, the \( R^2 \) score when predicting \( Y' \) (noisy data) will increase, indicating a better fit.

Reason for the Experiment:
This setup offers a more intuitive approach to understanding how the rank of feature matrices affects model performance, serving as a simpler proxy to the concept of effective dimensionality in models.

Comment:
This experiment supports the hypothesis that increasing the rank of the feature matrix improves R62, even when fitting on noisy data and testing on clean data. 
As the rank (or effective dimensionality) of the feature matrix increases, the model captures more structure, leading to better fits. 
This aligns with findings from the paper, which show that models with higher effective dimensionality yield better predictions, 
even if they may not perfectly represent the underlying system. Both cases demonstrate that richer representations improve performance, 
though they may introduce biases from the underlying regression methodology​

ref: https://chatgpt.com/c/67106106-5ba0-8001-90fd-131e34da0af0
"""
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

# Function to generate random data with a specific rank efficiently
def generate_data_with_rank_efficient(n_samples, n_features, rank):
    # Generate two random matrices and compute their product to control the rank
    X = np.random.randn(n_samples, rank) @ np.random.randn(rank, n_features)
    return X

# Function to generate structured target data Y and noisy Y' for training
def generate_structured_Y_Y_prime(n_samples, n_features, rank, noise_level=0.1):
    # Generate low-rank matrices
    A = np.random.randn(n_samples, rank)
    B = np.random.randn(rank, n_features)
    
    # Y is the clean structured target matrix
    Y = A @ B
    
    # Y' is the noisy version of Y (used for fitting)
    Y_prime = Y + noise_level * np.random.randn(n_samples, n_features)
    
    return Y, Y_prime

# Parameters
n_samples = 50
n_features = n_samples # Increase number of features and ranks
rank_values = range(1, n_features + 1)
n_trials = 20  # Number of trials for averaging to smooth out noise
noise_level = 0.01  # Reduce noise for clearer signal

# Store smoothed R^2 values and confidence intervals for each rank
r2_values_all_trials = np.zeros((n_trials, len(rank_values)))

# Conduct multiple trials and store the R^2 values
for trial in range(n_trials):
    # Generate structured target matrices
    Y_clean, Y_noisy = generate_structured_Y_Y_prime(n_samples, 1, 3, noise_level)  # Rank 3 structure
    
    for i, rank in enumerate(rank_values):
        X = generate_data_with_rank_efficient(n_samples, n_features, rank)
        model = LinearRegression()
        model.fit(X, Y_noisy)
        Y_pred = model.predict(X)
        r2 = r2_score(Y_clean, Y_pred)  # Evaluating on Y (clean structured data)
        r2_values_all_trials[trial, i] = r2

# Calculate mean and 0.95 confidence intervals for each rank
r2_means = np.mean(r2_values_all_trials, axis=0)
r2_stds = np.std(r2_values_all_trials, axis=0)
r2_cis = 1.96 * r2_stds / np.sqrt(n_trials)  # 0.95 confidence interval

# Plotting the smoothed results with 0.95 confidence intervals
plt.figure(figsize=(8, 6))
plt.plot(rank_values, r2_means, marker='o', label='Mean R^2')
plt.fill_between(rank_values, r2_means - r2_cis, r2_means + r2_cis, color='b', alpha=0.2, label='95% CI')
plt.title('Smoothed R^2 vs Rank of Feature Matrix (Averaged Over Trials)')
plt.xlabel('Rank of X')
plt.ylabel('R^2')
plt.legend()
plt.grid(True)
plt.savefig('r2_vs_rank_plot.png')
plt.show()
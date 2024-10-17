# https://chatgpt.com/c/6710723d-5a28-8001-a00d-db3ab790efb2
# https://discord.com/channels/@me/1063613968087797830/1296292231527010367
# Running the code to show the updated plot

# Full code with imports, confidence intervals, and tighter noise settings

# Import necessary libraries
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

# Function to generate Gaussian data with specified variances (diagonal covariance matrix)
def generate_gaussian_data_with_cov(n_samples, variances):
    """Generate Gaussian data with a diagonal covariance matrix defined by the variances."""
    return np.random.randn(n_samples, len(variances)) * np.sqrt(variances)

# Function to compute effective dimensionality based on eigenvalues of covariance matrix
def compute_effective_dimensionality(eigenvalues):
    """Compute the effective dimensionality using the participation ratio."""
    return (np.sum(eigenvalues) ** 2) / np.sum(eigenvalues ** 2)

# Function to generate structured target data Y with specified variances (using spectral theory)
def generate_spectral_target_with_cov(n_samples, variances):
    """Generate target data Y with covariance matrix based on the given variances."""
    return generate_gaussian_data_with_cov(n_samples, variances)

# Parameters
n_samples = 50  # Number of samples
n_features = 200  # Number of features
n_trials = 10  # Increase number of trials to tighten confidence intervals
noise_level = 1.0  # Reduce noise level to minimize variation

# Generate a list of variances to control the effective dimensionality
variance_sets = []
for i in range(1, n_features + 1):
    variances = [1] * i + [1e-8] * (n_features - i)  # Controlled variances, starting with large variance
    variance_sets.append(variances)

# Store results for effective dimensionality and R^2 scores
effective_dims_all_trials = []
r2_values_all_trials = []

# Conduct multiple trials and store results
for trial in range(n_trials):
    trial_effective_dims = []
    trial_r2_values = []
    
    for variances in variance_sets:
        # Generate feature matrix X with controlled variances
        X = generate_gaussian_data_with_cov(n_samples, variances)
        
        # Generate target matrix Y with the same varying effective dimensionality
        Y_clean = generate_spectral_target_with_cov(n_samples, variances)
        Y_noisy = Y_clean + noise_level * np.random.randn(n_samples, n_features)
        
        # Compute covariance and eigenvalues of X
        covariance_matrix = np.cov(X, rowvar=False)
        eigenvalues = np.linalg.eigvalsh(covariance_matrix)
        
        # Compute effective dimensionality
        ed = compute_effective_dimensionality(eigenvalues)
        trial_effective_dims.append(ed)
        
        # Fit linear regression model
        model = LinearRegression()
        model.fit(X, Y_noisy)
        Y_pred = model.predict(X)
        
        # Compute R^2
        r2 = r2_score(Y_clean, Y_pred)
        trial_r2_values.append(r2)
    
    effective_dims_all_trials.append(trial_effective_dims)
    r2_values_all_trials.append(trial_r2_values)

# Convert results to arrays
effective_dims_all_trials = np.array(effective_dims_all_trials)
r2_values_all_trials = np.array(r2_values_all_trials)

# Compute mean values and confidence intervals for effective dimensionality and R^2
effective_dim_means = np.mean(effective_dims_all_trials, axis=0)
r2_means = np.mean(r2_values_all_trials, axis=0)
r2_stds = np.std(r2_values_all_trials, axis=0)
r2_cis = 1.96 * r2_stds / np.sqrt(n_trials)  # 95% confidence interval

# Plot the results with confidence intervals
plt.figure(figsize=(8, 6))
plt.plot(effective_dim_means, r2_means, marker='o', label='Mean R^2')
plt.fill_between(effective_dim_means, r2_means - r2_cis, r2_means + r2_cis, color='b', alpha=0.2, label='95% CI')
plt.title('Effect of Effective Dimensionality on R^2 Score with 95% CI (Averaged Over Trials)')
plt.xlabel('Effective Dimensionality (ED)')
plt.ylabel('R^2 Score')
plt.legend()
plt.grid(True)
plt.savefig('r2_vs_ed_plot.png')
plt.show()

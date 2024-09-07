"""
ref: https://chatgpt.com/c/66dca9ee-1920-8001-aceb-ee02a23cea84
This script detects quadratic and polynomial correlations between X and Y.
It uses Canonical Correlation Analysis (CCA) and generalized correlation methods
to explore relationships like Y = X^2 + X + 1 and evaluate other non-linear forms.
"""
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.cross_decomposition import CCA
from scipy.interpolate import UnivariateSpline

# 1. Create Gaussian distributed data for X
np.random.seed(42)
X = np.random.normal(loc=0, scale=1, size=1000)

# 2. Define Y to have a quadratic relationship with X
Y_quadratic = X**2 + np.random.normal(0, 0.1, size=X.shape)

# 3. Quadratic correlation function
def quadratic_correlation(X, Y):
    X_centered_sq = (X - np.mean(X))**2
    Y_centered = Y - np.mean(Y)
    numerator = np.mean(X_centered_sq * Y_centered)
    denominator = np.std(X_centered_sq) * np.std(Y_centered)
    return numerator / denominator

# 4. Polynomial basis expansion and Canonical Correlation Analysis (CCA)
# Define Y = X^2 + X + 1
Y_poly = X**2 + X + 1 + np.random.normal(0, 0.1, size=X.shape)

# Expand X into polynomial features up to degree 10
poly = PolynomialFeatures(degree=10)
X_poly = poly.fit_transform(X.reshape(-1, 1))  # Polynomial basis for X

# Apply Canonical Correlation Analysis (CCA)
cca = CCA(n_components=1)  # Looking for the first canonical component
cca.fit(X_poly, Y_poly.reshape(-1, 1))  # Fit CCA on polynomial features and Y

# Get the canonical correlations and transform both X_poly and Y_poly
X_c, Y_c = cca.transform(X_poly, Y_poly.reshape(-1, 1))

# 5. Plot the canonical correlations for the polynomial basis vs Y
plt.figure(figsize=(10, 6))
plt.scatter(X_c, Y_c, alpha=0.5, label='Canonical Correlation')
plt.xlabel('Canonical Component of X')
plt.ylabel('Y')
plt.title('Canonical Correlation Analysis (CCA) for Polynomial Features')
plt.legend()
plt.show()

# Print canonical correlation value
canonical_corr = np.corrcoef(X_c[:, 0], Y_c[:, 0])[0, 1]
print(f"Canonical Correlation: {canonical_corr:.3f}")

# 6. Additional functional forms: logarithmic and sinusoidal
def logarithmic(X):
    return np.log(np.abs(X) + 1)  # Avoid log(0)

def sinusoidal(X):
    return np.sin(X)

# Compute generalized correlation for these forms
def generalized_correlation(f, X, Y):
    f_X = f(X)
    numerator = np.mean(f_X * Y) - np.mean(f_X) * np.mean(Y)
    denominator = np.std(f_X) * np.std(Y)
    return numerator / denominator

functions = [logarithmic, sinusoidal]
function_names = ["Logarithmic", "Sinusoidal"]

generalized_correlations = []
for func, name in zip(functions, function_names):
    gen_corr = generalized_correlation(func, X, Y_quadratic)
    generalized_correlations.append((name, gen_corr))

# Plot results for new functional forms
plt.figure(figsize=(8, 6))
sns.barplot(x=[name for name, _ in generalized_correlations], y=[corr for _, corr in generalized_correlations])
plt.title('Generalized Correlation for Different Functional Forms')
plt.xlabel('Function')
plt.ylabel('Generalized Correlation')
plt.show()

# Print results
for name, corr in generalized_correlations:
    print(f"{name}: {corr:.3f}")

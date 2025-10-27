# Step 1: Install necessary libraries if not already installed
# pip install pandas openpyxl scipy matplotlib seaborn

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats

# Step 2: Load the Excel file
# Adjust the file path to your Excel file
file_path = 'C:/Users/Josh Reynolds/Documents/Indpt Study - Prob Theory/Paper/excel files/Functions/Sin Function/RFS_Data_n_5000_a1.xlsx'
# Read the specific sheet or columns, specify columns you want to load by column name
df = pd.read_excel(file_path, usecols=["Trial Results"])  # Replace '' with your actual column name

# Inspect the first few rows to ensure it is loaded correctly
print(df.head())

# Assuming you are working with a column named '' (adjust based on your data)
data_values = df['Trial Results']  # Replace '' with the actual column name containing the data

# Step 3: Visualize the Data
plt.figure(figsize=(8, 6))
sns.histplot(data_values, kde=True, bins=60, color='blue', stat='density')
plt.title('Histogram of Data with KDE')
plt.xlabel('Value')
plt.ylabel('Density')
plt.show()

# Step 4: Fit the Data to a Distribution (Normal Distribution Example)
param = stats.norm.fit(data_values)  # Fit a normal distribution

# Plot the histogram and the fitted normal distribution
plt.figure(figsize=(8, 6))
sns.histplot(data_values, kde=False, bins=60, color='blue', stat='density')

# Get the x-values for plotting the normal pdf
xmin, xmax = plt.xlim()
x = np.linspace(xmin, xmax, 100)

# Plot the PDF of the fitted normal distribution
p = stats.norm.pdf(x, *param)
plt.plot(x, p, 'k', linewidth=2)

plt.title('Histogram with Fitted Normal Distribution')
plt.xlabel('Value')
plt.ylabel('Density')
plt.show()

# Step 5: Perform a goodness-of-fit test (Kolmogorov-Smirnov Test)
D, p_value = stats.kstest(data_values, 'norm', args=param)
print(f"KS Test result: D={D}, p-value={p_value}")

# Step 6: Try fitting a Lognormal Distribution
param_lognorm = stats.lognorm.fit(data_values, floc=0)  # Lognormal fit, fixing location to 0

# Plot the histogram and the fitted lognormal distribution
plt.figure(figsize=(8, 6))
sns.histplot(data_values, kde=False, bins=60, color='blue', stat='density')

# Plot the PDF of the fitted lognormal distribution
p_lognorm = stats.lognorm.pdf(x, *param_lognorm)
plt.plot(x, p_lognorm, 'r', linewidth=2)

plt.title('Histogram with Fitted Lognormal Distribution')
plt.xlabel('Value')
plt.ylabel('Density')
plt.show()

# Step 7: Perform goodness-of-fit test for the lognormal distribution
D_lognorm, p_value_lognorm = stats.kstest(data_values, 'lognorm', args=param_lognorm)


# Step 8: (Optional) Test other distributions like Exponential, Gamma, etc.
param_exp = stats.expon.fit(data_values)  # Fit exponential distribution

# Plot the histogram and the fitted exponential distribution
plt.figure(figsize=(8, 6))
sns.histplot(data_values, kde=False, bins=60, color='blue', stat='density')

# Plot the PDF of the fitted exponential distribution
p_exp = stats.expon.pdf(x, *param_exp)
plt.plot(x, p_exp, 'g', linewidth=2)

plt.title('Histogram with Fitted Exponential Distribution')
plt.xlabel('Value')
plt.ylabel('Density')
plt.show()

# Step 9: Perform goodness-of-fit test for the exponential distribution
D_exp, p_value_exp = stats.kstest(data_values, 'expon', args=param_exp)


# --- Step 10: Fit Gamma Distribution ---
param_gamma = stats.gamma.fit(data_values)  # Returns (shape, loc, scale)

# Plot the histogram with fitted Gamma distribution
plt.figure(figsize=(8,6))
sns.histplot(data_values, kde=False, bins=60, color='blue', stat='density')
x = np.linspace(min(data_values), max(data_values), 100)
p_gamma = stats.gamma.pdf(x, *param_gamma)
plt.plot(x, p_gamma, 'm', linewidth=2)
plt.title('Histogram with Fitted Gamma Distribution')
plt.xlabel('Value')
plt.ylabel('Density')
plt.show()

# KS Test for Gamma
D_gamma, p_value_gamma = stats.kstest(data_values, 'gamma', args=param_gamma)


# --- Step 11: Fit Beta Distribution ---
# First normalize the data to [0,1]
data_norm = (data_values - min(data_values)) / (max(data_values) - min(data_values))
param_beta = stats.beta.fit(data_norm)

# Plot histogram with Beta fit
plt.figure(figsize=(8,6))
sns.histplot(data_norm, kde=False, bins=60, color='blue', stat='density')
x = np.linspace(0, 1, 100)
p_beta = stats.beta.pdf(x, *param_beta)
plt.plot(x, p_beta, 'c', linewidth=2)
plt.title('Histogram with Fitted Beta Distribution (normalized data)')
plt.xlabel('Normalized Value')
plt.ylabel('Density')
plt.show()

# KS Test for Beta
D_beta, p_value_beta = stats.kstest(data_norm, 'beta', args=param_beta)



# Step 12: Conclusion - You can now interpret the p-values from the KS tests
# A high p-value (greater than 0.05) suggests a good fit for the distribution.

# Step 12: Conclusion and Comparison
print(f"KS Test for Normal: D={D}, p-value={p_value}")
print(f"KS Test for Lognormal: D={D_lognorm}, p-value={p_value_lognorm}")
print(f"KS Test for Exponential: D={D_exp}, p-value={p_value_exp}")
print(f"KS Test for Gamma: D={D_gamma}, p-value={p_value_gamma}")
print(f"KS Test for Beta: D={D_beta}, p-value={p_value_beta}")

# Determine which distribution is the best fit
if p_value > 0.05:
    print("Normal distribution is a good fit.")
elif p_value_lognorm > 0.05:
    print("Lognormal distribution is a good fit.")
elif p_value_exp > 0.05:
    print("Exponential distribution is a good fit.")
elif p_value_gamma > .05:
    print("Gamma distribution is a good fit")
elif p_value_beta > .05:
    print("Beta distribution is a good fit")
else:
    print("None of the distributions fit well.")



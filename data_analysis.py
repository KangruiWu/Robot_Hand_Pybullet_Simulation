#!/usr/bin/env python
# coding: utf-8

# In[33]:


import csv
import matplotlib.pyplot as plt


# In[34]:


import pandas as pd
import scipy.stats as stats
import numpy as np


# In[96]:


# Specify the paths to the PF and KF CSV files
pf_csv_file_path = r'C:\Users\foser\Downloads\robot_hand\data\pf_mse_data.csv'
kf_csv_file_path = r'C:\Users\foser\Downloads\robot_hand\data\kf_mse_data.csv'

# Read the PF and KF CSV files using pandas
pf_df = pd.read_csv(pf_csv_file_path)
kf_df = pd.read_csv(kf_csv_file_path)

# Ensure that the PF and KF data have the same number of values
min_length = min(len(pf_df), len(kf_df))
pf_df = pf_df[:min_length]
kf_df = kf_df[:min_length]

# Basic statistics for PF and KF
pf_mean_mse = pf_df['MSE'].mean()
pf_median_mse = pf_df['MSE'].median()
pf_std_mse = pf_df['MSE'].std()

kf_mean_mse = kf_df['MSE'].mean()
kf_median_mse = kf_df['MSE'].median()
kf_std_mse = kf_df['MSE'].std()

# Print the basic statistics for PF and KF
print("PF Mean MSE:", pf_mean_mse)
print("PF Median MSE:", pf_median_mse)
print("PF Standard Deviation of MSE:", pf_std_mse)

print("KF Mean MSE:", kf_mean_mse)
print("KF Median MSE:", kf_median_mse)
print("KF Standard Deviation of MSE:", kf_std_mse)

# Box Plots
pf_df['Filter'] = 'PF'
kf_df['Filter'] = 'KF'
combined_data = pd.concat([pf_df, kf_df])
combined_data.boxplot(column='MSE', by='Filter')
plt.xlabel('Filter')
plt.ylabel('MSE')
plt.show()

# Plotting the MSE values for PF and KF
plt.figure(figsize=(12, 6))
plt.plot(pf_df['Iteration'], pf_df['MSE'], label='PF')
plt.plot(kf_df['Iteration'], kf_df['MSE'], label='KF')
plt.xlabel('Iteration')
plt.ylabel('MSE')
plt.title('MSE - PF vs KF')
plt.legend()
plt.show()

# Additional data analysis with pandas
# Here are a few examples:

# Calculate rolling mean of MSE with a window size of 10 for PF and KF
pf_df['PF RollingMean'] = pf_df['MSE'].rolling(window=10).mean()
kf_df['KF RollingMean'] = kf_df['MSE'].rolling(window=10).mean()

# Filter MSE values above a threshold for PF and KF
pf_threshold = 0.1
kf_threshold = 0.15
pf_filtered_df = pf_df[pf_df['MSE'] > pf_threshold]
kf_filtered_df = kf_df[kf_df['MSE'] > kf_threshold]

# Count the number of MSE values above the threshold for PF and KF
pf_count_above_threshold = len(pf_filtered_df)
kf_count_above_threshold = len(kf_filtered_df)

# Calculate cumulative sum of MSE values for PF and KF
pf_df['PF CumulativeSum'] = pf_df['MSE'].cumsum()
kf_df['KF CumulativeSum'] = kf_df['MSE'].cumsum()

# Print the filtered DataFrame for PF and KF
print("PF Filtered DataFrame:")
print(pf_filtered_df)
print("KF Filtered DataFrame:")
print(kf_filtered_df)

# Statistical hypothesis test (e.g., t-test) to compare PF and KF MSE
from scipy.stats import ttest_ind

# Perform t-test
t_stat, p_value = ttest_ind(pf_df['MSE'], kf_df['MSE'])

# Effect Size Measures
effect_size = (pf_mean_mse - kf_mean_mse) / np.sqrt((pf_std_mse**2 + kf_std_mse**2) / 2)

# Print the results of the t-test
print("\nT-Test Results:")
print("T-Statistic:", t_stat)
print("P-Value:", p_value)
print("Effect Size (Cohen's d):", effect_size)

# Determine which filter is better
if pf_mean_mse < kf_mean_mse and pf_median_mse < kf_median_mse and pf_std_mse < kf_std_mse:
    print("Particle Filter performs better.")
elif kf_mean_mse < pf_mean_mse and kf_median_mse < pf_median_mse and kf_std_mse < pf_std_mse:
    print("Kalman Filter performs better.")
else:
    print("The performance is comparable.")
    
# Calculate the ratio of the MSEs
mse_ratio = kf_mean_mse / pf_mean_mse

# Display the ratio
if mse_ratio > 1:
    print(f"The Particle Filter performs {mse_ratio:.2f} times better than the Kalman Filter.")
else:
    print(f"The Kalman Filter performs {1/mse_ratio:.2f} times better than the Particle Filter.")


# In[89]:


# Read the data from the CSV file
data = pd.read_csv(r'C:\Users\foser\Downloads\robot_hand\data\data.csv')

# Calculate the differences between the filtered and unfiltered angles for each filter
data['PF Angle Diff (rad)'] = data['PF Angle (rad)'] - data['Unfiltered Angle (rad)']
data['KF Angle Diff (rad)'] = data['KF Angle (rad)'] - data['Unfiltered Angle (rad)']

# Calculate the absolute differences
data['PF Angle Diff Abs (rad)'] = data['PF Angle Diff (rad)'].abs()
data['KF Angle Diff Abs (rad)'] = data['KF Angle Diff (rad)'].abs()

# Plotting the changes in angles over iterations
fig, ax = plt.subplots(figsize=(12, 6))
data.plot(x='Iteration', y='PF Angle (rad)', secondary_y=True, ax=ax, label='PF Angle', color='red', alpha=0.5)
data.plot(x='Iteration', y='KF Angle (rad)', secondary_y=True, ax=ax, label='KF Angle', color='green', alpha=0.5)
data.plot(x='Iteration', y='Unfiltered Angle (rad)', ax=ax, label='Unfiltered Angle', color='blue', alpha=0.5)
ax.set_xlabel('Iteration')
ax.set_ylabel('Angle (rad)')
plt.title('Changes in Angles over Iterations')
plt.show()


# Plotting the absolute differences in angles over iterations
fig, ax = plt.subplots(figsize=(12, 6))
data.plot(x='Iteration', y='PF Angle Diff Abs (rad)', ax=ax, label='PF Angle Diff Abs', color='red', alpha=0.5)
data.plot(x='Iteration', y='KF Angle Diff Abs (rad)', secondary_y=True, ax=ax, label='KF Angle Diff Abs', color='green', alpha=0.5)
ax.set_xlabel('Iteration')
ax.set_ylabel('Absolute Angle Difference (rad)')
plt.title('Absolute Differences in Angles over Iterations')
plt.show()


# In[94]:


# Group the data by index
grouped_data = data.groupby('Index')

# Line plots to visualize the angle changes with iteration for each index
for index, group in grouped_data:
    plt.figure(figsize=(8, 6))
    plt.plot(group['Iteration'], group['Unfiltered Angle (rad)'], label='Unfiltered')
    plt.plot(group['Iteration'], group['PF Angle (rad)'], label='PF')
    plt.plot(group['Iteration'], group['KF Angle (rad)'], label='KF')
    plt.xlabel('Iteration')
    plt.ylabel('Angle (rad)')
    plt.title(f'Angle Changes - Index {index}')
    plt.legend()
    plt.show()

# Hypothesis testing (T-Test) for each joint index
for index, group in grouped_data:
    t_statistic, p_value = stats.ttest_rel(group['PF Angle (rad)'], group['KF Angle (rad)'])
    print(f"T-Test Results - Index {index}:")
    print("T-Statistic:", t_statistic)
    print("P-Value:", p_value)
    print()

# Effect Size (Cohen's d) for each joint index
for index, group in grouped_data:
    effect_size = (group['PF Angle (rad)'].mean() - group['KF Angle (rad)'].mean()) / np.sqrt(
        (group['PF Angle (rad)'].std() ** 2 + group['KF Angle (rad)'].std() ** 2) / 2
    )
    print(f"Effect Size (Cohen's d) - Index {index}: {effect_size}")
    print()


# In[ ]:





import json
import numpy as np
import matplotlib.pyplot as plt
import pickle

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

import random

# Function to apply min-max scaling to a list of values
def min_max_scale(values):
    min_val = min(values)
    max_val = max(values)
    # Avoid division by zero
    if max_val - min_val == 0:
        return [1 for _ in values]  # If all values are the same, set them to 1
    return [(v - min_val) / (max_val - min_val) for v in values]

path = '/home/ubuntu/Multimodal-Uncertainty-Quantification/runs/llava_gqa_yes_gsam_grounding_random_100/uncertainty/uncertainty_random_100.pkl'
with open(path, 'rb') as f:
    results = pickle.load(f)
    
filtered_results = {key: value for key, value in results.items() 
                    if 'uncertainty_from_entropy' in value and 'uncertainty_from_grounding' in value and 'accuracy' in value}

# Extract the individual values for each metric
entropy_values = [value["uncertainty_from_entropy"] for value in filtered_results.values()]
grounding_values = [value["uncertainty_from_grounding"] for value in filtered_results.values()]
accuracy_values = [value["accuracy"] for value in filtered_results.values()]

scaled_entropy = min_max_scale(entropy_values)
# scaled_grounding = min_max_scale(grounding_values)
# scaled_accuracy = min_max_scale(accuracy_values)

# Step 1: Compute confidence values for the whole dataset
confidence_from_entropy = 1 - np.array(scaled_entropy)
confidence_from_grounding = 1 - np.array(grounding_values)

# Step 2: Calculate the feature (difference between confidence_from_entropy and confidence_from_grounding)

# X = np.column_stack((confidence_from_entropy, confidence_from_grounding))
X = np.array(confidence_from_grounding).reshape(-1, 1)
y = np.array(accuracy_values)  # Ground truth accuracy values

# # Step 3: Select 20% of the dataset for regression
# n_samples = len(X)
# n_val = int(0.25 * n_samples)  # 20% of the total samples
# X_val = X[:n_val]  # Use first 20% of the data
# y_val = y[:n_val]  # Corresponding accuracy values for the first 20%

# n_samples = len(X)
# n_val = int(0.20 * n_samples)  # 20% of the total samples
# indices = random.sample(range(n_samples), n_val)

# # Use the indices to extract the random 20% of the data
# X_val = [X[i] for i in indices]
# y_val = [y[i] for i in indices]

# # Step 4: Fit linear regression on the selected 20% of data to find the optimal alpha
# model = LinearRegression()
# model.fit(X_val, y_val)  # Only use the 20% subset

# # Extract the learned alpha
# # alpha = model.coef_[0]
# # print(f"Optimal alpha found via regression: {alpha}")
# alpha_entropy, alpha_grounding = model.coef_

# print(f"Optimal alpha for confidence_from_entropy: {alpha_entropy}")
# print(f"Optimal alpha for confidence_from_grounding: {alpha_grounding}")

# new_confidence_values = alpha_entropy * (1 - np.array(scaled_entropy)) + alpha_grounding*(1 - np.array(scaled_grounding))
# scaled_new_confidence_values = min_max_scale(accuracy_values)

# # Create a new dictionary with the scaled values
# scaled_results = {}
# for i, key in enumerate(filtered_results.keys()):
#     scaled_results[key] = {
#         "entropy": scaled_entropy[i],
#         "grounding": scaled_grounding[i],
#         'confidence_from_entropy': 1 - scaled_entropy[i],
#         'confidence_from_grounding': 1 - scaled_grounding[i],
#         # 'confidence_from_entropy_grounding': alpha_entropy * (1 - scaled_entropy[i]) + alpha_grounding*(1 - scaled_grounding[i]),
#         'confidence_from_entropy_grounding': scaled_new_confidence_values[i],
#         "accuracy": accuracy_values[i]
#     }

# # Step 1: Select 20% random validation data
# n_samples = len(X)
# n_val = int(0.40 * n_samples)  # 20% of the total samples
# indices = random.sample(range(n_samples), n_val)

# # Use the indices to extract the random 20% of the data
# X_val = [X[i] for i in indices]
# y_val = [y[i] for i in indices]

# # Step 2: Remove validation samples from X, y, scaled_entropy, and scaled_grounding
# X_train = [X[i] for i in range(n_samples) if i not in indices]
# y_train = [y[i] for i in range(n_samples) if i not in indices]
# scaled_entropy_train = [scaled_entropy[i] for i in range(n_samples) if i not in indices]
# scaled_grounding_train = [scaled_grounding[i] for i in range(n_samples) if i not in indices]
# accuracy_train = [accuracy_values[i] for i in range(n_samples) if i not in indices]

# # Step 3: Fit linear regression on the selected 20% of data to find the optimal alpha
# model = LinearRegression()
# model.fit(X_val, y_val)  # Only use the 20% subset for regression

# # Extract the learned alpha
# alpha_entropy, alpha_grounding = model.coef_

# print(f"Optimal alpha for confidence_from_entropy: {alpha_entropy}")
# print(f"Optimal alpha for confidence_from_grounding: {alpha_grounding}")

# # Step 4: Compute new confidence values using the trained model
# new_confidence_values = alpha_entropy * (1 - np.array(scaled_entropy_train)) + alpha_grounding * (1 - np.array(scaled_grounding_train))

# # Assuming 'min_max_scale' is a function defined elsewhere in your code
# scaled_new_confidence_values = min_max_scale(new_confidence_values)

# # Step 5: Create a new dictionary with the scaled values
# scaled_results = {}
# for i, key in enumerate(range(len(X_train))):
#     if i not in indices:  # Only include non-validation samples in scaled_results
#         scaled_results[key] = {
#             "entropy": scaled_entropy_train[i],
#             "grounding": scaled_grounding_train[i],
#             'confidence_from_entropy': 1 - scaled_entropy_train[i],
#             'confidence_from_grounding': 1 - scaled_grounding_train[i],
#             'confidence_from_entropy_grounding': scaled_new_confidence_values[i],
#             "accuracy": accuracy_train[i]
#         }

# Set the random seed for reproducibility
random_seed = 42
random.seed(random_seed)
np.random.seed(random_seed)

# Step 1: Select 20% random validation data
n_samples = len(X)
n_val = int(0.20 * n_samples)  # 20% of the total samples
indices = random.sample(range(n_samples), n_val)

# Convert X and y to numpy arrays (if not already)
X = np.array(X)
y = np.array(y)

# Use the indices to extract the random 20% of the data
X_val = X[indices]
y_val = y[indices]

# Step 2: Remove validation samples from X, y, scaled_entropy, and scaled_grounding for the training set
X_train = np.delete(X, indices, axis=0)
y_train = np.delete(y, indices, axis=0)
scaled_entropy_train = np.delete(scaled_entropy, indices, axis=0)
scaled_grounding_train = np.delete(grounding_values, indices, axis=0)
accuracy_train = np.delete(accuracy_values, indices, axis=0)

# scaler = StandardScaler()
# X_val_scaled = scaler.fit_transform(X_val)

# Step 3: Fit linear regression on the selected 20% of data to find the optimal alpha
# model = LinearRegression()
model = Ridge(alpha=1.0)  # alpha controls the strength of regularization
model.fit(X_val, y_val)  # Only use the 20% subset for regression

# Extract the learned alpha
alpha_grounding = model.coef_

# print(f"Optimal alpha for confidence_from_entropy: {alpha_entropy}")
print(f"Optimal alpha for confidence_from_grounding: {alpha_grounding}")

# Step 4: Compute new confidence values using the trained model
# scaled_entropy_train = scaler.fit_transform(scaled_entropy_train)
# X_train_scaled = scaler.fit_transform(X_train)
# new_confidence_values = alpha_entropy * (1 - np.array(X_train_scaled[0,:])) + alpha_grounding * (1 - np.array(X_train_scaled[1,:]))

new_confidence_values = alpha_grounding*X_train

# Assuming 'min_max_scale' is a function defined elsewhere in your code
scaled_new_confidence_values = min_max_scale(new_confidence_values)

# Step 5: Create a new dictionary with the scaled values
scaled_results = {}
for i, key in enumerate(range(len(X_train))):
    scaled_results[key] = {
        "entropy": scaled_entropy_train[i],
        "grounding": scaled_grounding_train[i],
        'confidence_from_entropy': 1 - scaled_entropy_train[i],
        'confidence_from_grounding': 1 - scaled_grounding_train[i],
        'confidence_from_entropy_grounding': scaled_new_confidence_values[i],
        "accuracy": accuracy_train[i]
    }

    
# Function to plot the reliability diagram for a given confidence type
def plot_reliability_diagram(confidences, accuracies, confidence_type, save_path):
    bins = np.linspace(0, 1, 11)  # Create 10 bins between 0 and 1
    bin_indices = np.digitize(confidences, bins) - 1  # Get bin indices for each confidence value

    # Calculate average accuracy for each bin
    avg_accuracies = []
    for i in range(len(bins) - 1):
        bin_mask = bin_indices == i
        if bin_mask.any():
            avg_accuracy = np.mean(accuracies[bin_mask.flatten()])
            avg_accuracies.append(avg_accuracy)
        else:
            avg_accuracies.append(np.nan)  # If no data in bin, append NaN

    # Handle NaN values (e.g., for empty bins)
    avg_accuracies = np.array(avg_accuracies)
    valid_bins = ~np.isnan(avg_accuracies)

    # Get bin centers for plotting
    bin_centers = (bins[:-1] + bins[1:]) / 2

    # Plot the reliability diagram
    plt.plot(bin_centers[valid_bins], avg_accuracies[valid_bins], marker='o', label=f'Reliability ({confidence_type})')
    plt.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Perfect Calibration')  # Diagonal line for perfect reliability
    plt.xlabel('Confidence')
    plt.ylabel('Accuracy')
    plt.title(f'Reliability Diagram ({confidence_type})')
    plt.legend()
    plt.grid(True)

    # Save the plot
    plt.savefig(save_path)
    plt.clf()  # Clear the current plot for the next one

# Extract accuracy values from the dictionary
accuracies = np.array([v["accuracy"] for v in scaled_results.values()])

# Extract the three types of confidences
confidence_from_entropy = np.array([v["confidence_from_entropy"] for v in scaled_results.values()])
confidence_from_grounding = np.array([v["confidence_from_grounding"] for v in scaled_results.values()])
confidence_from_entropy_grounding = np.array([v["confidence_from_entropy_grounding"] for v in scaled_results.values()])

# Define file names to save the plots
files_to_save = {
    "Confidence from Entropy": "reliability_confidence_from_entropy_random_100_calib_random_samples.png",
    "Confidence from Grounding": "reliability_confidence_from_grounding_random_100_calib_random_samples.png",
    "Confidence from Entropy + Grounding": "reliability_confidence_from_entropy_grounding_random_100_calib_random_samples.png"
}

# Plot and save the three reliability diagrams
plot_reliability_diagram(confidence_from_entropy, accuracies, "Confidence from Entropy", files_to_save["Confidence from Entropy"])
plot_reliability_diagram(confidence_from_grounding, accuracies, "Confidence from Grounding", files_to_save["Confidence from Grounding"])
plot_reliability_diagram(confidence_from_entropy_grounding, accuracies, "Confidence from Entropy + Grounding", files_to_save["Confidence from Entropy + Grounding"])

print(f"Reliability diagrams saved as {', '.join(files_to_save.values())}")
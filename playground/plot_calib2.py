import json
import numpy as np
import matplotlib.pyplot as plt
import pickle
from sklearn.linear_model import Ridge
import random
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
import seaborn as sns

# Function to apply min-max scaling to a list of values
def min_max_scale(values):
    min_val = min(values)
    max_val = max(values)
    if max_val - min_val == 0:
        return [1 for _ in values]  # If all values are the same, set them to 1
    return [(v - min_val) / (max_val - min_val) for v in values]

# Load uncertainty data from a pickle file
def load_uncertainty_data(filepath):
    with open(filepath, 'rb') as f:
        return pickle.load(f)

# Filter out results that do not contain required keys
def filter_results(results):
    return {key: value for key, value in results.items() 
            if 'uncertainty_from_entropy' in value and 'uncertainty_from_grounding' in value and 'accuracy' in value}

# Extract the necessary metrics (entropy, grounding, accuracy)
def extract_metrics(filtered_results):
    entropy_values = [value["uncertainty_from_entropy"] for value in filtered_results.values()]
    grounding_values = [value["uncertainty_from_grounding"] for value in filtered_results.values()]
    accuracy_values = [value["accuracy"] for value in filtered_results.values()]
    return entropy_values, grounding_values, accuracy_values

# Prepare the dataset for training and validation
def split_dataset(X, y, confidence_from_entropy, confidence_from_grounding, accuracy_values, val_percent=0.30, seed=None):
    # Set random seed for reproducibility
    # print('Random Seed', seed)
    if seed is None:
        seed = random.randint(0, 10000)  # Generate a random seed if not provided
    print(f"Using random seed: {seed}")
    random.seed(seed)
    np.random.seed(seed)
    
    n_samples = len(X)
    n_val = int(val_percent * n_samples)
    indices = random.sample(range(n_samples), n_val)
    
    X_val = X[indices]
    y_val = y[indices]
    
    X_train = np.delete(X, indices, axis=0)
    y_train = np.delete(y, indices, axis=0)
    entropy_train = np.delete(confidence_from_entropy, indices, axis=0)
    grounding_train = np.delete(confidence_from_grounding, indices, axis=0)
    accuracy_train = np.delete(accuracy_values, indices, axis=0)
    
    return X_train, y_train, X_val, y_val, entropy_train, grounding_train, accuracy_train

# Fit polynomial regression and return coefficients
def fit_polynomial_regression(X_val, y_val, degree=6, alpha=3.0):
    model = make_pipeline(PolynomialFeatures(degree=degree), Ridge(alpha=alpha))
    model.fit(X_val, y_val)
    return model.named_steps['ridge'].coef_, model.named_steps['polynomialfeatures']

# Calculate the new confidence values
def calculate_confidence_values(X_train, alphas, poly_features):
    X_train_poly = poly_features.transform(X_train)
    new_confidence_values = np.dot(X_train_poly, alphas)
    return min_max_scale(new_confidence_values)

# Create a dictionary with scaled results
def create_scaled_results(X_train, entropy_train, grounding_train, new_confidence_values, accuracy_train):
    scaled_results = {}
    for i in range(len(X_train)):
        scaled_results[i] = {
            "entropy": entropy_train[i],
            "grounding": grounding_train[i],
            'confidence_from_entropy': 1 - entropy_train[i],
            'confidence_from_grounding': 1 - grounding_train[i],
            'confidence_from_entropy_grounding': new_confidence_values[i],
            'confidence_from_entropy_grounding_uncalibrated': 0.5 * (entropy_train[i] + grounding_train[i]),
            "accuracy": accuracy_train[i]
        }
    return scaled_results

def plot_reliability_diagram(confidences, accuracies, confidence_type, save_path):
    # Set the style to a built-in Matplotlib style
    plt.style.use('ggplot')

    # Create the figure and axis
    fig, ax = plt.subplots(figsize=(10, 8))

    # Calculate bin statistics
    bins = np.linspace(0, 1, 11)
    bin_indices = np.digitize(confidences, bins) - 1

    avg_accuracies = []
    bin_centers = []
    for i in range(len(bins) - 1):
        bin_mask = bin_indices == i
        if bin_mask.any():
            avg_accuracy = np.mean(accuracies[bin_mask])
            avg_accuracies.append(avg_accuracy)
            bin_centers.append((bins[i] + bins[i + 1]) / 2)
        else:
            print(f"No data found in bin {i}")
    
    avg_accuracies = np.array(avg_accuracies)
    bin_centers = np.array(bin_centers)

    # Plot the reliability curve
    ax.plot(bin_centers, avg_accuracies, marker='o', linestyle='-', linewidth=2, 
            markersize=8, label=f'Reliability ({confidence_type})')

    # Plot the perfect calibration line
    ax.plot([0, 1], [0, 1], linestyle='--', color='gray', linewidth=2, label='Perfect Calibration')

    # Fill the area between the curves
    ax.fill_between(bin_centers, bin_centers, avg_accuracies, alpha=0.2)

    # Set labels and title
    ax.set_xlabel('Confidence', fontsize=14)
    ax.set_ylabel('Accuracy', fontsize=14)
    ax.set_title(f'Reliability Diagram ({confidence_type})', fontsize=16, fontweight='bold')

    # Customize the grid
    ax.grid(True, linestyle=':', alpha=0.7)

    # Customize the legend
    ax.legend(fontsize=12, loc='lower right')

    # Set the aspect ratio to 'equal' for a square plot
    ax.set_aspect('equal')

    # Add textbox with statistics
    ece = np.mean(np.abs(avg_accuracies - bin_centers))
    mce = np.max(np.abs(avg_accuracies - bin_centers))
    stats_text = f'ECE: {ece:.3f}\nMCE: {mce:.3f}'
    ax.text(0.05, 0.95, stats_text, transform=ax.transAxes, fontsize=12,
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    # Customize ticks
    ax.tick_params(axis='both', which='major', labelsize=12)

    # Tight layout and save
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    
# Main function to run the pipeline
def run_pipeline(filepath, save_path_prefix, use_grounding_only = True):
    # Load and filter data
    results = load_uncertainty_data(filepath)
    filtered_results = filter_results(results)
    
    # Extract metrics
    entropy_values, grounding_values, accuracy_values = extract_metrics(filtered_results)
    
    # Scale entropy and grounding values
    confidence_from_entropy = 1 - np.array(min_max_scale(entropy_values))
    confidence_from_grounding = 1 - np.array(min_max_scale(grounding_values))
    
    # Prepare data for regression (use both confidence_from_entropy and confidence_from_grounding as features)
    if not use_grounding_only:
        X = np.column_stack((confidence_from_entropy, confidence_from_grounding))
    else:
        X = confidence_from_grounding.reshape(-1, 1)
    y = np.array(accuracy_values)
    
    # Split dataset into training and validation
    X_train, y_train, X_val, y_val, entropy_train, grounding_train, accuracy_train = split_dataset(
        X, y, confidence_from_entropy, confidence_from_grounding, accuracy_values, seed=122
    )
    
    # Fit polynomial regression and compute new confidence values
    alphas, poly_features = fit_polynomial_regression(X_val, y_val)
    new_confidence_values = calculate_confidence_values(X_train, alphas, poly_features)
    # uncalibrated_confidence = 0.5 * (confidence_from_entropy + confidence_from_grounding)
    
    # Create scaled results
    scaled_results = create_scaled_results(X_train, entropy_train, grounding_train, new_confidence_values, accuracy_train)
    
    # Extract accuracy values and confidences
    accuracies = np.array([v["accuracy"] for v in scaled_results.values()])
    confidence_from_entropy = np.array([v["confidence_from_entropy"] for v in scaled_results.values()])
    confidence_from_grounding = np.array([v["confidence_from_grounding"] for v in scaled_results.values()])
    confidence_from_entropy_grounding = np.array([v["confidence_from_entropy_grounding"] for v in scaled_results.values()])
    confidence_from_entropy_grounding_uncalibrated = np.array([v["confidence_from_entropy_grounding_uncalibrated"] for v in scaled_results.values()])
    
    # Plot reliability diagrams
    plot_reliability_diagram(confidence_from_entropy, accuracies, "Confidence from Entropy", f"{save_path_prefix}_entropy.png")
    plot_reliability_diagram(confidence_from_grounding, accuracies, "Confidence from Grounding", f"{save_path_prefix}_grounding.png")
    plot_reliability_diagram(confidence_from_entropy_grounding, accuracies, "Confidence from Entropy + Grounding", f"{save_path_prefix}_combined.png")
    plot_reliability_diagram(confidence_from_entropy_grounding_uncalibrated, accuracies, "Confidence from Entropy + Grounding (Uncalibrated)", f"{save_path_prefix}_uncalibrated.png")

    print(f"Reliability diagrams saved with prefix {save_path_prefix}")

# Run the pipeline
run_pipeline('/home/ubuntu/Multimodal-Uncertainty-Quantification/runs/llava_gqa_yes_gsam_grounding_random_100/uncertainty/uncertainty_random_100.pkl', 
             "reliability_diagram_random_samples", use_grounding_only=False)

# run_pipeline('/home/ubuntu/Multimodal-Uncertainty-Quantification/runs/llava_gqa_yes_gsam_grounding/uncertainty/uncertainty.pkl', "reliability_diagram_true_samples", use_grounding_only=False)